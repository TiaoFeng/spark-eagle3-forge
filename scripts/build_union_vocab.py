#!/usr/bin/env python3
"""Build a bilingual union vocabulary on top of an existing d2t.

The base ids keep their positions, so a checkpoint trained on the base
vocabulary keeps every lm_head row; only the appended rows are new. New ids are
the most frequent target tokens that the base vocabulary is missing, mined from
an English corpus and, optionally, from recorded generation steps (the output
tokens and the target argmax of the actual workload).

Usage:
  python scripts/build_union_vocab.py --model M.gguf \
      --base datasets/vocab/d2t_v1_32k.i32 --out datasets/vocab/d2t_v2_48k \
      --vocab-size 48000 \
      --english datasets/raw/wikipedia_en/train-00000-of-00041.parquet \
      --steps logs/nlp/steps.i32
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from spark_eagle3.llamacpp import Shim  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parent.parent


def is_cjk_piece(piece: bytes) -> bool:
    return any(0xE4 <= b <= 0xE9 for b in piece)


def iter_parquet(path: pathlib.Path, budget: int):
    import pyarrow.parquet as pq
    used = 0
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=512, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if not text:
                continue
            yield text
            used += len(text)
            if used >= budget:
                return


def iter_text(path: pathlib.Path, budget: int):
    text = path.read_text(encoding="utf-8", errors="ignore")
    for start in range(0, min(len(text), budget), 1 << 20):
        yield text[start:start + (1 << 20)]


def iter_english(paths, budget: int):
    for root in paths:
        root = pathlib.Path(root)
        if not root.exists():
            print("warn: %s does not exist" % root, flush=True)
            continue
        files = sorted(root.rglob("*.parquet")) if root.is_dir() else [root]
        if root.is_dir():
            files += sorted(root.rglob("*.txt")) + sorted(root.rglob("*.jsonl"))
        for path in files:
            print("  %s" % path, flush=True)
            if path.suffix == ".parquet":
                yield from iter_parquet(path, budget)
            elif path.suffix in (".txt", ".jsonl", ".md"):
                yield from iter_text(path, budget)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="datasets/vocab/d2t_v1_32k.i32")
    ap.add_argument("--out", default="datasets/vocab/d2t_v2_48k")
    ap.add_argument("--vocab-size", type=int, default=48000)
    ap.add_argument("--english", action="append", default=[],
                    help="English parquet/text files or directories (repeatable)")
    ap.add_argument("--steps", default="", help="optional steps.i32 to weight real output tokens")
    ap.add_argument("--step-weight", type=int, default=50)
    ap.add_argument("--chars-per-source", type=int, default=20_000_000)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    args = ap.parse_args()

    base = np.fromfile(args.base, dtype=np.int32).astype(np.int64)
    if len(set(base.tolist())) != base.size:
        sys.exit("base d2t contains duplicate ids")
    if base.size >= args.vocab_size:
        sys.exit("base size %d >= vocab size %d" % (base.size, args.vocab_size))
    base_set = set(int(x) for x in base.tolist())

    english = args.english or ["datasets/raw/wikipedia_en"]
    steps_path = pathlib.Path(args.steps) if args.steps else pathlib.Path("logs/nlp/steps.i32")

    with Shim(args.model, n_gpu_layers=args.n_gpu_layers, n_ctx=64, n_batch=64) as shim:
        counts: collections.Counter = collections.Counter()
        chars = 0
        for text in iter_english(english, args.chars_per_source):
            chars += len(text)
            ids = shim.tokenize(text, add_special=False)
            if ids.size:
                counts.update(ids.tolist())
        print("english chars=%d tokens=%d distinct=%d"
              % (chars, sum(counts.values()), len(counts)), flush=True)

        step_rows = 0
        if steps_path.exists():
            steps = np.fromfile(str(steps_path), dtype=np.int32).reshape(-1, 4)
            step_rows = int(steps.shape[0])
            for tok, arg in zip(steps[:, 1].tolist(), steps[:, 2].tolist()):
                counts[tok] += args.step_weight
                counts[arg] += args.step_weight
            print("steps=%s rows=%d weight=%d"
                  % (steps_path, step_rows, args.step_weight), flush=True)
        else:
            steps = None
            print("warn: no steps at %s" % steps_path, flush=True)

        forced = set()
        try:
            from gguf import GGUFReader
            reader = GGUFReader(args.model)
            for key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
                field = reader.fields.get(key)
                if field is not None:
                    forced.add(int(field.contents()))
        except Exception as exc:  # noqa: BLE001
            print("warn: could not read special ids: %s" % exc, flush=True)

        forced_new = sorted(t for t in forced if t not in base_set)
        keep = args.vocab_size - int(base.size) - len(forced_new)
        if keep <= 0:
            sys.exit("no room for new ids: base=%d forced=%d vocab=%d"
                     % (base.size, len(forced_new), args.vocab_size))
        ranked = [t for t, _c in counts.most_common() if t not in base_set]
        selected = ranked[:keep]
        vocab = np.asarray(base.tolist() + selected + forced_new, dtype=np.int32)
        if len(set(vocab.tolist())) != vocab.size:
            sys.exit("union vocabulary contains duplicate ids")

        cjk_base = sum(1 for t in base.tolist() if is_cjk_piece(shim.token_to_piece(int(t))))
        cjk_new = sum(1 for t in selected if is_cjk_piece(shim.token_to_piece(int(t))))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vocab.tofile(str(out) + ".i32")

    meta = {
        "model": pathlib.Path(args.model).name,
        "base": str(args.base),
        "base_size": int(base.size),
        "vocab_size": int(vocab.size),
        "new_ids": int(len(selected) + len(forced_new)),
        "new_cjk": int(cjk_new),
        "base_cjk": int(cjk_base),
        "forced_special": forced_new,
        "english_sources": [str(p) for p in english],
        "english_chars": int(chars),
        "step_rows": step_rows,
        "step_weight": args.step_weight,
    }
    if steps is not None:
        mask = np.zeros(1 << 18, dtype=bool)
        mask[vocab] = True
        meta["step_token_coverage"] = float(mask[steps[:, 1]].mean())
        meta["step_argmax_coverage"] = float(mask[steps[:, 2]].mean())
        labels_path = steps_path.with_suffix(".json")
        if labels_path.exists():
            labels = json.loads(labels_path.read_text(encoding="utf-8")).get("labels", [])
            per = {}
            for i, label in enumerate(labels):
                sel = steps[:, 0] == i
                if sel.any():
                    per[label] = {
                        "tokens": int(sel.sum()),
                        "token_coverage": float(mask[steps[sel, 1]].mean()),
                        "argmax_coverage": float(mask[steps[sel, 2]].mean()),
                    }
            meta["per_workload"] = per

    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print("wrote %s.i32 (%d ids, %d CJK base + %d CJK new)"
          % (out, vocab.size, cjk_base, cjk_new), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
