#!/usr/bin/env python3
"""Select a draft vocabulary by token frequency.

Why: the v8 draft vocabulary is a specific corpus's high-frequency set. Measured
on the real workload it covers 1.1% of CJK tokens and 49.9% of ASCII tokens, so
the head cannot propose most of what the target actually emits. This script
builds a replacement vocabulary from text that matches the target workload
(Simplified Chinese instructions, Chinese prose) plus code, and writes the
absolute target ids to a raw int32 file that sp_feat_cli --d2t accepts for
coverage measurement.

Equal character budgets per domain keep a large prose dump from swamping the
conversational data.

Usage:
  python scripts/build_vocab.py --model M.gguf --out datasets/vocab/d2t_v1_32k \
      --vocab-size 32000 --chars-per-domain 40000000
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


def iter_jsonl_cqia(path, budget):
    used = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            parts = [rec.get("instruction") or "", rec.get("input") or "",
                     rec.get("output") or ""]
            text = "\n".join(p for p in parts if p)
            if not text:
                continue
            yield text
            used += len(text)
            if used >= budget:
                return


def iter_jsonl_firefly(path, budget):
    """firefly-train-1.1M is {"kind", "input", "target"}, one record per line."""
    used = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = "\n".join(p for p in (rec.get("input") or "",
                                         rec.get("target") or "") if p)
            if not text:
                continue
            yield text
            used += len(text)
            if used >= budget:
                return


def iter_parquet(path, budget, field="text"):
    import pyarrow.parquet as pq
    used = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=512, columns=[field]):
        for text in batch.column(0).to_pylist():
            if not text:
                continue
            yield text
            used += len(text)
            if used >= budget:
                return


def iter_code(dirs, budget):
    suffixes = {".py", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".sh", ".md", ".json", ".yaml"}
    used = 0
    for root in dirs:
        root = pathlib.Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text:
                continue
            yield text
            used += len(text)
            if used >= budget:
                return


def is_cjk_piece(piece: bytes) -> bool:
    return any(0xE4 <= b <= 0xE9 for b in piece)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. datasets/vocab/d2t_v1_32k")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--chars-per-domain", type=int, default=40_000_000)
    ap.add_argument("--raw-dir", default=str(BASE / "datasets" / "raw"))
    ap.add_argument("--code-dir", action="append", default=[],
                    help="extra directories to mine for code (repeatable)")
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    args = ap.parse_args()

    raw = pathlib.Path(args.raw_dir)
    budget = args.chars_per_domain

    sources = []
    cqia = raw / "coig_cqia" / "COIG-CQIA-full.jsonl"
    if cqia.exists():
        sources.append(("coig_cqia", lambda: iter_jsonl_cqia(cqia, budget), 1.0))
    firefly = raw / "firefly" / "firefly-train-1.1M.jsonl"
    if firefly.exists():
        sources.append(("firefly", lambda: iter_jsonl_firefly(firefly, budget), 1.0))
    wiki = raw / "wikipedia_zh" / "train-00000-of-00006.parquet"
    if wiki.exists():
        sources.append(("wikipedia_zh", lambda: iter_parquet(wiki, budget), 1.0))

    code_dirs = args.code_dir or [str(BASE)]
    sources.append(("code", lambda: iter_code(code_dirs, budget), 1.0))

    if len(sources) < 2:
        sys.exit("no corpora found under %s" % raw)

    with Shim(args.model, n_gpu_layers=args.n_gpu_layers, n_ctx=64, n_batch=64) as shim:
        counts = collections.Counter()
        per_source = {}
        for name, factory, _w in sources:
            before = sum(counts.values())
            chars = 0
            for text in factory():
                chars += len(text)
                ids = shim.tokenize(text, add_special=False)
                if ids.size:
                    counts.update(ids.tolist())
            after = sum(counts.values())
            per_source[name] = {"chars": chars, "tokens": after - before}
            print("%-14s chars=%12d tokens=%10d" % (name, chars, after - before), flush=True)

        # always keep the special tokens the model may emit
        forced = set()
        try:
            from gguf import GGUFReader
            reader = GGUFReader(args.model)
            for key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
                field = reader.fields.get(key)
                if field is not None:
                    forced.add(int(field.contents()))
        except Exception as e:  # noqa: BLE001
            print("warn: could not read special ids: %s" % e, flush=True)

        ranked = [t for t, _c in counts.most_common() if t not in forced]
        keep = args.vocab_size - len(forced)
        selected = ranked[:keep]
        vocab = selected + sorted(forced)          # frequency order first
        cjk = sum(1 for t in vocab if is_cjk_piece(shim.token_to_piece(t)))

        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(vocab, dtype=np.int32)
        arr.tofile(str(out) + ".i32")

        total = sum(counts.values())
        covered = sum(counts[t] for t in vocab)
        meta = {
            "model": pathlib.Path(args.model).name,
            "vocab_size": int(arr.size),
            "cjk_tokens": int(cjk),
            "forced_special": sorted(int(x) for x in forced),
            "sources": per_source,
            "total_tokens": int(total),
            "distinct_tokens": int(len(counts)),
            "self_coverage": covered / total if total else 0.0,
        }
        with open(str(out) + ".json", "w") as fh:
            json.dump(meta, fh, indent=2)

        print(json.dumps(meta, indent=2), flush=True)
        print("wrote %s.i32 (%d ids, %d CJK)" % (out, arr.size, cjk), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
