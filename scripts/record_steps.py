#!/usr/bin/env python3
"""Record greedy generation steps for draft-vocabulary scoring.

Writes int32 rows ``[workload index, generated_token, target_argmax, is_cjk]``
over the generated region of every prompt in a rendered prompt directory. The
format is the one ``scripts/eval_vocab.py`` consumes, so a candidate vocabulary
can be scored on the actual workload without touching the draft head.

The replay is greedy, exactly like ``spark_eagle3.eval_p0``: the target decodes
by argmax, so the recorded token is also the token the runtime would generate.
Prompts are prefilled in chunks and generated one token at a time, which keeps
the logits buffer small enough for long-context prompts.

Usage:
  python scripts/record_steps.py --model M.gguf \
      --prompt-dir datasets/prompts/nlp --out logs/nlp/steps.i32
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from spark_eagle3.feats import FeatureStore  # noqa: E402
from spark_eagle3.llamacpp import Shim  # noqa: E402

WORKLOAD_ORDER = ["chat", "zh2en", "en2zh", "sumzh", "sumen", "org", "longctx"]


def is_cjk_piece(piece: bytes) -> bool:
    return any(0xE4 <= b <= 0xE9 for b in piece)


def workload_labels(manifest: dict) -> list[str]:
    # new manifests carry "workload"; old rendered prompts carry "register"
    seen = {rec.get("workload") or rec.get("register", name) for name, rec in manifest.items()}
    known = [r for r in WORKLOAD_ORDER if r in seen]
    return known + sorted(r for r in seen if r not in WORKLOAD_ORDER)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-dir", default="datasets/prompts/nlp")
    ap.add_argument("--out", default="logs/nlp/steps.i32")
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--n-ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="2,18,33")
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    prompt_dir = pathlib.Path(args.prompt_dir)
    manifest = json.loads((prompt_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = workload_labels(manifest)
    label_index = {name: i for i, name in enumerate(labels)}

    rows: list[tuple[int, int, int, int]] = []
    counts = {name: 0 for name in labels}

    with Shim(args.model, n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx,
              n_batch=args.chunk) as shim:
        shim.set_layers(layers)
        with FeatureStore(args.n_ctx * 20000, len(layers), shim.n_embd) as store:
            for name in sorted(manifest):
                rec = manifest[name]
                wl = label_index[rec.get("workload") or rec.get("register", name)]
                ids = shim.tokenize(pathlib.Path(rec["path"]).read_text(encoding="utf-8"),
                                    add_special=False)
                n_prompt = int(ids.size)
                n_total = min(n_prompt + args.n_gen, args.n_ctx)
                if n_total <= n_prompt + 1:
                    print("%-18s skipped (too long)" % name, flush=True)
                    continue

                toks = np.zeros(n_total, dtype=np.int32)
                toks[:n_prompt] = ids[:n_prompt]
                argmax = np.empty(n_total, dtype=np.int32)
                shim.reset()
                store.require(n_total)

                # prefill everything but the last prompt token, then decode that
                # token with logits so the first generated token is known
                for start in range(0, n_prompt - 1, args.chunk):
                    take = min(args.chunk, n_prompt - 1 - start)
                    shim.decode(toks[start:start + take], start_pos=start,
                                feat_ptr=store.row_ptr(start), want_logits=False)
                a = shim.decode(toks[n_prompt - 1:n_prompt], start_pos=n_prompt - 1,
                                feat_ptr=store.row_ptr(n_prompt - 1), want_logits=True)
                argmax[n_prompt - 1] = a[0]

                n = n_prompt
                nxt = int(argmax[n_prompt - 1])
                while n < n_total:
                    toks[n] = nxt
                    a = shim.decode(toks[n:n + 1], start_pos=n,
                                    feat_ptr=store.row_ptr(n), want_logits=True)
                    argmax[n] = a[0]
                    n += 1
                    if shim.is_eog(nxt):
                        break
                    nxt = int(argmax[n - 1])

                for p in range(n_prompt - 1, n - 1):
                    tok = int(toks[p + 1])
                    arg = int(argmax[p + 1])
                    cjk = 1 if is_cjk_piece(shim.token_to_piece(tok)) else 0
                    rows.append((wl, tok, arg, cjk))
                    counts[labels[wl]] += 1
                print("%-18s workload=%-8s prompt=%5d generated=%3d"
                      % (name, labels[wl], n_prompt, n - n_prompt), flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rows, dtype=np.int32).reshape(-1, 4)
    arr.tofile(str(out))
    meta = {
        "prompt_dir": str(prompt_dir),
        "n_gen": args.n_gen,
        "labels": labels,
        "rows": int(arr.shape[0]),
        "per_workload": counts,
        "cjk_share": float(arr[:, 3].mean()) if len(arr) else 0.0,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
