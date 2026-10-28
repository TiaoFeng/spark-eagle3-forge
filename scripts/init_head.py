#!/usr/bin/env python3
"""Build a v8-initialised head whose lm_head rows follow a new draft vocabulary.

Everything except lm_head is copied verbatim from the published v8 checkpoint.
lm_head rows are re-indexed: a token that exists in both vocabularies keeps its
trained row, and only genuinely new tokens (mostly CJK) start from small random
weights. That keeps the code/English behaviour of v8 while giving the new
vocabulary a place to grow.

Usage:
  python scripts/init_head.py --v8-dir datasets/v8 \
      --d2t datasets/vocab/d2t_v1_32k.i32 --out checkpoints/head_init.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from spark_eagle3.model import Eagle3Head, config_from_json  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8-dir", default="datasets/v8")
    ap.add_argument("--d2t", required=True, help="raw int32 target ids of the new vocabulary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    v8 = pathlib.Path(args.v8_dir)
    cfg_json = json.loads((v8 / "config.json").read_text())
    cfg = config_from_json(cfg_json)

    sd = load_file(str(v8 / "model.safetensors"))
    model = Eagle3Head(cfg)
    model_keys = set(model.state_dict().keys())
    extra_keys = sorted(set(sd) - model_keys)
    missing_keys = sorted(model_keys - set(sd))
    if missing_keys:
        sys.exit("v8 is missing model parameters: %s" % missing_keys)
    model.load_state_dict({k: v for k, v in sd.items() if k in model_keys}, strict=True)
    print("v8 loaded strict: %d parameters, non-parameter tensors %s"
          % (len(model_keys), extra_keys), flush=True)

    # old draft index -> target token id (delta-encoded in the safetensors)
    d2t_old = sd["d2t"].to(torch.int64) + torch.arange(sd["d2t"].numel(), dtype=torch.int64)
    old_row = {int(t): i for i, t in enumerate(d2t_old.tolist())}

    d2t_new = np.fromfile(args.d2t, dtype=np.int32).astype(np.int64)
    if d2t_new.size != cfg["draft_vocab_size"]:
        sys.exit("d2t size %d != draft_vocab_size %d"
                 % (d2t_new.size, cfg["draft_vocab_size"]))
    if len(set(d2t_new.tolist())) != d2t_new.size:
        sys.exit("d2t contains duplicate target ids")

    g = torch.Generator().manual_seed(args.seed)
    old_head = sd["lm_head.weight"]
    std = float(old_head.std())
    new_head = torch.empty_like(old_head)
    kept = 0
    for i, tid in enumerate(d2t_new.tolist()):
        j = old_row.get(tid)
        if j is None:
            new_head[i] = torch.randn(old_head.shape[1], generator=g) * (std * 0.5)
        else:
            new_head[i] = old_head[j]
            kept += 1

    out_sd = {k: v for k, v in sd.items() if k in model_keys}
    out_sd["lm_head.weight"] = new_head
    out_sd["d2t"] = torch.from_numpy(d2t_new)
    # t2d from v8 describes the old vocabulary and would be wrong here; the
    # exporter regenerates it from d2t if anything needs it.

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out_sd, "config": cfg_json,
                "d2t_sha256": sha256_file(pathlib.Path(args.d2t))}, str(out))

    # round-trip: the new weights must load strictly into the same module
    check = Eagle3Head(cfg)
    check.load_state_dict({k: v for k, v in out_sd.items() if k in model_keys}, strict=True)

    meta = {
        "out": str(out),
        "draft_vocab_size": int(d2t_new.size),
        "tokens_kept_from_v8": kept,
        "tokens_newly_initialised": int(d2t_new.size - kept),
        "lm_head_row_std": std,
        "d2t_sha256": sha256_file(pathlib.Path(args.d2t)),
        "v8_safetensors_sha256": sha256_file(v8 / "model.safetensors"),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
