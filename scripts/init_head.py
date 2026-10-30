#!/usr/bin/env python3
"""Build a head whose lm_head rows follow a new draft vocabulary.

By default the source is the published v8 checkpoint; with --from-checkpoint it
is a training checkpoint (for example S2a). Everything except lm_head is copied
and lm_head rows are re-indexed: a token that exists in both vocabularies keeps
its trained row, new tokens start from small random weights. Starting from S2a
is what keeps the 32K Chinese rows when the vocabulary grows to 48K.

Usage:
  python scripts/init_head.py --v8-dir datasets/v8 \
      --d2t datasets/vocab/d2t_v1_32k.i32 --out checkpoints/head_init.pt
  python scripts/init_head.py --from-checkpoint checkpoints/s2a.pt \
      --d2t datasets/vocab/d2t_v2_48k.i32 --out checkpoints/head_init_48k.pt
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
    ap.add_argument("--from-checkpoint", default="",
                    help="training checkpoint to copy lm_head rows from instead of v8")
    ap.add_argument("--d2t", required=True, help="raw int32 target ids of the new vocabulary")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d2t_new = np.fromfile(args.d2t, dtype=np.int32).astype(np.int64)
    if d2t_new.size == 0:
        sys.exit("empty d2t")
    if len(set(d2t_new.tolist())) != d2t_new.size:
        sys.exit("d2t contains duplicate target ids")

    if args.from_checkpoint:
        blob = torch.load(args.from_checkpoint, map_location="cpu", weights_only=False)
        cfg_json = blob["config"]
        old_sd = blob["state_dict"]
        d2t_old = blob["d2t"].to(torch.int64)
        source = str(args.from_checkpoint)
    else:
        v8 = pathlib.Path(args.v8_dir)
        cfg_json = json.loads((v8 / "config.json").read_text())
        old_sd = load_file(str(v8 / "model.safetensors"))
        d2t_old = (old_sd["d2t"].to(torch.int64)
                   + torch.arange(old_sd["d2t"].numel(), dtype=torch.int64))
        source = str(v8 / "model.safetensors")

    cfg = config_from_json(cfg_json)
    cfg["draft_vocab_size"] = int(d2t_new.size)
    cfg_json = dict(cfg_json)
    cfg_json["draft_vocab_size"] = int(d2t_new.size)

    model = Eagle3Head(cfg)
    model_keys = set(model.state_dict().keys())
    old_keys = model_keys - {"lm_head.weight"}
    missing = sorted(old_keys - set(old_sd))
    if missing:
        sys.exit("source is missing model parameters: %s" % missing)
    model.load_state_dict({k: v for k, v in old_sd.items() if k in old_keys}, strict=False)
    print("source %s: %d parameters, d2t %d -> %d"
          % (source, len(old_keys), d2t_old.numel(), d2t_new.size), flush=True)

    old_row = {int(t): i for i, t in enumerate(d2t_old.tolist())}
    if len(old_row) != d2t_old.numel():
        sys.exit("source d2t contains duplicate ids")

    g = torch.Generator().manual_seed(args.seed)
    old_head = old_sd["lm_head.weight"].float()
    std = float(old_head.std())
    new_head = torch.empty((int(d2t_new.size), old_head.shape[1]), dtype=old_head.dtype)
    kept = 0
    for i, tid in enumerate(d2t_new.tolist()):
        j = old_row.get(int(tid))
        if j is None:
            new_head[i] = torch.randn(old_head.shape[1], generator=g) * (std * 0.5)
        else:
            new_head[i] = old_head[j]
            kept += 1

    out_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out_sd["lm_head.weight"] = new_head
    out_sd["d2t"] = torch.from_numpy(d2t_new)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out_sd, "config": cfg_json,
                "d2t": torch.from_numpy(d2t_new),
                "d2t_sha256": sha256_file(pathlib.Path(args.d2t))}, str(out))

    # round-trip: the new weights must load strictly into the same module
    check = Eagle3Head(cfg)
    check.load_state_dict({k: v for k, v in out_sd.items() if k in model_keys}, strict=True)

    meta = {
        "out": str(out),
        "source": source,
        "draft_vocab_size": int(d2t_new.size),
        "tokens_kept": kept,
        "tokens_newly_initialised": int(d2t_new.size - kept),
        "lm_head_row_std": std,
        "d2t_sha256": sha256_file(pathlib.Path(args.d2t)),
    }
    if args.from_checkpoint:
        meta["source_checkpoint_sha256"] = sha256_file(
            pathlib.Path(args.from_checkpoint))
    else:
        meta["v8_safetensors_sha256"] = sha256_file(
            pathlib.Path(args.v8_dir) / "model.safetensors")
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
