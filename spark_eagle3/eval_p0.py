"""Offline p0: how often would the draft head have predicted the target's choice?

p0 is the probability that the first drafted token matches what the target model
picks. It is the quantity that predicts whether speculative decoding pays for
itself (see AGENTS.md: break-even p0 is about 0.12), and it is hardware
independent, so an offline measurement here can be compared with llama.cpp's
"draft acceptance" line on any machine.

The replay follows the runtime exactly:

    draft input at position p  = (embedding(token[p+1]), fc(features[p]))
    draft prediction           = argmax over the 32k draft vocab, mapped by d2t
    truth                      = target argmax at position p+1

A single causal forward over the sequence is equivalent to the runtime's
incremental pass, because the draft consumes only real tokens and real features
(n_max = 1 never feeds its own output back).

Usage:
  python -m spark_eagle3.eval_p0 --model M.gguf --head datasets/v8 \
      --prompt-file p1.txt [--prompt-file p2.txt ...] --n-gen 128 --out p0.json

`--head` also accepts a training checkpoint (checkpoints/s2a.pt).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from .feats import FeatureStore
from .llamacpp import Shim
from .model import Eagle3Head, config_from_json


def draft_vocab_target_ids(v8_sd) -> np.ndarray:
    """d2t is delta-encoded in the safetensors: target[i] = d2t[i] + i."""
    d2t = v8_sd["d2t"].to(torch.int64)
    return (d2t + torch.arange(d2t.numel(), dtype=torch.int64)).numpy()


def load_head(path: pathlib.Path, dtype=torch.float32):
    """Return (model, cfg, target ids of the draft vocabulary).

    Two sources are understood: the published v8 directory / safetensors, whose
    `d2t` is delta-encoded, and a training checkpoint written by train.py, whose
    `d2t` holds absolute target ids.
    """
    if path.is_dir():
        sd = load_file(str(path / "model.safetensors"))
        cfg = json.loads((path / "config.json").read_text())
        target_ids = draft_vocab_target_ids(sd)
    elif path.suffix == ".pt":
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = blob["state_dict"]
        cfg = blob["config"]
        target_ids = blob["d2t"].to(torch.int64).numpy()
    else:
        sd = load_file(str(path))
        cfg_path = path.parent / "config.json"
        cfg = json.loads(cfg_path.read_text())
        target_ids = draft_vocab_target_ids(sd)
    cfg = config_from_json(cfg)
    model = Eagle3Head(cfg).to(dtype).eval()
    keys = set(model.state_dict().keys())
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items() if k in keys}, strict=True)
    return model, cfg, target_ids


def run_prompt(shim: Shim, store: FeatureStore, head, head_dtype,
               target_ids_of_draft: np.ndarray, text: str, n_gen: int,
               n_layer_feat: int) -> dict:
    ids = shim.tokenize(text, add_special=False)
    if ids.size < 2:
        return {"skip": "too short"}
    n_prompt = int(ids.size)
    n_total = min(n_prompt + n_gen, shim.n_ctx)
    if n_total <= n_prompt:
        return {"skip": "prompt fills the context"}

    # a fresh buffer, not a slice: a view would keep the prompt length
    toks = np.zeros(n_total, dtype=np.int32)
    toks[:n_prompt] = ids[:n_prompt]
    argmax = np.empty(n_total, dtype=np.int32)

    shim.reset()
    store.require(n_total)

    # prefill the prompt in one batch, like the server does
    a = shim.decode(toks[:n_prompt], start_pos=0, feat_ptr=store.row_ptr(0))
    argmax[:n_prompt] = a

    n_done = n_prompt
    hit_eog = False
    while n_done < n_total:
        nxt = int(argmax[n_done - 1])          # greedy
        if nxt < 0:
            break
        toks[n_done] = nxt
        a = shim.decode(toks[n_done:n_done + 1], start_pos=n_done,
                        feat_ptr=store.row_ptr(n_done))
        argmax[n_done] = a[0]
        n_done += 1
        # llama-server stops at EOG, so stop too or the replay scores a tail the
        # runtime never produces
        if shim.is_eog(nxt):
            hit_eog = True
            break

    n_total = n_done
    # draft rows live at positions p where both p+1 and the label are known
    last_p = n_total - 2
    first_p = n_prompt - 1
    if last_p < first_p:
        return {"skip": "nothing generated"}

    rows = np.arange(first_p, last_p + 1)
    # sp_decode() writes row p with the values it computed at p; the draft needs
    # the shift-left pair (token[p+1], argmax[p+1]) instead
    store.write_tails(first_p, toks[first_p + 1:last_p + 2],
                      argmax[first_p + 1:last_p + 2])

    stride = store.row_bytes
    feat_bytes = n_layer_feat * shim.n_embd * 2

    raw = np.frombuffer(store.buf, dtype=np.uint8,
                        count=n_total * stride, offset=0)
    feats = raw.reshape(n_total, stride)[rows, :feat_bytes].copy()
    f32 = (feats.view("<u2").astype(np.uint32) << 16).view(np.float32)
    f32 = f32.reshape(len(rows), n_layer_feat, shim.n_embd)

    tail_tok = raw.reshape(n_total, stride)[rows, feat_bytes:feat_bytes + 4].view("<i4")
    tail_am = raw.reshape(n_total, stride)[rows, feat_bytes + 4:feat_bytes + 8].view("<i4")

    tok_embeds = np.empty((len(rows), shim.n_embd), dtype=np.float32)
    tok_ids = np.ascontiguousarray(tail_tok.ravel(), dtype=np.int32)
    shim.tok_embd_rows(tok_ids, tok_embeds)

    with torch.no_grad():
        feat_t = torch.from_numpy(f32.reshape(len(rows), n_layer_feat * shim.n_embd)) \
                      .to(head_dtype)
        emb_t = torch.from_numpy(tok_embeds).to(head_dtype)
        pos_t = torch.arange(len(rows))
        out = head(feat_t, emb_t, pos_t)
        logits = head.lm_head(out).float()

        # map the 32k draft logits back to target ids: the draft can only ever
        # propose tokens inside its own vocabulary
        order = torch.from_numpy(np.argsort(target_ids_of_draft))
        sorted_ids = torch.from_numpy(np.sort(target_ids_of_draft))
        draft_pick = sorted_ids[logits[:, order].argmax(dim=-1)].numpy()

    truth = tail_am.ravel()
    hit = draft_pick == truth

    # reference points: the target's answer space, and how often a draft pick is
    # even representable by the 32k vocabulary
    in_vocab = np.isin(truth, target_ids_of_draft)

    # agreement between our replayed "target argmax" and the one decoded live
    replay_ok = bool(np.array_equal(argmax[first_p + 1:last_p + 2], truth))

    return {
        "n_prompt": n_prompt,
        "n_total": int(n_total),
        "n_steps": int(len(rows)),
        "p0": float(hit.mean()),
        "truth_in_draft_vocab": float(in_vocab.mean()),
        "replay_argmax_matches": replay_ok,
        "hit_eog": hit_eog,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True, help="v8 checkpoint dir or safetensors")
    ap.add_argument("--prompt-file", action="append", required=True)
    ap.add_argument("--layers", default="2,18,33")
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    head, cfg, target_ids = load_head(pathlib.Path(args.head), dtype)

    results = []
    with Shim(args.model, n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx,
              n_batch=args.n_ctx) as shim:
        shim.set_layers(layers)
        with FeatureStore(args.n_ctx * 20000, len(layers), shim.n_embd) as store:
            for path in args.prompt_file:
                text = pathlib.Path(path).read_text(encoding="utf-8")
                r = run_prompt(shim, store, head, dtype, target_ids, text,
                               args.n_gen, len(layers))
                r["prompt_file"] = path
                results.append(r)
                print("%-42s p0=%.4f steps=%s cover=%.4f%s"
                      % (pathlib.Path(path).name, r.get("p0", -1),
                         r.get("n_steps", "-"),
                         r.get("truth_in_draft_vocab", -1),
                         "" if r.get("replay_argmax_matches", True) else "  [argmax mismatch]"),
                      flush=True)

    scored = [r["p0"] for r in results if "p0" in r]
    summary = {
        "head": str(args.head),
        "dtype": args.dtype,
        "n_gen": args.n_gen,
        "mean_p0": float(np.mean(scored)) if scored else None,
        "mean_truth_in_draft_vocab": float(np.mean(
            [r["truth_in_draft_vocab"] for r in results if "p0" in r])) if scored else None,
        "results": results,
    }
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
