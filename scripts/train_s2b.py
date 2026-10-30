#!/usr/bin/env python3
"""S2b: train the 48K head on the generated NLP mixture.

Differences from `train.py` (the S2a driver):

  * the corpus is the generated `datasets/nlp/*.jsonl` (prompt + target
    response), not the raw public corpora;
  * extraction is contiguous per document: sub-batches do not reset the KV
    cache, so features from a long summary see the whole document prefix;
  * the training-time test mixes the target token and the draft's own argmax
    per position (the runtime's accept/reject), instead of a lagged chain.

Everything else (feature format, memfd store, parking the head during
extraction, checkpoint format) is shared with `train.py`.

Usage:
  python scripts/train_s2b.py --model "$MODELS_DIR/<target>.gguf" \
      --d2t datasets/vocab/d2t_v2_48k.i32 --init checkpoints/head_init_48k.pt \
      --out checkpoints/s2b --steps 2000 --segments 4 --tokens-per-segment 1000000 \
      --seq-len 1024 --batch-seqs 2 --n-ctx 16384 --chunk 2048
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from spark_eagle3.extract import build_sources, extract_contiguous_into  # noqa: E402
from spark_eagle3.feats import FeatureStore, row_bytes  # noqa: E402
from spark_eagle3.llamacpp import Shim  # noqa: E402
from spark_eagle3.train import (  # noqa: E402
    IGNORE, WindowSampler, build_model, load_init, load_vocabulary, lr_at,
    park, save_checkpoint, target_embeddings, to_bf16, unpark,
)

TASKS = ("sum", "trans", "org", "chat", "long")


def parse_weights(text: str) -> dict[str, float]:
    out = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        out[key.strip()] = float(value)
    return out


def plan_segment(args, segment: int):
    weights = parse_weights(args.weights)
    data_dir = pathlib.Path(args.data_dir)
    plan = []
    for task in TASKS:
        path = data_dir / ("%s.jsonl" % task)
        if not path.exists() or task not in weights:
            continue
        quota = int(args.tokens_per_segment * weights[task])
        if quota <= 0:
            continue
        chars = int(quota * args.chars_per_token)
        ns = types.SimpleNamespace(
            text=None, code=None, coig=None, firefly=None, wiki=None,
            nlp=[str(path)], chars_per_source=chars,
            skip_chars=segment * chars,
        )
        plan.append((task, quota, build_sources(ns)))
    return plan


def compute_loss_mix(model, emb, d2t_t, t2d_t, feats, tok, lab, lam, mix_rate):
    """CE(target) + lam * CE(draft-token-mixed), mixing per position.

    At position p the runtime feeds token[p+1]. If the previous draft was
    accepted that token is the draft's own argmax; otherwise it is the target's
    token. The mixed pass samples that choice per position, which is closer to
    the runtime than a pure teacher-forced or pure self-fed pass.
    """
    L = feats.shape[1] - 1
    device = next(model.parameters()).device
    pos = torch.arange(L, device=device)

    f1 = to_bf16(feats[:, :L]).to(device)
    t1 = torch.from_numpy(np.ascontiguousarray(tok[:, :L])).to(device).long()
    labels = t2d_t[torch.from_numpy(np.ascontiguousarray(lab[:, :L])).to(device).long()]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.lm_head(model(f1, F.embedding(t1, emb), pos))
        loss1 = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                labels.reshape(-1), ignore_index=IGNORE)

        with torch.no_grad():
            pick = logits.float().argmax(dim=-1)
        mix = torch.rand(t1.shape, device=device) < mix_rate
        t_mix = torch.where(mix, d2t_t[pick], t1)
        logits2 = model.lm_head(model(f1, F.embedding(t_mix, emb), pos))
        loss2 = F.cross_entropy(logits2.reshape(-1, logits2.shape[-1]),
                                labels.reshape(-1), ignore_index=IGNORE)

    loss = loss1 + lam * loss2
    with torch.no_grad():
        mask = labels != IGNORE
        denom = mask.sum().clamp(min=1)
        top1 = (((pick == labels) & mask).sum().float() / denom).item()
        masked = 1.0 - denom.item() / labels.numel()
    return loss, {
        "loss": float(loss.item()),
        "ce": float(loss1.item()),
        "ce_mix": float(loss2.item()),
        "top1": top1,
        "masked": masked,
    }


def evaluate(model, emb, d2t_t, t2d_t, store, idx, lam, mix_rate):
    model.eval()
    feats = store.feature_view()[idx]
    tok = store.token_view()[idx]
    lab = store.argmax_view()[idx]
    with torch.no_grad():
        _loss, stats = compute_loss_mix(model, emb, d2t_t, t2d_t,
                                        feats, tok, lab, lam, mix_rate)
    model.train()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--d2t", default="datasets/vocab/d2t_v2_48k.i32")
    ap.add_argument("--init", default="checkpoints/head_init_48k.pt")
    ap.add_argument("--out", default="checkpoints/s2b")
    ap.add_argument("--data-dir", default="datasets/nlp")
    ap.add_argument("--weights", default="sum=0.20,trans=0.20,org=0.15,chat=0.15,long=0.30")
    ap.add_argument("--chars-per-token", type=float, default=3.0)

    ap.add_argument("--layers", default="2,18,33")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--n-ctx", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--tokens-per-segment", type=int, default=1_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-seqs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--final-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-ttt", type=float, default=0.5)
    ap.add_argument("--mix-rate", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-chunk-every", type=int, default=25)
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    d2t_path = pathlib.Path(args.d2t)
    init_sd, cfg, cfg_json = load_init(pathlib.Path(args.init), d2t_path)
    model = build_model(cfg, init_sd).to(device).train()
    print("head: %.1fM parameters, draft vocab %d"
          % (sum(p.numel() for p in model.parameters()) / 1e6,
             cfg["draft_vocab_size"]), flush=True)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if len(layers) != len(cfg["eagle_aux_hidden_state_layer_ids"]):
        raise SystemExit("--layers does not match the checkpoint's aux layer list")

    d2t = np.fromfile(d2t_path, dtype=np.int32).astype(np.int64)
    n_vocab = 131072
    if int(d2t.max()) >= n_vocab:
        raise SystemExit("d2t id outside the target vocabulary")
    d2t, t2d = load_vocabulary(d2t_path, n_vocab)
    d2t_t = torch.from_numpy(d2t).to(device)
    t2d_t = torch.from_numpy(t2d).to(device)
    in_set = np.zeros(n_vocab, dtype=bool)
    in_set[d2t] = True

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay, eps=1e-8)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = pathlib.Path(args.log) if args.log else out.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")

    steps_per_segment = max(1, args.steps // args.segments)
    row_bytes_one = row_bytes(len(layers), cfg["hidden_size"])
    global_step = 0
    t_start = time.time()

    for segment in range(args.segments):
        plan = plan_segment(args, segment)
        if not plan:
            raise SystemExit("no NLP data in %s" % args.data_dir)
        seg_tokens = sum(quota for _t, quota, _s in plan)

        park(model, opt)
        emb = None
        torch.cuda.empty_cache()

        t0 = time.time()
        shim = Shim(args.model, n_gpu_layers=args.n_gpu_layers,
                    n_ctx=args.n_ctx + 1, n_batch=args.chunk)
        shim.set_layers(layers)
        emb = target_embeddings(shim)
        store = FeatureStore((seg_tokens + 8) * row_bytes_one, len(layers),
                             cfg["hidden_size"])
        store.require(seg_tokens + 1)

        chunks = []
        cursor = 0
        for name, quota, sources in plan:
            rep = extract_contiguous_into(shim, store, sources, n_ctx=args.n_ctx,
                                          chunk=args.chunk, max_tokens=quota,
                                          start_row=cursor, in_set=in_set)
            chunks.extend(rep["chunks"])
            cursor += rep["n_tokens"]
            print("  segment %d %-6s rows=%7d docs=%4d sub=%5d cov=%s"
                  % (segment, name, rep["n_tokens"], rep["n_docs"],
                     rep["n_sub_batches"],
                     "%.3f" % rep["label_coverage"] if rep["label_coverage"] else "-"),
                  flush=True)
        store.n_tokens = cursor
        store.flush()
        extract_s = time.time() - t0
        print("segment %d: %d rows in %.1f s (%.0f tok/s), %.1f GiB resident"
              % (segment, cursor, extract_s, cursor / max(extract_s, 1e-9),
                 cursor * store.row_bytes / 2**30), flush=True)

        shim.close()
        del shim
        torch.cuda.empty_cache()
        unpark(model, opt, device)

        if args.val_chunk_every > 0:
            val_chunks = [c for i, c in enumerate(chunks) if i % args.val_chunk_every == 0]
            train_chunks = [c for i, c in enumerate(chunks) if i % args.val_chunk_every != 0]
        else:
            val_chunks, train_chunks = [], chunks
        sampler = WindowSampler(train_chunks, args.seq_len)
        if len(sampler) == 0:
            raise SystemExit("no window fits --seq-len %d" % args.seq_len)
        val_sampler = WindowSampler(val_chunks, args.seq_len) if val_chunks else None
        val_starts = (val_sampler.sample(np.random.default_rng(args.seed + 7919),
                                         args.val_batches)
                      if val_sampler is not None and len(val_sampler) else None)

        rng = np.random.default_rng(args.seed + segment)
        model.train()
        running: dict[str, float] = {}
        n_run = 0
        t_log = time.time()
        seg_end = min((segment + 1) * steps_per_segment, args.steps)

        while global_step < seg_end:
            idx = sampler.sample(rng, args.batch_seqs)
            feats = store.feature_view()[idx]
            tok = store.token_view()[idx]
            lab = store.argmax_view()[idx]

            loss, stats = compute_loss_mix(model, emb, d2t_t, t2d_t,
                                           feats, tok, lab, args.lambda_ttt,
                                           args.mix_rate)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = lr_at(global_step, args.steps, args.lr, args.final_lr_frac)
            if args.warmup > 0:
                lr *= min(1.0, (global_step + 1) / args.warmup)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.step()
            global_step += 1

            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + value
            running["grad_norm"] = running.get("grad_norm", 0.0) + float(grad_norm)
            n_run += 1

            if global_step % args.log_every == 0 or global_step == seg_end:
                mean = {k: v / n_run for k, v in running.items()}
                mean.update({"step": global_step, "segment": segment,
                             "lr": lr, "tokens_seen": global_step * args.batch_seqs * args.seq_len})
                if val_starts is not None and (global_step % args.log_every == 0):
                    mean["val"] = evaluate(model, emb, d2t_t, t2d_t, store,
                                           val_starts, args.lambda_ttt, args.mix_rate)
                log_fh.write(json.dumps(mean) + "\n")
                log_fh.flush()
                print("step %5d seg %d ce=%.4f mix=%.4f top1=%.3f lr=%.2e %.1fs"
                      % (global_step, segment, mean["ce"], mean["ce_mix"],
                         mean["top1"], lr, time.time() - t_log), flush=True)
                running.clear()
                n_run = 0
                t_log = time.time()

            if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
                save_checkpoint(out.parent / ("s2b_step%06d.pt" % global_step),
                                model, cfg_json, d2t, step=global_step, segment=segment)

        save_checkpoint(out.parent / "s2b_resume.pt", model, cfg_json, d2t,
                        step=global_step, segment=segment + 1, optimizer=opt)
        save_checkpoint(out.parent / "s2b.pt", model, cfg_json, d2t,
                        step=global_step, segment=segment + 1, extra={
                            "mix_rate": args.mix_rate,
                            "lambda_ttt": args.lambda_ttt,
                            "segments": args.segments,
                            "tokens_per_segment": args.tokens_per_segment,
                        })
        store.close()

    log_fh.close()
    print("done in %.1f min" % ((time.time() - t_start) / 60), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
