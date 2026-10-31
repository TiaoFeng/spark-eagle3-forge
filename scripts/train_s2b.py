#!/usr/bin/env python3
"""Train the 48K draft head on the generated NLP mixture (S2b, and S3 with flags).

Differences from `train.py` (the S2a driver):

  * the corpus is the generated `datasets/nlp/*.jsonl` (prompt + target
    response), not the raw public corpora;
  * extraction is contiguous per document: sub-batches do not reset the KV
    cache, so features from a long summary see the whole document prefix;
  * the training-time test mixes the target token and the draft's own argmax
    per position (the runtime's accept/reject), instead of a lagged chain.

S3 adds two options rather than a second driver:

  * `--answer-dir` / `--answer-weights` mix in a thinking-off corpus from
    `gen_nlp_data.py --think-off`. S2b only ever saw the thinking channel
    (`docs/notes.md` §11g) and P0 measured the answer phase 0.11 lower (§13e).
  * `--short-seq-len` / `--short-mix` sample from a second window pool built
    from the documents the long pool drops, because a chunk shorter than
    `seq_len + 1` rows used to contribute nothing at all — which is exactly the
    shape of an answer document.

Everything else (feature format, memfd store, parking the head during
extraction, checkpoint format) is shared with `train.py`.

Usage (S2b):
  python scripts/train_s2b.py --model "$MODELS_DIR/<target>.gguf" \
      --d2t datasets/vocab/d2t_v2_48k.i32 --init checkpoints/head_init_48k.pt \
      --out checkpoints/s2b --steps 2000 --segments 4 --tokens-per-segment 1000000 \
      --seq-len 1024 --batch-seqs 2 --n-ctx 16384 --chunk 2048

Usage (S3, warm start from S2b):
  python scripts/train_s2b.py --model "$MODELS_DIR/<target>.gguf" \
      --d2t datasets/vocab/d2t_v2_48k.i32 --init checkpoints/s2b.pt \
      --out checkpoints/s3 --steps 6000 --segments 6 --tokens-per-segment 2500000 \
      --weights sum=0.17,trans=0.17,org=0.13,chat=0.13,long=0.25 \
      --answer-dir datasets/nlp_answer \
      --answer-weights sum=0.05,trans=0.05,org=0.03,chat=0.02 \
      --short-seq-len 256 --short-mix 0.25 --n-ctx 32768
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
    park, save_checkpoint, split_window_pools, target_embeddings, to_bf16, unpark,
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

    def add(name: str, path: pathlib.Path, weight: float) -> None:
        if weight <= 0:
            return
        quota = int(args.tokens_per_segment * weight)
        if quota <= 0:
            return
        if not path.exists():
            raise SystemExit("missing %s (task %s)" % (path, name))
        chars = int(quota * args.chars_per_token)
        ns = types.SimpleNamespace(
            text=None, code=None, coig=None, firefly=None, wiki=None,
            nlp=[str(path)], chars_per_source=chars,
            skip_chars=segment * chars,
        )
        plan.append((name, quota, build_sources(ns)))

    for task in TASKS:
        if task in weights:
            add(task, data_dir / ("%s.jsonl" % task), weights[task])

    # Answer-channel slice (S3): the same tasks rendered with thinking disabled,
    # so the head finally sees the phase it has never been trained on (P0 §13e
    # measured 0.385 there against 0.493 in thinking). Logged with an `a_` prefix.
    if args.answer_dir:
        ans_weights = parse_weights(args.answer_weights) if args.answer_weights else {}
        ans_dir = pathlib.Path(args.answer_dir)
        for task in TASKS:
            if task in ans_weights:
                add("a_" + task, ans_dir / ("%s.jsonl" % task), ans_weights[task])
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
    ap.add_argument("--answer-dir", default="",
                    help="thinking-off corpus from gen_nlp_data.py --think-off")
    ap.add_argument("--answer-weights", default="",
                    help="e.g. sum=0.12,trans=0.10,org=0.08,chat=0.08")
    ap.add_argument("--short-seq-len", type=int, default=0,
                    help="second window length for documents the long pool drops")
    ap.add_argument("--short-mix", type=float, default=0.0,
                    help="fraction of steps drawn from the short pool")
    ap.add_argument("--short-batch-seqs", type=int, default=8,
                    help="sequences per short step, so tokens/step stay comparable")
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
        sampler, short_sampler = split_window_pools(train_chunks, args.seq_len,
                                                    args.short_seq_len)
        if len(sampler) == 0:
            if short_sampler is None or len(short_sampler) == 0:
                raise SystemExit("no window fits --seq-len %d" % args.seq_len)
            print("segment %d: no long windows; promoting the short pool" % segment,
                  flush=True)
            sampler, short_sampler = short_sampler, None
        elif short_sampler is not None and len(short_sampler) == 0:
            short_sampler = None
        print("segment %d: %d long windows, %d short windows"
              % (segment, len(sampler),
                 len(short_sampler) if short_sampler is not None else 0), flush=True)
        val_sampler = WindowSampler(val_chunks, args.seq_len) if val_chunks else None
        val_starts = (val_sampler.sample(np.random.default_rng(args.seed + 7919),
                                         args.val_batches)
                      if val_sampler is not None and len(val_sampler) else None)

        rng = np.random.default_rng(args.seed + segment)
        model.train()
        running: dict[str, float] = {}
        n_run = 0
        tokens_seen = 0
        t_log = time.time()
        seg_end = min((segment + 1) * steps_per_segment, args.steps)

        while global_step < seg_end:
            use_short = (short_sampler is not None and args.short_mix > 0.0
                         and float(rng.random()) < args.short_mix)
            pool = short_sampler if use_short else sampler
            step_len = args.short_seq_len if use_short else args.seq_len
            step_batch = args.short_batch_seqs if use_short else args.batch_seqs
            idx = pool.sample(rng, step_batch)
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
            running["short"] = running.get("short", 0.0) + (1.0 if use_short else 0.0)
            running["grad_norm"] = running.get("grad_norm", 0.0) + float(grad_norm)
            n_run += 1
            tokens_seen += step_batch * step_len

            if global_step % args.log_every == 0 or global_step == seg_end:
                mean = {k: v / n_run for k, v in running.items()}
                mean.update({"step": global_step, "segment": segment,
                             "lr": lr, "tokens_seen": tokens_seen})
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
                save_checkpoint(out.parent / ("%s_step%06d.pt" % (out.stem, global_step)),
                                model, cfg_json, d2t, step=global_step, segment=segment)

        # checkpoint names follow --out, so `--out checkpoints/s3` never touches
        # the S2b head: s2b.pt / s2b_resume.pt / s2b_step*.pt keep their old
        # names because the documented S2b command still uses `--out checkpoints/s2b`
        save_checkpoint(out.parent / (out.stem + "_resume.pt"), model, cfg_json, d2t,
                        step=global_step, segment=segment + 1, optimizer=opt)
        save_checkpoint(out.with_suffix(".pt"), model, cfg_json, d2t,
                        step=global_step, segment=segment + 1, extra={
                            "mix_rate": args.mix_rate,
                            "lambda_ttt": args.lambda_ttt,
                            "segments": args.segments,
                            "tokens_per_segment": args.tokens_per_segment,
                            "answer_weights": args.answer_weights,
                            "short_seq_len": args.short_seq_len,
                            "short_mix": args.short_mix,
                        })
        store.close()

    log_fh.close()
    print("done in %.1f min" % ((time.time() - t_start) / 60), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
