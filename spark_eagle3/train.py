"""Stage 2: train the EAGLE-3 draft head on extracted features.

One process does everything, per segment:

    build shim -> load token_embd -> extract a segment into a memfd -> drop the
    target model -> train N steps on that segment -> checkpoint

Two reasons for the single process. The feature buffer is a memfd (`feats.py`),
so it dies with the process; and constraint 4 of `AGENTS.md` keeps features in
RAM and off NVMe, which rules out writing them to a shard file for a second
process to read. Segmenting (extract/train/checkpoint, repeat) is what makes
this safe: 40 GiB of features never has to exist at once, and a crash costs one
segment of extraction instead of the whole run.

Loss, following the runtime:

    draft input at position p = (embed(token[p+1]), fc(features[p]))
    prediction                = draft argmax, mapped back to target ids by d2t
    target                    = the target model's argmax at p+1

    L = CE(logits, d2t_index(target)) + lambda * CE(training-time test)

The training-time test (second term) re-runs the draft with its *own* argmax fed
back as the input token at the next position, which is exactly what the runtime
does when a draft is accepted: the draft then has to stay correct on its own
output instead of on teacher-forced tokens.

Labels whose target id is outside the draft vocabulary are masked with
`ignore_index` (-100): the head has no output dimension for them, and mapping
them to some neighbouring id would teach it a false target.

Usage:
  python -m spark_eagle3.train --model "$MODELS_DIR/<target>.gguf" \
      --d2t datasets/vocab/d2t_v1_32k.i32 --init checkpoints/head_init.pt \
      --out checkpoints/s2a --steps 8000 --segments 4 --tokens-per-segment 700000 \
      --coig datasets/raw/coig_cqia/COIG-CQIA-full.jsonl \
      --firefly datasets/raw/firefly/firefly-train-1.1M.jsonl \
      --wiki datasets/raw/wikipedia_zh/train-00000-of-00006.parquet \
      --code-dir "$LLAMACPP_ROOT/src"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from .extract import build_sources, extract_into
from .feats import FeatureStore, row_bytes
from .llamacpp import Shim
from .model import Eagle3Head, config_from_json

BASE = pathlib.Path(__file__).resolve().parent.parent
IGNORE = -100


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vocabulary(path: pathlib.Path, n_vocab: int):
    """Return (d2t, t2d): draft index -> target id, and the inverse with -100."""
    d2t = np.fromfile(path, dtype=np.int32).astype(np.int64)
    if d2t.size != len(set(d2t.tolist())):
        raise SystemExit("d2t contains duplicate target ids")
    if d2t.min() < 0 or d2t.max() >= n_vocab:
        raise SystemExit("d2t id outside the target vocabulary")
    t2d = np.full(n_vocab, IGNORE, dtype=np.int64)
    t2d[d2t] = np.arange(d2t.size, dtype=np.int64)
    return d2t, t2d


# ---------------------------------------------------------------------------
# model / checkpoint plumbing
# ---------------------------------------------------------------------------
def load_init(path: pathlib.Path, d2t_path: pathlib.Path):
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg_json = blob["config"]
    cfg = config_from_json(cfg_json)
    sd = blob["state_dict"]
    want_d2t = blob.get("d2t_sha256")
    if want_d2t and want_d2t != sha256_file(d2t_path):
        raise SystemExit(
            "%s was built against a different d2t (%s); refusing to start"
            % (path, want_d2t[:12]))
    return sd, cfg, cfg_json


def build_model(cfg: dict, sd: dict) -> Eagle3Head:
    model = Eagle3Head(cfg)
    keys = set(model.state_dict().keys())
    missing = sorted(keys - set(sd))
    if missing:
        raise SystemExit("checkpoint is missing parameters: %s" % missing)
    model.load_state_dict({k: v for k, v in sd.items() if k in keys}, strict=True)
    return model


def target_embeddings(shim: Shim, dtype=torch.bfloat16) -> torch.Tensor:
    """The target's token_embd as a frozen [n_vocab, n_embd] table on the GPU.

    Copying it once beats a host round-trip per batch: the draft needs an
    embedding for every teacher-forced token *and* for every token it produced
    itself in the training-time test.
    """
    n_vocab, n_embd = shim.n_vocab, shim.n_embd
    host = np.empty((n_vocab, n_embd), dtype=np.float32)
    shim.tok_embd_rows(np.arange(n_vocab, dtype=np.int32), host.reshape(-1))
    out = torch.from_numpy(host).to(dtype).to(torch.device("cuda"))
    del host
    return out


def park(model, opt) -> None:
    """Move the head and its optimiser state off the GPU.

    The target model needs ~6 GiB of VRAM for feature extraction and the head in
    training state holds ~4 GiB (fp32 parameters, gradients, two AdamW moments).
    They cannot be resident together on a 12 GiB card, so extraction and training
    alternate ownership of the GPU.
    """
    model.to("cpu")
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to("cpu")
    torch.cuda.empty_cache()


def unpark(model, opt, device) -> None:
    model.to(device)
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(path: pathlib.Path, model, cfg_json, d2t, *, step, segment,
                    optimizer=None, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": cfg_json,
        "d2t": torch.from_numpy(d2t),
        "step": int(step),
        "next_segment": int(segment),
    }
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    if extra:
        blob["meta"] = extra
    torch.save(blob, str(path))


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------
class WindowSampler:
    """Uniform sampler over (chunk, offset) windows of seq_len + 1 rows.

    Chunks come from independent decodes, so a window may never cross a chunk
    boundary. The extra row is the training-time-test lookahead: forward 2 reads
    features at rows o+1..o+L.
    """

    def __init__(self, chunks, seq_len: int):
        spans = [n - (seq_len + 1) + 1 for _s, n in chunks]
        keep = [(s, span) for (s, n), span in zip(chunks, spans) if span > 0]
        self.seq_len = seq_len
        self.base = np.array([s for s, _ in keep], dtype=np.int64)
        self.span = np.array([span for _, span in keep], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.span.sum())

    def sample(self, rng: np.random.Generator, batch: int) -> np.ndarray:
        """[batch, seq_len + 1] row indices."""
        pick = rng.integers(0, len(self.base), size=batch)
        off = (rng.random(batch) * self.span[pick]).astype(np.int64)
        starts = self.base[pick] + off
        return starts[:, None] + np.arange(self.seq_len + 1, dtype=np.int64)[None, :]


def to_bf16(u8: np.ndarray) -> torch.Tensor:
    """[B, L, feat_bytes] uint8 -> [B, L, n_bf16] bf16, host side, no extra copy."""
    u16 = np.ascontiguousarray(u8).view(np.uint16)
    return torch.from_numpy(u16).view(torch.bfloat16)


# ---------------------------------------------------------------------------
# one optimisation step
# ---------------------------------------------------------------------------
def compute_loss(model, emb, d2t_t, t2d_t, feats, tok, lab, lam: float):
    """feats/tok/lab: [B, L+1, ...] uint8 / [B, L+1] int32 numpy arrays."""
    L = feats.shape[1] - 1
    device = next(model.parameters()).device
    pos = torch.arange(L, device=device)

    f1 = to_bf16(feats[:, :L]).to(device)
    t1 = torch.from_numpy(np.ascontiguousarray(tok[:, :L])).to(device)
    labels = t2d_t[torch.from_numpy(np.ascontiguousarray(lab[:, :L])).to(device).long()]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.lm_head(model(f1, F.embedding(t1.long(), emb), pos))
        loss1 = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                labels.reshape(-1), ignore_index=IGNORE)

        # training-time test: feed the draft's own pick back as the next token.
        # The pick is taken in fp32 because the runtime argmaxes F32 logits;
        # doing it in bf16 would train against a slightly different choice.
        with torch.no_grad():
            pick = logits.float().argmax(dim=-1)
        f2 = to_bf16(feats[:, 1:]).to(device)
        labels2 = t2d_t[torch.from_numpy(np.ascontiguousarray(lab[:, 1:])).to(device).long()]
        logits2 = model.lm_head(model(f2, F.embedding(d2t_t[pick], emb), pos))
        loss2 = F.cross_entropy(logits2.reshape(-1, logits2.shape[-1]),
                                labels2.reshape(-1), ignore_index=IGNORE)

    with torch.no_grad():
        mask = labels != IGNORE
        denom = mask.sum().clamp(min=1)
        top1 = (((pick == labels) & mask).sum().float() / denom).item()
        masked = (1.0 - denom.item() / labels.numel())
    return loss1 + lam * loss2, {
        "loss": float((loss1 + lam * loss2).item()),
        "ce": float(loss1.item()),
        "ce_ttt": float(loss2.item()),
        "top1": top1,
        "masked": masked,
    }


def evaluate(model, emb, d2t_t, t2d_t, store: FeatureStore, idx: np.ndarray,
             lam: float) -> dict:
    """Validation loss on a fixed set of [B, L+1] row indices."""
    model.eval()
    feats = store.feature_view()[idx]
    tok = store.token_view()[idx]
    lab = store.argmax_view()[idx]
    with torch.no_grad():
        _loss, stats = compute_loss(model, emb, d2t_t, t2d_t, feats, tok, lab, lam)
    model.train()
    return stats


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
DOMAINS = ("coig", "firefly", "wiki", "code")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--d2t", required=True)
    ap.add_argument("--init", default="checkpoints/head_init.pt")
    ap.add_argument("--out", default="checkpoints/s2a")
    ap.add_argument("--layers", default="2,18,33")
    ap.add_argument("--chunk", type=int, default=2048)

    ap.add_argument("--steps", type=int, default=8000, help="total optimiser steps")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--tokens-per-segment", type=int, default=700_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-seqs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--final-lr-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-ttt", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-chunk-every", type=int, default=25,
                    help="every Nth chunk is held out (0 disables validation)")
    ap.add_argument("--val-batches", type=int, default=4)
    ap.add_argument("--resume", default="", help="full-state checkpoint to continue from")

    ap.add_argument("--raw-dir", default=str(BASE / "datasets" / "raw"))
    ap.add_argument("--coig", action="append")
    ap.add_argument("--firefly", action="append")
    ap.add_argument("--wiki", action="append")
    ap.add_argument("--code-dir", action="append")
    ap.add_argument("--weights", default="coig=0.35,firefly=0.35,wiki=0.20,code=0.10")
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--log", default="")
    return ap.parse_args()


def resolve_sources(args) -> dict[str, list[pathlib.Path]]:
    raw = pathlib.Path(args.raw_dir)
    found = {
        "coig": [pathlib.Path(p) for p in (args.coig or [])],
        "firefly": [pathlib.Path(p) for p in (args.firefly or [])],
        "wiki": [pathlib.Path(p) for p in (args.wiki or [])],
        "code": [pathlib.Path(p) for p in (args.code_dir or [])],
    }
    defaults = {
        "coig": raw / "coig_cqia" / "COIG-CQIA-full.jsonl",
        "firefly": raw / "firefly" / "firefly-train-1.1M.jsonl",
        "wiki": raw / "wikipedia_zh" / "train-00000-of-00006.parquet",
    }
    for name, path in defaults.items():
        if not found[name] and path.exists():
            found[name] = [path]
    return found


def segment_sources(args, found, segment: int):
    """One domain at a time, each starting where the previous segment stopped."""
    weights = {}
    for part in args.weights.split(","):
        key, _, value = part.partition("=")
        weights[key.strip()] = float(value)

    plan = []
    for name in DOMAINS:
        if name not in weights or not found[name]:
            continue
        quota = int(args.tokens_per_segment * weights[name])
        if quota <= 0:
            continue
        # 4 characters per token is above every domain's real ratio (~1.4-2.8),
        # so the token quota is what binds and the char budget just bounds the
        # read. Tiling by the char budget keeps segments disjoint.
        chars = quota * 4
        ns = types.SimpleNamespace(
            text=None, code=None, coig=None, firefly=None, wiki=None,
            chars_per_source=chars, skip_chars=segment * chars,
        )
        if name == "coig":
            ns.coig = [str(p) for p in found["coig"]]
        elif name == "firefly":
            ns.firefly = [str(p) for p in found["firefly"]]
        elif name == "wiki":
            ns.wiki = [str(p) for p in found["wiki"]]
        else:
            ns.code = [str(p) for p in found["code"]]
        plan.append((name, quota, build_sources(ns)))
    return plan


def lr_at(step: int, total: int, base: float, frac: float) -> float:
    if total <= 0:
        return base
    t = min(step / total, 1.0)
    final = base * frac
    return final + (base - final) * 0.5 * (1.0 + math.cos(math.pi * t))


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    d2t_path = pathlib.Path(args.d2t)
    init_sd, cfg, cfg_json = load_init(pathlib.Path(args.init), d2t_path)
    model = build_model(cfg, init_sd).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    print("head: %.1fM parameters, draft vocab %d"
          % (n_params / 1e6, cfg["draft_vocab_size"]), flush=True)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if len(layers) != len(cfg["eagle_aux_hidden_state_layer_ids"]):
        raise SystemExit("--layers does not match the checkpoint's aux layer list")

    steps_per_segment = args.steps // args.segments
    if steps_per_segment <= 0:
        raise SystemExit("--steps must be at least --segments")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay, eps=1e-8)

    start_step = 0
    first_segment = 0
    if args.resume:
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v for k, v in blob["state_dict"].items()
                               if k in set(model.state_dict().keys())}, strict=True)
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
        start_step = int(blob.get("step", 0))
        first_segment = int(blob.get("next_segment", 0))
        print("resumed at step %d, next segment %d" % (start_step, first_segment),
              flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = pathlib.Path(args.log) if args.log else out.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")

    found = resolve_sources(args)
    print("sources: " + ", ".join("%s=%d" % (k, len(v)) for k, v in found.items()),
          flush=True)
    if sum(len(v) for v in found.values()) < 2:
        raise SystemExit("need at least two corpora; run scripts/fetch_corpora.sh first")

    emb = None
    global_step = start_step
    t_start = time.time()

    for segment in range(first_segment, args.segments):
        plan = segment_sources(args, found, segment)
        seg_tokens = 0
        for _name, quota, _src in plan:
            seg_tokens += quota
        # one scratch row per chunk is written past its valid rows; the last
        # chunk of each domain therefore reaches seg_tokens + 1 at worst
        capacity = (seg_tokens + 8) * row_bytes(len(layers), cfg["hidden_size"])

        # the target model cannot share the card with the head's optimiser state
        park(model, opt)
        if emb is not None:
            del emb
            emb = None
            torch.cuda.empty_cache()

        t0 = time.time()
        shim = Shim(args.model, n_gpu_layers=args.n_gpu_layers,
                    n_ctx=args.chunk + 1, n_batch=args.chunk + 1)
        shim.set_layers(layers)
        emb = target_embeddings(shim)
        print("token_embd: %.0f MB on GPU" % (emb.numel() * 2 / 2**20), flush=True)
        store = FeatureStore(capacity, len(layers), cfg["hidden_size"])
        store.require(seg_tokens + 1)

        d2t, t2d = load_vocabulary(d2t_path, shim.n_vocab)
        d2t_t = torch.from_numpy(d2t).to(device)
        t2d_t = torch.from_numpy(t2d).to(device)
        in_set = np.zeros(shim.n_vocab, dtype=bool)
        in_set[d2t] = True

        chunks = []
        cursor = 0
        for name, quota, sources in plan:
            rep = extract_into(shim, store, sources, chunk=args.chunk,
                               max_tokens=quota, start_row=cursor, in_set=in_set)
            chunks.extend(rep["chunks"])
            cursor += rep["n_tokens"]
            print("  segment %d %-8s rows=%6d docs=%4d cov=%s"
                  % (segment, name, rep["n_tokens"], rep["n_docs"],
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

        # hold out every Nth chunk so the validation set covers all domains
        if args.val_chunk_every > 0:
            val_chunks = [c for i, c in enumerate(chunks) if i % args.val_chunk_every == 0]
            train_chunks = [c for i, c in enumerate(chunks) if i % args.val_chunk_every != 0]
        else:
            val_chunks, train_chunks = [], chunks
        sampler = WindowSampler(train_chunks, args.seq_len)
        if len(sampler) == 0:
            raise SystemExit("no window fits --seq-len %d in a %d-token chunk"
                             % (args.seq_len, args.chunk))
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

            loss, stats = compute_loss(model, emb, d2t_t, t2d_t,
                                       feats, tok, lab, args.lambda_ttt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = lr_at(global_step, args.steps, args.lr, args.final_lr_frac)
            for group in opt.param_groups:
                group["lr"] = lr
            opt.step()
            global_step += 1

            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v
            running["grad_norm"] = running.get("grad_norm", 0.0) + float(grad_norm)
            n_run += 1

            if global_step % args.log_every == 0 or global_step == seg_end:
                mean = {k: v / n_run for k, v in running.items()}
                now = time.time()
                tps = args.batch_seqs * args.seq_len * n_run / max(now - t_log, 1e-9)
                t_log = now
                row = {"step": global_step, "segment": segment, "lr": lr,
                       "tok_s": round(tps, 1), **mean}
                print("step %5d  loss %.4f  ce %.4f  ce_ttt %.4f  top1 %.4f  "
                      "masked %.4f  gnorm %.2f  lr %.2e  %.0f tok/s"
                      % (global_step, mean["loss"], mean["ce"], mean["ce_ttt"],
                         mean["top1"], mean["masked"], mean["grad_norm"], lr, tps),
                      flush=True)
                log_fh.write(json.dumps(row) + "\n")
                log_fh.flush()
                running, n_run = {}, 0

            if global_step % args.ckpt_every == 0 and global_step < args.steps:
                p = out.with_name("%s_step%06d.pt" % (out.name, global_step))
                save_checkpoint(p, model, cfg_json, d2t, step=global_step,
                                segment=segment)
                if val_starts is not None:
                    v = evaluate(model, emb, d2t_t, t2d_t, store, val_starts,
                                 args.lambda_ttt)
                    print("  val ce %.4f top1 %.4f" % (v["ce"], v["top1"]), flush=True)
                    log_fh.write(json.dumps({"step": global_step, "segment": segment,
                                             "val_ce": v["ce"], "val_top1": v["top1"]})
                                 + "\n")
                    log_fh.flush()

        save_checkpoint(out.with_name(out.name + "_resume.pt"), model, cfg_json, d2t,
                        step=global_step, segment=segment + 1, optimizer=opt,
                        extra={"lambda_ttt": args.lambda_ttt, "lr": args.lr,
                               "seq_len": args.seq_len, "batch_seqs": args.batch_seqs,
                               "tokens_per_segment": args.tokens_per_segment,
                               "weights": args.weights})
        print("segment %d done at step %d (%.1f min elapsed)"
              % (segment, global_step, (time.time() - t_start) / 60), flush=True)
        store.close()

    save_checkpoint(out.with_suffix(".pt"), model, cfg_json, d2t,
                    step=global_step, segment=args.segments,
                    extra={"lambda_ttt": args.lambda_ttt, "lr": args.lr})
    log_fh.close()
    print("final checkpoint %s at step %d" % (out.with_suffix(".pt"), global_step),
          flush=True)
    if emb is not None:
        del emb
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
