#!/usr/bin/env python3
"""P0 diagnostics: the measurements that decide the next training stage.

Nothing here trains anything, and all four modes are hardware independent
because they replay a recorded path through the draft head in PyTorch, exactly
as `spark_eagle3/eval_p0.py` does for protocol P.

Modes
-----
``alpha``
    ``sum_x min(p, q)`` at the deployment sampling law (temp 1.0, top_p 0.95,
    min_p 0.05), swept over a *draft* temperature and both with and without the
    runtime's top-k(10) draft truncation. This is the ceiling for probabilistic
    speculative decoding and the only number that can exceed the argmax ceiling.
    Note `src/models/eagle3.cpp`-side drafting always takes an argmax today, so
    a good alpha is a *reason* to patch the draft loop, not a result.

``p1``
    the runtime's second draft step. The head consumes its own prenorm hidden
    (no `fc`, see `src/models/eagle3.cpp`) and its own argmax, attending over the
    generated prefix. Decomposed into the four token/hidden variants so the cost
    of the current training-time test is measurable.

``channel``
    p0 split by thinking / answer channel. Every S2b training document stops
    inside the thinking channel (§11g), so the answer phase is untrained.

``ctx``
    p0 as a function of session context length. Features are extracted with a
    16k window while the deployment context is 131072.

Usage
-----
    python scripts/p0_diag.py --mode alpha --model "$MODELS_DIR/<target>.gguf" \
        --head checkpoints/s2b.pt --prompt-dir datasets/prompts/nlp \
        --n-gen 128 --out logs/p0_alpha.json

    python scripts/p0_diag.py --mode p1 ... --out logs/p0_p1.json
    python scripts/p0_diag.py --mode channel ... --n-gen 900 --out logs/p0_channel.json
    python scripts/p0_diag.py --mode ctx --doc datasets/prompts/nlp/nlp_zh2en_01.txt \
        --ctx-lens 2048,8192,16384,32768 --out logs/p0_ctx.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

import numpy as np

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from spark_eagle3.diag import (  # noqa: E402
    TargetSampler, alpha_of_row, collect_rows, find_marker, generate, head_pass,
    long_prompt, softmax_probs, thinking_marker, token_embeds,
)
from spark_eagle3.eval_p0 import load_head  # noqa: E402
from spark_eagle3.feats import FeatureStore, row_bytes  # noqa: E402
from spark_eagle3.llamacpp import Shim  # noqa: E402


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def load_prompts(args):
    out = []
    if args.prompt_dir:
        d = pathlib.Path(args.prompt_dir)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for name in sorted(manifest):
            rec = manifest[name]
            out.append({
                "name": name,
                "workload": rec.get("workload") or rec.get("register", name),
                "text": pathlib.Path(rec["path"]).read_text(encoding="utf-8"),
            })
    for p in args.prompt_file or []:
        path = pathlib.Path(p)
        out.append({"name": path.stem, "workload": path.stem,
                    "text": path.read_text(encoding="utf-8")})
    if args.limit:
        out = out[:args.limit]
    if not out:
        raise SystemExit("no prompts: pass --prompt-dir or --prompt-file")
    return out


def draft_vocab_maps(target_ids) -> np.ndarray:
    """target id -> draft index, -1 outside the draft vocabulary."""
    t2d = np.full(int(max(target_ids.max(), 131071)) + 1, -1, dtype=np.int64)
    t2d[np.asarray(target_ids, dtype=np.int64)] = np.arange(len(target_ids),
                                                           dtype=np.int64)
    return t2d


def replay_prompt(shim, store, head, dtype, device, text, n_gen, sampler,
                  collect_support, target_ids, n_layer_feat, max_rows=0,
                  max_prompt_tokens=16384):
    """Generate, extract, and run the head once; returns the scored rows."""
    ids = shim.tokenize(text, add_special=False)
    if ids.size < 2:
        raise ValueError("prompt too short")
    if max_prompt_tokens and ids.size > max_prompt_tokens:
        # keep the tail: the question sits at the end of every rendered prompt
        ids = ids[-int(max_prompt_tokens):]
    rec = generate(shim, store, ids, n_gen, sampler=sampler,
                   collect_support=collect_support)
    first_p = rec["n_prompt"] - 1
    last_p = rec["n_total"] - 2
    if last_p < first_p:
        raise ValueError("nothing generated")
    toks, argmax = rec["tokens"], rec["argmax"]
    store.write_tails(first_p, toks[first_p + 1:last_p + 2],
                      argmax[first_p + 1:last_p + 2])
    rows, feats, tok_ids, labels = collect_rows(
        store, shim, rec["n_total"], first_p, last_p, n_layer_feat)
    if max_rows and max_rows < rows.size:
        rows = rows[:max_rows]
        feats = feats[:max_rows]
        tok_ids = tok_ids[:max_rows]
        labels = labels[:max_rows]
    p = head_pass(head, shim, feats, tok_ids, dtype, device)
    a1 = np.asarray(target_ids)[p["logits"].argmax(-1).cpu().numpy()]
    return {"rec": rec, "rows": rows, "labels": labels, "pass": p, "a1": a1,
            "toks": toks, "argmax": argmax}


# ---------------------------------------------------------------------------
# mode: alpha
# ---------------------------------------------------------------------------
def run_alpha(args, shim, store, head, dtype, device, target_ids, t2d, prompts):
    temps = [float(x) for x in args.draft_temps.split(",")]
    per_prompt = []
    for i, item in enumerate(prompts, 1):
        try:
            r = replay_prompt(shim, store, head, dtype, device, item["text"],
                              args.n_gen, TargetSampler(args.temp, args.top_p,
                                                        args.min_p, args.seed),
                              True, target_ids, args.n_layers,
                              max_prompt_tokens=args.max_prompt_tokens)
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            traceback.print_exc()
            print("  %-24s FAILED: %s" % (item["name"], exc), flush=True)
            per_prompt.append({"name": item["name"], "workload": item["workload"],
                               "error": str(exc)})
            continue

        lg = r["pass"]["logits"].cpu().numpy().astype(np.float64)
        rows = r["rows"]
        labels = r["labels"]
        a1 = r["a1"]
        toks = r["toks"]

        # The draft at row p consumes token[p+1] and predicts *the token after
        # it*, i.e. argmax[p+1] (eval_p0 scores exactly this). The target's
        # sampling law for that prediction is therefore the support recorded at
        # row p+1, not p — using p would compare distributions one position apart.
        keep = [k for k, p in enumerate(rows) if int(p) + 1 in r["rec"]["support"]]
        rows = rows[keep]
        labels = labels[keep]
        a1 = a1[keep]
        lg = lg[keep]
        supports = [r["rec"]["support"][int(p) + 1] for p in rows]

        # the same ordering is valid at every temperature, so sort once
        top10 = np.argsort(-lg, axis=1)[:, :10]

        alpha = {}
        for t in temps:
            z = softmax_probs(lg, t)
            full = np.array([alpha_of_row(z[k], supports[k][0], supports[k][1], t2d)
                             for k in range(rows.size)])
            trunc = np.array([alpha_of_row(z[k], supports[k][0], supports[k][1],
                                           t2d, keep_idx=top10[k])
                              for k in range(rows.size)])
            alpha["%.2f" % t] = {"full": float(full.mean()),
                                 "top10": float(trunc.mean())}

        sampled = np.array([int(toks[int(p) + 2]) for p in rows])
        item_out = {
            "name": item["name"],
            "workload": item["workload"],
            "steps": int(rows.size),
            "alpha": alpha,
            "draft_vs_truth": float((a1 == labels).mean()),
            "draft_vs_sampled": float((a1 == sampled).mean()),
            "support_top_is_label": float(np.mean(
                [supports[k][0][0] == labels[k] for k in range(rows.size)])),
            "support_top_prob_mean": float(np.mean([supports[k][1][0]
                                                    for k in range(rows.size)])),
            "support_size_mean": float(np.mean([s[0].size for s in supports])),
            "support_mass_mean": float(np.mean([s[1].sum() for s in supports])),
        }
        if args.debug_rows:
            for k in range(min(args.debug_rows, rows.size)):
                ids_k, p_k = supports[k]
                print("    row=%d label=%d sampled=%d draft=%d support=%s"
                      % (int(rows[k]), int(labels[k]), int(sampled[k]), int(a1[k]),
                         [("%d:%.3f" % (int(i), float(pv)))
                          for i, pv in zip(ids_k[:6], p_k[:6])]), flush=True)
        per_prompt.append(item_out)
        best = max(alpha.items(), key=lambda kv: kv[1]["full"])
        print("%3d/%d %-24s %-8s steps=%3d alpha(full) best=%s %.3f  top10 %.3f"
              % (i, len(prompts), item["name"], item["workload"], rows.size,
                 best[0], best[1]["full"], best[1]["top10"]), flush=True)
    return {"mode": "alpha", "draft_temps": temps, "results": per_prompt,
            "by_workload": aggregate_alpha(per_prompt, temps)}


def aggregate_alpha(results, temps):
    groups = {}
    for r in results:
        if "error" in r:
            continue
        groups.setdefault(r["workload"], []).append(r)
    out = {}
    for name, rs in groups.items():
        steps = sum(r["steps"] for r in rs)
        out[name] = {
            "prompts": len(rs),
            "steps": steps,
            "alpha": {"%.2f" % t: {
                "full": float(np.average([r["alpha"]["%.2f" % t]["full"] for r in rs],
                                         weights=[r["steps"] for r in rs])),
                "top10": float(np.average([r["alpha"]["%.2f" % t]["top10"] for r in rs],
                                          weights=[r["steps"] for r in rs])),
            } for t in temps},
        }
    if out:
        steps = sum(v["steps"] for v in out.values())
        out["all"] = {
            "prompts": sum(v["prompts"] for v in out.values()),
            "steps": steps,
            "alpha": {"%.2f" % t: {
                "full": float(np.average([v["alpha"]["%.2f" % t]["full"] for v in out.values()],
                                         weights=[v["steps"] for v in out.values()])),
                "top10": float(np.average([v["alpha"]["%.2f" % t]["top10"] for v in out.values()],
                                          weights=[v["steps"] for v in out.values()])),
            } for t in temps},
        }
    return out


# ---------------------------------------------------------------------------
# mode: p1
# ---------------------------------------------------------------------------
def run_p1(args, shim, store, head, dtype, device, target_ids, prompts):
    import torch

    rows_out = []
    for i, item in enumerate(prompts, 1):
        try:
            r = replay_prompt(shim, store, head, dtype, device, item["text"],
                              args.n_gen, None, False, target_ids, args.n_layers,
                              max_rows=args.p1_max_rows,
                              max_prompt_tokens=args.max_prompt_tokens)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print("  %-24s FAILED: %s" % (item["name"], exc), flush=True)
            rows_out.append({"name": item["name"], "workload": item["workload"],
                             "error": str(exc)})
            continue

        p = r["pass"]
        labels = r["labels"]
        a1 = r["a1"]
        R = int(labels.size)
        p0 = float((a1 == labels).mean())

        g_all, e_all, h_all = p["g"], p["e"], p["h"]
        emb_a1 = torch.from_numpy(token_embeds(shim, a1)).to(device=device, dtype=dtype)

        names = ("own_own", "target_own", "target_true")
        hits = {k: 0 for k in names}
        cond = a1[:R - 1] == labels[:R - 1]
        n_cond = int(cond.sum())

        # Conditional on the first draft being accepted, the draft's token *is*
        # the true token, so the (hidden, token) 2x2 collapses onto the hidden
        # axis: each hidden variant is counted twice and the two pairs agree,
        # which the third name below checks rather than assumes.
        for r_i in range(R - 1):
            L = r_i + 2
            pos = torch.arange(L, device=device)
            g = g_all[:, :L, :].clone()
            e = e_all[:, :L, :].clone()
            for name in names:
                if name == "own_own":
                    gv, ev = h_all[0, r_i], emb_a1[r_i]
                elif name == "target_own":
                    gv, ev = g_all[0, r_i + 1], emb_a1[r_i]
                else:
                    gv, ev = g_all[0, r_i + 1], e_all[0, r_i + 1]
                g[0, r_i + 1] = gv
                e[0, r_i + 1] = ev
                with torch.no_grad():
                    hh = head.midlayer(e, g, pos)
                    lg = head.lm_head(head.norm(hh))[0, -1].float()
                a2 = int(np.asarray(target_ids)[int(lg.argmax().item())])
                if cond[r_i] and a2 == int(labels[r_i + 1]):
                    hits[name] += 1

        p1 = {k: (hits[k] / n_cond if n_cond else 0.0) for k in names}
        p0_shifted = float((a1[1:] == labels[1:]).mean()) if R > 1 else 0.0
        out = {
            "name": item["name"], "workload": item["workload"],
            "steps": R, "p0": p0, "p0_shifted": p0_shifted,
            "n_conditional": n_cond,
            "p1": p1,
            "mean_len_nmax2_own_own": 1.0 + p0 + p0 * p1["own_own"],
            "mean_len_nmax2_target_own": 1.0 + p0 + p0 * p1["target_own"],
        }
        rows_out.append(out)
        print("%3d/%d %-24s steps=%3d p0=%.3f p1 own/own=%.3f target/own=%.3f "
              "target/true=%.3f (n=%d)"
              % (i, len(prompts), item["name"], R, p0, p1["own_own"],
                 p1["target_own"], p1["target_true"], n_cond), flush=True)
    return {"mode": "p1", "n_gen": args.n_gen, "results": rows_out,
            "summary": summarise_p1(rows_out)}


def summarise_p1(results):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {}
    w = [r["steps"] for r in ok]
    return {
        "prompts": len(ok),
        "steps": int(sum(w)),
        "p0": float(np.average([r["p0"] for r in ok], weights=w)),
        "p0_shifted": float(np.average([r["p0_shifted"] for r in ok], weights=w)),
        "p1_own_own": float(np.average([r["p1"]["own_own"] for r in ok], weights=w)),
        "p1_target_own": float(np.average([r["p1"]["target_own"] for r in ok],
                                          weights=w)),
        "p1_target_true": float(np.average([r["p1"]["target_true"] for r in ok],
                                           weights=w)),
        "mean_len_nmax2_own_own": float(np.average(
            [r["mean_len_nmax2_own_own"] for r in ok], weights=w)),
    }


# ---------------------------------------------------------------------------
# mode: channel
# ---------------------------------------------------------------------------
def run_channel(args, shim, store, head, dtype, device, target_ids, prompts):
    marker = thinking_marker(args.model)
    marker_b = marker.encode("utf-8")
    print("channel marker: %r (tail %r)" % (marker, args.think_off), flush=True)
    results = []
    for i, item in enumerate(prompts, 1):
        text = item["text"]
        if args.think_off:
            # the template's no-thinking generation prompt is <|Bot|> + marker,
            # so replacing the tail switches channel without re-rendering
            at = text.rfind("<|Bot|>")
            if at < 0:
                print("  %-24s FAILED: no <|Bot|> in the rendered prompt" % item["name"],
                      flush=True)
                continue
            text = text[:at] + "<|Bot|>" + marker
        try:
            r = replay_prompt(shim, store, head, dtype, device, text,
                              args.n_gen, None, False, target_ids, args.n_layers,
                              max_prompt_tokens=args.max_prompt_tokens)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print("  %-24s FAILED: %s" % (item["name"], exc), flush=True)
            results.append({"name": item["name"], "workload": item["workload"],
                            "error": str(exc)})
            continue

        rec, rows, labels, a1 = r["rec"], r["rows"], r["labels"], r["a1"]
        n_prompt = rec["n_prompt"]
        gen = rec["tokens"][n_prompt:rec["n_total"]]
        if args.think_off:
            first, end = -1, -1
        else:
            first, end = find_marker(shim, gen, marker_b)
        if first < 0:
            b0 = b1 = -1
        else:
            b0, b1 = n_prompt + first, n_prompt + end

        tok_of_row = rows + 1
        if args.think_off:
            mask_think = np.zeros(rows.size, dtype=bool)
            mask_ans = np.ones(rows.size, dtype=bool)
        elif b0 < 0:
            mask_think = np.ones(rows.size, dtype=bool)
            mask_ans = np.zeros(rows.size, dtype=bool)
        else:
            mask_think = tok_of_row < b0
            mask_ans = tok_of_row > b1
        hit = a1 == labels

        def _p0(mask):
            return float(hit[mask].mean()) if mask.any() else None

        out = {
            "name": item["name"], "workload": item["workload"],
            "steps": int(rows.size), "n_gen": int(gen.size),
            "marker_found": bool(b0 >= 0),
            "marker_token": int(b0 - n_prompt) if b0 >= 0 else None,
            "think_steps": int(mask_think.sum()),
            "answer_steps": int(mask_ans.sum()),
            "p0_think": _p0(mask_think),
            "p0_answer": _p0(mask_ans),
            "p0_all": float(hit.mean()),
        }
        results.append(out)
        print("%3d/%d %-24s gen=%3d marker=%s think=%3d p0=%.3f | answer=%3d p0=%s"
              % (i, len(prompts), item["name"], gen.size,
                 (out["marker_token"] if out["marker_found"] else "none"),
                 out["think_steps"], out["p0_think"] or -1.0,
                 out["answer_steps"],
                 "-" if out["p0_answer"] is None else "%.3f" % out["p0_answer"]),
              flush=True)
    return {"mode": "channel", "marker": marker, "n_gen": args.n_gen,
            "think_off": bool(args.think_off),
            "results": results, "summary": summarise_channel(results)}


def summarise_channel(results):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {}
    think_steps = sum(r["think_steps"] for r in ok)
    answer_steps = sum(r["answer_steps"] for r in ok)
    think_num = sum((r["p0_think"] or 0.0) * r["think_steps"] for r in ok)
    answer_num = sum((r["p0_answer"] or 0.0) * r["answer_steps"] for r in ok)
    return {
        "prompts": len(ok),
        "prompts_with_marker": sum(1 for r in ok if r["marker_found"]),
        "think_steps": think_steps,
        "answer_steps": answer_steps,
        "p0_think": think_num / think_steps if think_steps else None,
        "p0_answer": answer_num / answer_steps if answer_steps else None,
    }


# ---------------------------------------------------------------------------
# mode: ctx
# ---------------------------------------------------------------------------
def run_ctx(args, shim, store, head, dtype, device, target_ids):
    text = pathlib.Path(args.doc).read_text(encoding="utf-8")
    lens = [int(x) for x in args.ctx_lens.split(",")]
    results = []
    for i, n_ctx in enumerate(lens, 1):
        if n_ctx + args.n_gen + 8 > shim.n_ctx:
            print("skip ctx=%d (shim n_ctx=%d)" % (n_ctx, shim.n_ctx), flush=True)
            continue
        try:
            prompt = long_prompt(shim, text, n_ctx)
            rec = generate(shim, store, prompt, args.n_gen, sampler=None,
                           collect_support=False)
            first_p = rec["n_prompt"] - 1
            last_p = rec["n_total"] - 2
            toks, argmax = rec["tokens"], rec["argmax"]
            store.write_tails(first_p, toks[first_p + 1:last_p + 2],
                              argmax[first_p + 1:last_p + 2])
            rows, feats, tok_ids, labels = collect_rows(
                store, shim, rec["n_total"], first_p, last_p, args.n_layers)
            p = head_pass(head, shim, feats, tok_ids, dtype, device)
            a1 = np.asarray(target_ids)[p["logits"].argmax(-1).cpu().numpy()]
            p0 = float((a1 == labels).mean())
        except Exception as exc:  # noqa: BLE001
            print("  ctx=%d FAILED: %s" % (n_ctx, exc), flush=True)
            results.append({"ctx": n_ctx, "error": str(exc)})
            continue
        out = {"ctx": n_ctx, "steps": int(rows.size), "p0": p0}
        results.append(out)
        print("%d/%d ctx=%6d steps=%3d p0=%.3f" % (i, len(lens), n_ctx,
                                                   rows.size, p0), flush=True)
    return {"mode": "ctx", "doc": args.doc, "n_gen": args.n_gen,
            "results": results}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("alpha", "p1", "channel", "ctx"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True, help="training checkpoint (.pt)")
    ap.add_argument("--prompt-dir", default="")
    ap.add_argument("--prompt-file", action="append")
    ap.add_argument("--doc", default="", help="ctx mode source document")
    ap.add_argument("--ctx-lens", default="2048,8192,16384,32768")
    ap.add_argument("--layers", default="2,18,33")
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--n-ctx", type=int, default=0, help="0 = prompt cap + n_gen + 256")
    ap.add_argument("--max-prompt-tokens", type=int, default=16384,
                    help="truncate longer prompts to their tail")
    ap.add_argument("--n-batch", type=int, default=2048)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts (smoke)")
    ap.add_argument("--max-rows", type=int, default=0, help="cap scored rows")
    ap.add_argument("--p1-max-rows", type=int, default=128)
    ap.add_argument("--debug-rows", type=int, default=0)
    ap.add_argument("--think-off", action="store_true",
                    help="channel mode: render the no-thinking prompt instead")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--draft-temps", default="0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    args.n_layers = len(layers)

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable, falling back to cpu", flush=True)
        args.device = "cpu"
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    head, cfg, target_ids = load_head(pathlib.Path(args.head), dtype)
    head = head.to(device).eval()
    if cfg["eagle_aux_hidden_state_layer_ids"] != layers:
        raise SystemExit("--layers %s does not match the checkpoint's aux layers %s"
                         % (layers, cfg["eagle_aux_hidden_state_layer_ids"]))
    if len(target_ids) != cfg["draft_vocab_size"]:
        raise SystemExit("checkpoint vocabulary and d2t disagree")
    t2d = draft_vocab_maps(target_ids)

    prompts = load_prompts(args) if args.mode != "ctx" else []
    if args.mode == "ctx":
        lens = [int(x) for x in args.ctx_lens.split(",")]
        n_ctx = args.n_ctx or (max(lens) + args.n_gen + 8)
        args.max_prompt_tokens = max(lens)
    elif args.n_ctx:
        n_ctx = args.n_ctx
    else:
        # long-context prompts in the NLP set are ~100k characters; cap the
        # prompt so the KV cache stays affordable and the probe stays inside
        # the extraction distribution (16k, docs/notes.md §11c)
        n_ctx = args.max_prompt_tokens + args.n_gen + 256
    n_ctx = max(n_ctx, 2048)

    n_layer_feat = len(layers)

    with Shim(args.model, n_gpu_layers=args.n_gpu_layers, n_ctx=n_ctx,
              n_batch=min(args.n_batch, n_ctx)) as shim:
        shim.set_layers(layers)
        row_b = row_bytes(n_layer_feat, shim.n_embd)
        with FeatureStore((shim.n_ctx + 8) * row_b, n_layer_feat, shim.n_embd) as store:
            if args.mode == "alpha":
                out = run_alpha(args, shim, store, head, dtype, device,
                                target_ids, t2d, prompts)
            elif args.mode == "p1":
                out = run_p1(args, shim, store, head, dtype, device,
                             target_ids, prompts)
            elif args.mode == "channel":
                out = run_channel(args, shim, store, head, dtype, device,
                                  target_ids, prompts)
            else:
                if not args.doc:
                    raise SystemExit("ctx mode needs --doc")
                out = run_ctx(args, shim, store, head, dtype, device, target_ids)

    out.update({"model": args.model, "head": args.head, "dtype": args.dtype,
                "layers": layers})
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if out.get("summary"):
        print(json.dumps(out["summary"], indent=2))
    if out.get("by_workload"):
        print(json.dumps(out["by_workload"], indent=2))
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001 - a traceback is more useful than a bare exit
        traceback.print_exc()
        raise
