#!/usr/bin/env python3
"""D0: deployment-matched acceptance matrix.

Every gate so far in this repo is greedy (``temperature = 0``), while a real
session samples. This harness measures the same prompts under three server
configurations and writes one JSON line per (mode, temperature, prompt):

    baseline       no draft head; post-sampling top-1 / top-10 mass, i.e. the
                   ceiling a perfect draft could reach at that temperature
    greedy         eagle3 argmax drafts; acceptance from the server log
    probabilistic  eagle3 sampled drafts verified by rejection; acceptance from
                   the server log

The baseline is a separate server because llama.cpp does not report
post-sampling probabilities while speculative decoding is enabled.

The run is resumable: keys already present in ``--out`` are skipped, so a killed
process only loses the request in flight.

Usage:
  python scripts/d0_matrix.py \
      --server "$LLAMACPP_ROOT/build/bin/llama-server" \
      --model "$MODELS_DIR/<target>.gguf" \
      --head "$MODELS_DIR/spark-eagle3-s2a.gguf" \
      --prompt-dir datasets/prompts/daily \
      --out logs/d0_requests.jsonl --log-dir logs/d0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

ACCEPT_RE = re.compile(
    r"draft acceptance = ([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\)")

MODES = ("baseline", "greedy", "probabilistic")


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_health(port: int, proc: subprocess.Popen, timeout: float) -> bool:
    deadline = time.time() + timeout
    url = "http://127.0.0.1:%d/health" % port
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    # VRAM keeps growing for a moment after /health passes
                    time.sleep(3.0)
                    return True
        except (urllib.error.URLError, socket.timeout, ConnectionError):
            pass
        time.sleep(1.0)
    return False


def wait_accept_since(path: pathlib.Path, start_pos: int, timeout: float = 20.0):
    """One acceptance line per request; search only bytes written after start_pos."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(start_pos)
                data = fh.read()
        except FileNotFoundError:
            data = ""
        match = ACCEPT_RE.search(data)
        if match:
            return {
                "p0": float(match.group(1)),
                "accepted": int(match.group(2)),
                "generated": int(match.group(3)),
            }
        time.sleep(0.2)
    return None


def complete(port: int, payload: dict, timeout: float = 1200.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/completion" % port,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def server_cmd(args, mode: str) -> list[str]:
    cmd = [
        args.server,
        "-m", args.model,
        "-c", str(args.ctx),
        "-np", "1",
        "-ngl", str(args.ngl),
        "-fa", "on",
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]
    if mode != "baseline":
        cmd += [
            "-md", args.head,
            "--spec-type", "draft-eagle3",
            "--spec-draft-n-max", "1",
            "--spec-draft-sampling", mode,
        ]
    cmd += args.extra_arg
    return cmd


def load_done(path: pathlib.Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # a half-written last line after a kill; keep the rest
                continue
            done.add("%s|%s|%s" % (row.get("mode"), row.get("temp"), row.get("name")))
    return done


def load_prompts(prompt_dir: pathlib.Path):
    manifest = json.loads((prompt_dir / "manifest.json").read_text(encoding="utf-8"))
    out = []
    for name in sorted(manifest):
        rec = manifest[name]
        out.append((name, rec))
    return out


def run_mode(args, mode: str, prompts, done: set[str], out_fh) -> int:
    log_path = pathlib.Path(args.log_dir) / ("%s.log" % mode)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    cmd = server_cmd(args, mode)
    print("start %-13s %s" % (mode, " ".join(cmd)), flush=True)
    if mode == "probabilistic" and not any(
            "no-spec-draft-backend-sampling" in arg for arg in args.extra_arg):
        print("warning: the default backend sampler is top-k only (greedy); add "
              "--extra-arg=--no-spec-draft-backend-sampling to sample", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    failures = 0
    try:
        if not wait_health(args.port, proc, args.startup_timeout):
            raise SystemExit("server for %s did not become healthy; see %s"
                             % (mode, log_path))
        for temp in args.temps:
            for name, rec in prompts:
                key = "%s|%s|%s" % (mode, temp, name)
                if key in done:
                    continue
                payload = {
                    "prompt": pathlib.Path(rec["path"]).read_text(encoding="utf-8"),
                    "n_predict": args.n_predict,
                    "temperature": temp,
                    "top_p": args.top_p,
                    "min_p": args.min_p,
                    "top_k": 0,
                    "repeat_penalty": 1.0,
                    "seed": args.seed,
                    "cache_prompt": False,
                    "stream": False,
                }
                if mode == "baseline":
                    payload["n_probs"] = args.n_probs
                    payload["post_sampling_probs"] = True

                start_pos = log_path.stat().st_size if log_path.exists() else 0
                t0 = time.time()
                try:
                    resp = complete(args.port, payload)
                except (urllib.error.URLError, socket.timeout) as exc:
                    print("request failed: %s %s" % (key, exc), flush=True)
                    failures += 1
                    continue
                elapsed = time.time() - t0
                timings = resp.get("timings") or {}
                row = {
                    "mode": mode,
                    "temp": temp,
                    "name": name,
                    "register": rec.get("register", name),
                    "seconds": round(elapsed, 3),
                    "n_predicted": timings.get("predicted_n"),
                    "tok_per_s": timings.get("predicted_per_second"),
                }
                if mode == "baseline":
                    rows = resp.get("completion_probabilities") or []
                    top1, top10, in_vocab = [], [], []
                    for token in rows:
                        probs = token.get("top_probs") or []
                        if not probs:
                            continue
                        best = max(probs, key=lambda p: p["prob"])
                        top1.append(best["prob"])
                        top10.append(sum(p["prob"] for p in probs))
                        in_vocab.append(best["id"] in args.d2t_set)
                    row["n_scored"] = len(top1)
                    row["top1"] = sum(top1) / len(top1) if top1 else None
                    row["top10"] = sum(top10) / len(top10) if top10 else None
                    row["n_in_vocab"] = sum(in_vocab)
                    row["coverage"] = (sum(in_vocab) / len(in_vocab)) if in_vocab else None
                else:
                    acc = wait_accept_since(log_path, start_pos)
                    if acc is None:
                        row["p0"] = None
                        row["warning"] = "no draft acceptance line"
                        failures += 1
                    else:
                        row.update(acc)

                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fh.flush()
                print("%-13s T=%.2f %-18s p0=%-6s top1=%-6s %.1fs" % (
                    mode, temp, name,
                    "-" if row.get("p0") is None else "%.3f" % row["p0"],
                    "-" if row.get("top1") is None else "%.3f" % row["top1"],
                    elapsed), flush=True)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--prompt-dir", default="datasets/prompts/daily")
    ap.add_argument("--out", default="logs/d0_requests.jsonl")
    ap.add_argument("--log-dir", default="logs/d0")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--temps", default="0.6,0.7,0.85,1.0")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--n-probs", type=int, default=10)
    ap.add_argument("--d2t", default="datasets/vocab/d2t_v1_32k.i32",
                    help="draft vocabulary; the baseline pass scores coverage against it")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--startup-timeout", type=float, default=300.0)
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="extra llama-server flag (repeatable); use --extra-arg=--flag")
    args = ap.parse_args()

    if port_open(args.port):
        raise SystemExit("port %d is already in use; stop the other server or pass --port"
                         % args.port)

    d2t_path = pathlib.Path(args.d2t)
    if not d2t_path.exists():
        raise SystemExit("d2t not found: %s" % d2t_path)
    args.d2t_set = set(int(x) for x in np.fromfile(d2t_path, dtype=np.int32).tolist())
    print("d2t: %d ids from %s" % (len(args.d2t_set), d2t_path), flush=True)

    prompts = load_prompts(pathlib.Path(args.prompt_dir))
    if not prompts:
        raise SystemExit("no prompts in %s" % args.prompt_dir)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit("unknown modes: %s" % unknown)
    args.temps = [float(t) for t in args.temps.split(",") if t.strip()]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    print("%d prompts, %d temperatures, %d modes, %d keys already done"
          % (len(prompts), len(args.temps), len(modes), len(done)), flush=True)

    total_failures = 0
    with open(out_path, "a", encoding="utf-8") as out_fh:
        for mode in modes:
            total_failures += run_mode(args, mode, prompts, done, out_fh)

    print("done: %d request failures" % total_failures, flush=True)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
