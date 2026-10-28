#!/usr/bin/env python3
"""G1: measure llama.cpp's own draft acceptance for the same prompts.

The server prints, per request,

    draft acceptance = <ratio> (<accepted> accepted / <generated> generated)

With n-max = 1, generated == number of verification steps, so the ratio is
exactly p0 -- the same quantity eval_p0.py computes offline. Both sides run
greedy so the comparison is free of sampling noise.

Usage:
  python scripts/g1_online.py --model M.gguf --head v8.gguf \
      --prompt-file p1.txt [...] --n-predict 128 --out logs/g1_online.json
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

ACCEPT_RE = re.compile(r"draft acceptance = ([0-9.]+)\s*\(\s*(\d+) accepted / (\d+) generated\)")


def wait_health(port: int, proc: subprocess.Popen, timeout: float = 240.0) -> bool:
    deadline = time.time() + timeout
    url = "http://127.0.0.1:%d/health" % port
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    # VRAM keeps growing for a moment after /health passes
                    time.sleep(3.0)
                    return True
        except (urllib.error.URLError, socket.timeout, ConnectionError):
            pass
        time.sleep(1.0)
    return False


def complete(port: int, prompt: str, n_predict: int) -> dict:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "stream": False,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="llama-server binary")
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True, help="v8 eagle3 GGUF")
    ap.add_argument("--spec-type", default="draft-eagle3")
    ap.add_argument("--prompt-file", action="append", required=True)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--extra-arg", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-log", required=True)
    args = ap.parse_args()

    cmd = [
        args.server,
        "-m", args.model,
        "-md", args.head,
        "--spec-type", args.spec_type,
        "--spec-draft-n-max", "1",
        "-c", str(args.ctx),
        "-np", "1",
        "-ngl", "99",
        "-fa", "on",
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ] + args.extra_arg

    log_path = pathlib.Path(args.server_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    results = []
    try:
        if not wait_health(args.port, proc):
            print("server did not become healthy; see %s" % log_path, file=sys.stderr)
            return 1

        for path in args.prompt_file:
            text = pathlib.Path(path).read_text(encoding="utf-8")
            t0 = time.time()
            resp = complete(args.port, text, args.n_predict)
            elapsed = time.time() - t0
            results.append({
                "prompt_file": path,
                "seconds": round(elapsed, 2),
                "n_predicted": resp.get("timings", {}).get("predicted_n"),
            })
            print("%-28s %.1fs" % (pathlib.Path(path).name, elapsed), flush=True)
            time.sleep(0.5)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    # llama-server prints its timing block after the response is sent, so the
    # acceptance lines are collected once everything has finished. One line per
    # request, in request order.
    lines = []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ACCEPT_RE.search(line)
            if m:
                lines.append((float(m.group(1)), int(m.group(2)), int(m.group(3))))
    if len(lines) != len(results):
        print("warning: %d acceptance lines for %d requests"
              % (len(lines), len(results)), file=sys.stderr)
    for entry, (p0, acc, gen) in zip(results, lines):
        entry["p0"], entry["accepted"], entry["generated"] = p0, acc, gen

    scored = [r["p0"] for r in results if r.get("p0") is not None]
    summary = {
        "spec_type": args.spec_type,
        "n_predict": args.n_predict,
        "mean_p0": sum(scored) / len(scored) if scored else None,
        "results": results,
    }
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    print("mean p0 = %s" % summary["mean_p0"])
    return 0 if scored else 1


if __name__ == "__main__":
    raise SystemExit(main())
