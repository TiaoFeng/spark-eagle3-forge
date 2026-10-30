#!/usr/bin/env python3
"""Generate the S2b NLP corpus with the target model.

The draft is trained on the target's own output, so every document is a rendered
chat prompt plus the continuation the target produced for it. Tasks follow the
daily workload: summarisation, translation (both directions), document
organisation, chat, and a long-context variant of the first two.

The generator starts its own llama-server (no draft head), renders prompts with
the model's chat template, and writes one JSONL record per request:

    {"id", "task", "source", "prompt", "response", "n_predict", "timings"}

Runs are resumable: records already present in the output are skipped.

Usage:
  python scripts/gen_nlp_data.py --server "$LLAMACPP_ROOT/build/bin/llama-server" \
      --model "$MODELS_DIR/<target>.gguf" --out-dir datasets/nlp \
      --scale 1.0 --parallel 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import jinja2

BASE = pathlib.Path(__file__).resolve().parent.parent
TASKS = ("sum", "trans", "org", "chat", "long")


# ---------------------------------------------------------------------------
# public source pool
# ---------------------------------------------------------------------------
def section(rel: str, index: int, chars: int) -> str:
    parts = (BASE / rel).read_text(encoding="utf-8").split("\n## ")
    if index >= len(parts):
        return ""
    return parts[index][:chars].strip()


def iter_parquet_texts(path: pathlib.Path, count: int, lo: int, hi: int):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    seen = 0
    for batch in pf.iter_batches(batch_size=256, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if not text:
                continue
            text = text.strip()
            if not (lo <= len(text) <= hi):
                continue
            yield text
            seen += 1
            if seen >= count:
                return


def iter_coig_instructions(path: pathlib.Path, count: int):
    seen = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (rec.get("instruction") or "").strip()
            if len(text) < 20:
                continue
            yield text
            seen += 1
            if seen >= count:
                return


def build_jobs(scale: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    en_wiki = BASE / "datasets/raw/wikipedia_en/train-00000-of-00041.parquet"
    zh_wiki = BASE / "datasets/raw/wikipedia_zh/train-00000-of-00006.parquet"
    coig = BASE / "datasets/raw/coig_cqia/COIG-CQIA-full.jsonl"

    n_sum = int(600 * scale)
    n_trans = int(1200 * scale)
    n_org = int(400 * scale)
    n_chat = int(1200 * scale)
    n_long = int(120 * scale)

    en_articles = list(iter_parquet_texts(en_wiki, max(n_sum, n_trans) + 40, 900, 4200))
    zh_articles = list(iter_parquet_texts(zh_wiki, max(n_sum, n_trans) + 40, 500, 2200))
    instructions = list(iter_coig_instructions(coig, n_chat + 20))
    rng.shuffle(en_articles)
    rng.shuffle(zh_articles)
    rng.shuffle(instructions)

    jobs = []

    def add(task, source, text, effort="high"):
        jobs.append({"id": "%s_%04d" % (task, sum(1 for j in jobs if j["task"] == task)),
                     "task": task, "source": source, "text": text, "effort": effort})

    for i in range(n_sum):
        # mostly the deployment `high` effort; a third `low`, intended to let
        # some responses reach the answer channel — in practice all of them
        # stopped inside thinking (docs/notes.md §11g)
        effort = "high" if i % 3 else "low"
        if i % 2 == 0 and zh_articles:
            add("sum", "zh_wiki", "请用 200 字左右总结下面的中文材料，保留关键数字与结论。\n\n"
                + zh_articles[i % len(zh_articles)], effort)
        elif en_articles:
            add("sum", "en_wiki", "请用中文总结下面的英文材料，约 200 字，保留关键数字与结论。\n\n"
                + en_articles[i % len(en_articles)], effort)

    for i in range(n_trans):
        effort = "high" if i % 3 else "low"
        if i % 2 == 0 and zh_articles:
            add("trans", "zh2en", "把下面的中文翻译成英文，只输出译文，不要解释。\n\n"
                + zh_articles[(i + 7) % len(zh_articles)], effort)
        elif en_articles:
            add("trans", "en2zh", "把下面的英文翻译成简体中文，只输出译文，不要解释。\n\n"
                + en_articles[(i + 7) % len(en_articles)], effort)

    org_prompts = (
        "把下面的材料整理成三级大纲，每节写一句摘要。\n\n",
        "把下面的材料整理成表格，列为：主题、要点、结论。\n\n",
        "把下面的材料整理成 JSON，字段包括主题、要点、结论。\n\n",
    )
    for i in range(n_org):
        pool = zh_articles if i % 2 == 0 else en_articles
        if not pool:
            continue
        effort = "high" if i % 3 else "low"
        add("org", "wiki", org_prompts[i % len(org_prompts)] + pool[i % len(pool)], effort)

    for i in range(n_chat):
        if instructions:
            add("chat", "coig", instructions[i % len(instructions)], "high")

    # Long-context documents: the whole design notes in English, and a
    # concatenation of Chinese wiki articles.
    notes = (BASE / "docs/notes.md").read_text(encoding="utf-8")
    long_en = [notes[i:i + 24000] for i in range(0, min(len(notes), 24000 * max(1, n_long // 2)),
                                                   24000)] or [notes]
    long_zh = []
    if zh_articles:
        for start in range(0, min(len(zh_articles), n_long), 8):
            long_zh.append("\n\n".join(zh_articles[start:start + 8]))
    long_prompts = (
        "请综合下面几份材料，写一份 500 字左右的整合摘要，并在最后列出三条结论。\n\n",
        "请把下面的长文档整理成三级大纲，并在每节后写一句摘要。\n\n",
    )
    for i in range(n_long):
        effort = "high" if i % 3 else "low"
        if i % 2 == 0 and long_zh:
            add("long", "zh_long", long_prompts[i % 2] + long_zh[i % len(long_zh)], effort)
        elif long_en:
            add("long", "en_long", long_prompts[i % 2] + long_en[i % len(long_en)], effort)

    return jobs


# ---------------------------------------------------------------------------
# server / HTTP
# ---------------------------------------------------------------------------
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
                    time.sleep(3.0)
                    return True
        except (urllib.error.URLError, socket.timeout, ConnectionError):
            pass
        time.sleep(1.0)
    return False


def complete(port: int, prompt: str, n_predict: int, seed: int) -> dict:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.7,
        "top_p": 0.95,
        "min_p": 0.05,
        "top_k": 0,
        "repeat_penalty": 1.0,
        "seed": seed,
        "cache_prompt": True,
        "stream": False,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % port, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="datasets/nlp")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--n-predict", type=int, default=640)
    ap.add_argument("--long-n-predict", type=int, default=896)
    ap.add_argument("--port", type=int, default=8933)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--startup-timeout", type=float, default=300.0)
    args = ap.parse_args()

    if port_open(args.port):
        sys.exit("port %d is already in use" % args.port)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from gguf import GGUFReader
    reader = GGUFReader(args.model)
    field = reader.fields.get("tokenizer.chat_template")
    if field is None:
        sys.exit("model has no tokenizer.chat_template")
    tmpl = jinja2.Environment(trim_blocks=False, lstrip_blocks=False).from_string(
        str(field.contents()))
    tmpl.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))

    jobs = build_jobs(args.scale, args.seed)
    print("%d jobs" % len(jobs), flush=True)

    # resume: one JSONL per task group
    done = set()
    handles = {}
    for task in TASKS:
        path = out_dir / ("%s.jsonl" % task)
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:  # noqa: BLE001
                        continue
        handles[task] = open(path, "a", encoding="utf-8")
    print("%d jobs already done" % len(done), flush=True)

    log = open(out_dir / "server.log", "w")
    cmd = [args.server, "-m", args.model, "-c", str(args.ctx), "-np", str(args.parallel),
           "-ngl", "99", "-fa", "on", "--host", "127.0.0.1", "--port", str(args.port)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    lock = threading.Lock()
    stats = {"done": 0, "failed": 0, "chars": 0}

    def run(job):
        if job["id"] in done:
            return
        prompt = tmpl.render(messages=[{"role": "user", "content": job["text"]}],
                             add_generation_prompt=True, enable_thinking=True,
                             reasoning_effort=job["effort"])
        n_predict = args.long_n_predict if job["task"] == "long" else args.n_predict
        try:
            resp = complete(args.port, prompt, n_predict, args.seed + stats["done"])
        except Exception as exc:  # noqa: BLE001
            with lock:
                stats["failed"] += 1
            print("failed %s: %s" % (job["id"], exc), flush=True)
            return
        text = (resp.get("content") or "").strip()
        if not text:
            with lock:
                stats["failed"] += 1
            return
        rec = {
            "id": job["id"],
            "task": job["task"],
            "source": job["source"],
            "effort": job["effort"],
            "prompt": prompt,
            "response": text,
            "n_predict": n_predict,
            "timings": resp.get("timings", {}),
        }
        with lock:
            handles[job["task"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
            handles[job["task"]].flush()
            stats["done"] += 1
            stats["chars"] += len(prompt) + len(text)
        if stats["done"] % 20 == 0:
            print("done=%d failed=%d chars=%d"
                  % (stats["done"], stats["failed"], stats["chars"]), flush=True)

    try:
        if not wait_health(args.port, proc, args.startup_timeout):
            print("server did not become healthy; see %s" % (out_dir / "server.log"),
                  file=sys.stderr)
            return 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            list(pool.map(run, jobs))
    finally:
        for fh in handles.values():
            fh.close()
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    manifest = {
        "model": pathlib.Path(args.model).name,
        "scale": args.scale,
        "jobs": len(jobs),
        "done": stats["done"],
        "failed": stats["failed"],
        "chars": stats["chars"],
        "tasks": {t: sum(1 for j in jobs if j["task"] == t) for t in TASKS},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
