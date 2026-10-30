#!/usr/bin/env python3
"""Aggregate an acceptance matrix (D0/D1) into tables.

Reads the JSONL written by ``scripts/d0_matrix.py`` and reports, per
temperature and workload:

  * baseline  -- post-sampling top-1 and top-10 mass, i.e. the ceiling
  * greedy    -- eagle3 acceptance and how much of that ceiling it converts
  * probabilistic -- the same for sampled drafts with rejection

Acceptance is token weighted (sum accepted / sum generated) because that is the
quantity the speed-up formula consumes.

Usage:
  python scripts/d0_report.py --requests logs/d0_requests.jsonl \
      --out-md logs/d0_report.md --out-json logs/d0_report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

WORKLOAD_ORDER = ["chat", "chatnt", "zh2en", "en2zh", "sumzh", "sumen",
                  "org", "tech", "code", "struct", "longctx"]


def load(path: pathlib.Path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def group(rows, key_fn):
    out = defaultdict(list)
    for row in rows:
        out[key_fn(row)].append(row)
    return out


def spec_stats(rows):
    accepted = sum(r.get("accepted") or 0 for r in rows)
    generated = sum(r.get("generated") or 0 for r in rows)
    seconds = sum(r.get("seconds") or 0.0 for r in rows)
    predicted = sum(r.get("n_predicted") or 0 for r in rows)
    scored = [r for r in rows if r.get("generated")]
    return {
        "requests": len(rows),
        "steps": generated,
        "accepted": accepted,
        "p0": accepted / generated if generated else None,
        "mean_request_p0": (sum(r["p0"] for r in scored) / len(scored)) if scored else None,
        "tok_per_s": predicted / seconds if seconds > 0 else None,
        "seconds": seconds,
    }


def base_stats(rows):
    scored = [r for r in rows if r.get("n_scored")]
    steps = sum(r["n_scored"] for r in scored)
    top1 = sum(r["top1"] * r["n_scored"] for r in scored) / steps if steps else None
    top10 = sum(r["top10"] * r["n_scored"] for r in scored) / steps if steps else None
    n_in = sum(r.get("n_in_vocab") or 0 for r in scored)
    seconds = sum(r.get("seconds") or 0.0 for r in rows)
    predicted = sum(r.get("n_predicted") or 0 for r in rows)
    return {
        "requests": len(rows),
        "steps": steps,
        "top1": top1,
        "top10": top10,
        "coverage": n_in / steps if steps else None,
        "tok_per_s": predicted / seconds if seconds > 0 else None,
    }


def fmt(value, digits=3):
    return "-" if value is None else ("%." + str(digits) + "f") % value


def workload_key(row):
    # new rows carry "workload"; old D0/D1 logs carry the historical "register"
    return row.get("workload") or row.get("register") or row.get("name", "?")


def ordered(workloads):
    known = [r for r in WORKLOAD_ORDER if r in workloads]
    return known + sorted(r for r in workloads if r not in WORKLOAD_ORDER)


def table(title, temps, workloads, stats_of):
    lines = ["### " + title, ""]
    lines.append("| temp | workload | n | steps | value | secondary | tok/s |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for temp in temps:
        for wl in workloads:
            stat = stats_of(temp, wl)
            if stat is None:
                continue
            lines.append("| %.2f | %s | %d | %d | %s | %s | %s |" % (
                temp, wl, stat["requests"], stat["steps"],
                fmt(stat["primary"]), fmt(stat["secondary"]),
                fmt(stat["tok_per_s"], 1)))
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="logs/d0_requests.jsonl")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    rows = load(pathlib.Path(args.requests))
    if not rows:
        raise SystemExit("no rows in %s" % args.requests)

    temps = sorted({float(r["temp"]) for r in rows})
    by_mode = group(rows, lambda r: r["mode"])
    base_by = {}
    for key, recs in group(by_mode.get("baseline", []),
                           lambda r: (float(r["temp"]), workload_key(r))).items():
        base_by[key] = base_stats(recs)
    greedy_by = {}
    for key, recs in group(by_mode.get("greedy", []),
                           lambda r: (float(r["temp"]), workload_key(r))).items():
        greedy_by[key] = spec_stats(recs)

    lines = ["# acceptance matrix", ""]
    summary = {"temps": temps, "modes": {}, "workloads": {}}

    for mode in ("baseline", "greedy", "probabilistic"):
        if mode not in by_mode:
            continue
        by = group(by_mode[mode], lambda r: (float(r["temp"]), workload_key(r)))
        workloads = ordered({wl for _t, wl in by})

        def stats_of(temp, wl, by=by):
            recs = by.get((temp, wl))
            if not recs:
                return None
            if mode == "baseline":
                s = base_stats(recs)
                return {"requests": s["requests"], "steps": s["steps"],
                        "primary": s["top1"], "secondary": s["top10"],
                        "tok_per_s": s["tok_per_s"], "raw": s}
            s = spec_stats(recs)
            ceiling = base_by.get((temp, wl))
            if mode == "greedy":
                ceil = ceiling["top1"] if ceiling else None
            else:
                # the interesting comparison for probabilistic drafting is the
                # greedy acceptance at the same temperature, not the (near 1.0)
                # top-10 mass
                greedy = greedy_by.get((temp, wl))
                ceil = greedy["p0"] if greedy else None
            ratio = (s["p0"] / ceil) if (ceil and s["p0"] is not None) else None
            return {"requests": s["requests"], "steps": s["steps"],
                    "primary": s["p0"], "secondary": ratio,
                    "tok_per_s": s["tok_per_s"], "raw": s, "ceiling": ceil}

        title = {
            "baseline": "baseline: ceiling (top-1 | top-10 mass)",
            "greedy": "greedy draft: acceptance | p0 / top-1",
            "probabilistic": "probabilistic draft: acceptance | vs greedy",
        }[mode]
        lines += table(title, temps, workloads, stats_of)

        # overall, token weighted across every workload
        lines.append("**%s, all workloads**" % mode)
        lines.append("")
        lines.append("| temp | n | steps | value | secondary | tok/s |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for temp in temps:
            recs = [r for r in by_mode[mode] if float(r["temp"]) == temp]
            if not recs:
                continue
            if mode == "baseline":
                s = base_stats(recs)
                lines.append("| %.2f | %d | %d | %s | %s | %s |" % (
                    temp, s["requests"], s["steps"], fmt(s["top1"]),
                    fmt(s["top10"]), fmt(s["tok_per_s"], 1)))
                summary["modes"].setdefault(mode, {})[temp] = s
            else:
                s = spec_stats(recs)
                if mode == "greedy":
                    b = base_stats([r for r in by_mode["baseline"]
                                    if float(r["temp"]) == temp])
                    ceil = b["top1"]
                else:
                    g = spec_stats([r for r in by_mode["greedy"]
                                    if float(r["temp"]) == temp])
                    ceil = g["p0"]
                ratio = (s["p0"] / ceil) if (ceil and s["p0"] is not None) else None
                lines.append("| %.2f | %d | %d | %s | %s | %s |" % (
                    temp, s["requests"], s["steps"], fmt(s["p0"]),
                    fmt(ratio), fmt(s["tok_per_s"], 1)))
                s["ceiling"] = ceil
                s["ratio"] = ratio
                summary["modes"].setdefault(mode, {})[temp] = s
        lines.append("")

    if "baseline" in by_mode:
        lines.append("### baseline: draft-vocabulary coverage of the target top-1")
        lines.append("")
        lines.append("| temp | workload | n | steps | coverage |")
        lines.append("|---:|---|---:|---:|---:|")
        cov_by = group(by_mode["baseline"],
                       lambda r: (float(r["temp"]), workload_key(r)))
        workloads = ordered({wl for _t, wl in cov_by})
        for temp in temps:
            for wl in workloads:
                recs = cov_by.get((temp, wl))
                if not recs:
                    continue
                s = base_stats(recs)
                lines.append("| %.2f | %s | %d | %d | %s |" % (
                    temp, wl, s["requests"], s["steps"], fmt(s["coverage"])))
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out_md:
        pathlib.Path(args.out_md).write_text(text + "\n", encoding="utf-8")
    if args.out_json:
        pathlib.Path(args.out_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
