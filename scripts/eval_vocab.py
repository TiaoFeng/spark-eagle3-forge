#!/usr/bin/env python3
"""Score a candidate draft vocabulary against recorded generation steps.

Every step is (prompt index, sampled token, argmax token, contains-CJK). Since
`p0 <= Pr[target token in draft vocab]`, the in-set rate over recorded steps is
the coverage ceiling that vocabulary can reach on that workload -- measured
without touching the model again.

Usage:
  python scripts/eval_vocab.py logs/real/steps.i32 datasets/v8/d2t_target_ids.i32 \
      datasets/vocab/d2t_v1_32k.i32
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

PROBES = ["zh_tech", "zh_daily", "zh_math", "zh_code",
          "zh_rewrite", "zh_summary", "zh_advice", "zh_translate"]


def load_mask(path: str) -> np.ndarray:
    ids = np.fromfile(path, dtype=np.int32).astype(np.int64)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("vocabs", nargs="+")
    ap.add_argument("--labels", default="",
                    help="comma-separated probe labels; defaults to the eight realistic prompts")
    args = ap.parse_args()
    labels = args.labels.split(",") if args.labels else PROBES

    steps = np.fromfile(args.steps, dtype=np.int32).reshape(-1, 4)
    pidx, tok, arg, cjk = (steps[:, i] for i in range(4))
    n_vocab = int(max(tok.max(), arg.max())) + 1

    vocabs = {}
    for path in args.vocabs:
        ids = load_mask(path)
        mask = np.zeros(1 << 18, dtype=bool)
        mask[ids] = True
        vocabs[pathlib.Path(path).name] = mask
        print("%-24s size=%d" % (pathlib.Path(path).name, ids.size))

    header = "%-14s %5s %7s" % ("probe", "n", "cjk%")
    for name in vocabs:
        header += " %10s" % name.replace("d2t_", "").replace(".i32", "")[:10]
    print()
    print(header)

    for i, probe in enumerate(labels):
        m = pidx == i
        if not m.any():
            continue
        row = "%-14s %5d %6.1f%%" % (probe, int(m.sum()), 100.0 * cjk[m].mean())
        for mask in vocabs.values():
            row += " %10.4f" % mask[tok[m]].mean()
        print(row)

    print()
    for label, sel in (("ALL", np.ones(len(steps), bool)),
                       ("CJK only", cjk == 1),
                       ("ASCII only", cjk == 0)):
        row = "%-14s %5d %6.1f%%" % (label, int(sel.sum()), 100.0 * cjk[sel].mean())
        for mask in vocabs.values():
            row += " %10.4f" % mask[tok[sel]].mean()
        print(row)

    print()
    print("speed-up estimate = (1 + p0) / 1.12, p0 <= coverage:")
    for name, mask in vocabs.items():
        cov = mask[tok].mean()
        print("  %-24s ceiling p0=%.3f -> %.2fx" % (name, cov, (1 + cov) / 1.12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
