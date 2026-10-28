"""Corpus readers shared by the vocabulary builder and the feature extractor.

Each reader yields plain text strings and honours a character budget so a large
prose dump cannot starve the rest of a mixed run.
"""

from __future__ import annotations

import json

__all__ = ["iter_coig_cqia", "iter_firefly", "iter_wikipedia", "iter_text_files"]


def iter_coig_cqia(path, budget_chars: int):
    """COIG-CQIA: {"instruction", "input", "output", ...} per line."""
    used = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = "\n".join(p for p in (rec.get("instruction") or "",
                                         rec.get("input") or "",
                                         rec.get("output") or "") if p)
            if not text:
                continue
            yield text
            used += len(text)
            if used >= budget_chars:
                return


def iter_firefly(path, budget_chars: int):
    """firefly-train-1.1M: {"kind", "input", "target"} per line."""
    used = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = "\n".join(p for p in (rec.get("input") or "",
                                         rec.get("target") or "") if p)
            if not text:
                continue
            yield text
            used += len(text)
            if used >= budget_chars:
                return


def iter_wikipedia(path, budget_chars: int, field: str = "text"):
    """wikimedia/wikipedia parquet shard; one article per row."""
    import pyarrow.parquet as pq

    used = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256, columns=[field]):
        for text in batch.column(0).to_pylist():
            if not text:
                continue
            yield text
            used += len(text)
            if used >= budget_chars:
                return


def iter_text_files(paths, budget_chars: int):
    used = 0
    for path in paths:
        text = open(path, encoding="utf-8", errors="replace").read()
        if not text:
            continue
        yield text
        used += len(text)
        if used >= budget_chars:
            return
