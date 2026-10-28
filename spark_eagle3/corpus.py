"""Corpus readers shared by the vocabulary builder and the feature extractor.

Each reader yields plain text strings and honours a character budget so a large
prose dump cannot starve the rest of a mixed run. `skip_chars` lets a caller
start part-way into a source, which is how the training driver tiles several
segments over the same corpora without re-reading the already used prefix.
"""

from __future__ import annotations

import json
import pathlib

__all__ = ["iter_coig_cqia", "iter_firefly", "iter_wikipedia", "iter_text_files",
           "iter_code_dirs", "CODE_SUFFIXES"]


CODE_SUFFIXES = {".py", ".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".sh",
                 ".md", ".json", ".yaml"}


def _jsonl_texts(path, parts, budget_chars: int, skip_chars: int):
    """Yield the joined fields of each JSON object, honouring skip/budget.

    A record that straddles the skip boundary is emitted whole; the counters are
    approximate on purpose, because over-sampling one document does not matter
    and half a document does.
    """
    consumed = 0
    emitted = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = "\n".join(p for p in (rec.get(k) or "" for k in parts) if p)
            if not text:
                continue
            consumed += len(text)
            if consumed <= skip_chars:
                continue
            yield text
            emitted += len(text)
            if emitted >= budget_chars:
                return


def iter_coig_cqia(path, budget_chars: int, skip_chars: int = 0):
    """COIG-CQIA: {"instruction", "input", "output", ...} per line."""
    return _jsonl_texts(path, ("instruction", "input", "output"),
                        budget_chars, skip_chars)


def iter_firefly(path, budget_chars: int, skip_chars: int = 0):
    """firefly-train-1.1M: {"kind", "input", "target"} per line."""
    return _jsonl_texts(path, ("input", "target"), budget_chars, skip_chars)


def iter_wikipedia(path, budget_chars: int, skip_chars: int = 0, field: str = "text"):
    """wikimedia/wikipedia parquet shard; one article per row."""
    import pyarrow.parquet as pq

    consumed = 0
    emitted = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256, columns=[field]):
        for text in batch.column(0).to_pylist():
            if not text:
                continue
            consumed += len(text)
            if consumed <= skip_chars:
                continue
            yield text
            emitted += len(text)
            if emitted >= budget_chars:
                return


def iter_text_files(paths, budget_chars: int, skip_chars: int = 0):
    consumed = 0
    emitted = 0
    for path in paths:
        text = open(path, encoding="utf-8", errors="replace").read()
        if not text:
            continue
        consumed += len(text)
        if consumed <= skip_chars:
            continue
        yield text
        emitted += len(text)
        if emitted >= budget_chars:
            return


def iter_code_dirs(dirs, budget_chars: int, skip_chars: int = 0,
                   suffixes=CODE_SUFFIXES):
    """Source files under one or more directories, in sorted order."""
    consumed = 0
    emitted = 0
    for root in dirs:
        root = pathlib.Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text:
                continue
            consumed += len(text)
            if consumed <= skip_chars:
                continue
            yield text
            emitted += len(text)
            if emitted >= budget_chars:
                return
