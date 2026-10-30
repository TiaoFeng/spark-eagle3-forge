"""Stage 1: corpus -> feature shard.

Each chunk is decoded with the target model, the input hidden states of the
configured layers are copied into a memfd row buffer, and every row gets the
shift-left label pair (token[p+1], argmax[p+1]) the draft actually consumes.

Chunk boundaries: a chunk of N tokens is decoded with one extra token so the
last row still has its p+1 target. Chunks are independent (the KV cache is
cleared), which keeps extraction simple and bounded; the price is that a row
near a chunk start sees less context than it would in a live session.

`extract_into()` is also the entry point of the S2a driver (`train.py`), which
runs extraction and training in one process because the feature buffer is a
memfd and cannot outlive it.

Usage:
  python -m spark_eagle3.extract --model M.gguf --max-tokens 32768 \
      --d2t datasets/vocab/d2t_v1_32k.i32 --out logs/shards/s1.json \
      --text AGENTS.md --text docs/notes.md --coig datasets/raw/coig_cqia/COIG-CQIA-full.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from . import corpus
from .feats import FeatureStore
from .llamacpp import Shim

BASE = pathlib.Path(__file__).resolve().parent.parent


def build_sources(args):
    """Turn the CLI's source flags into a list of (name, factory) pairs.

    `args.skip_chars` shifts every source forward by that many characters, so a
    driver can tile several disjoint segments over the same corpora.
    """
    skip = getattr(args, "skip_chars", 0) or 0
    out = []
    for path in args.text or []:
        out.append(("text", lambda p=path: corpus.iter_text_files(
            [p], args.chars_per_source, skip)))
    for path in args.coig or []:
        out.append(("coig_cqia", lambda p=path: corpus.iter_coig_cqia(
            p, args.chars_per_source, skip)))
    for path in args.firefly or []:
        out.append(("firefly", lambda p=path: corpus.iter_firefly(
            p, args.chars_per_source, skip)))
    for path in args.wiki or []:
        out.append(("wikipedia_zh", lambda p=path: corpus.iter_wikipedia(
            p, args.chars_per_source, skip)))
    for path in getattr(args, "nlp", None) or []:
        out.append(("nlp", lambda p=path: corpus.iter_nlp_docs(
            [p], args.chars_per_source, skip)))
    for path in getattr(args, "code", None) or []:
        out.append(("code", lambda p=path: corpus.iter_code_dirs(
            [p], args.chars_per_source, skip)))
    return out


def extract_into(shim: Shim, store: FeatureStore, sources, *, chunk: int,
                 max_tokens: int, start_row: int = 0, in_set=None) -> dict:
    """Decode `sources` into `store` starting at `start_row`.

    Returns a report dict; `chunks` holds [first_row, n_rows] pairs so a trainer
    knows where document boundaries are and never samples across one.
    """
    n_rows = 0
    n_docs = 0
    n_chunks = 0
    label_hits = 0
    label_checked = 0
    chunks: list[list[int]] = []
    per_source: dict[str, int] = {}

    for name, factory in sources:
        before = n_rows
        for text in factory():
            if n_rows >= max_tokens:
                break
            ids = shim.tokenize(text, add_special=False)
            if ids.size < 2:
                continue
            n_docs += 1
            for start in range(0, ids.size - 1, chunk):
                if n_rows >= max_tokens:
                    break
                take = min(chunk, ids.size - 1 - start, max_tokens - n_rows)
                if take <= 0:
                    break
                slice_ids = ids[start:start + take + 1]
                # sp_decode() emits one row per decoded token, so the chunk
                # needs `take + 1` rows even though only `take` of them get a
                # shift-left label and enter the dataset
                store.require(start_row + n_rows + take + 1)

                # chunks are independent: positions restart at 0, so the KV
                # cache has to be empty or llama.cpp rejects the batch
                shim.reset()
                argmax = shim.decode(slice_ids, start_pos=0,
                                     feat_ptr=store.row_ptr(start_row + n_rows))
                store.write_tails(start_row + n_rows, slice_ids[1:], argmax[1:])
                if in_set is not None:
                    am = argmax[1:]
                    label_hits += int(in_set[am].sum())
                    label_checked += int(am.size)

                chunks.append([start_row + n_rows, int(take)])
                n_rows += int(take)
                n_chunks += 1
        per_source[name] = n_rows - before
        if n_rows >= max_tokens:
            break

    return {
        "n_tokens": n_rows,
        "n_docs": n_docs,
        "n_chunks": n_chunks,
        "chunks": chunks,
        "per_source": per_source,
        "label_coverage": (label_hits / label_checked) if label_checked else None,
    }


def extract_contiguous_into(shim: Shim, store: FeatureStore, sources, *, n_ctx: int,
                            chunk: int, max_tokens: int, start_row: int = 0,
                            in_set=None) -> dict:
    """Decode each document with a contiguous KV cache (up to `n_ctx` tokens).

    Unlike `extract_into`, sub-batches do not reset the KV cache, so a feature
    at row p sees the whole document prefix. The sub-batch size is `chunk`; the
    shim context must allow `n_ctx >= the longest document` and
    `n_batch >= chunk`. One `chunks` entry is emitted per document, so the
    trainer never samples across a document boundary.
    """
    n_rows = 0
    n_docs = 0
    n_chunks = 0
    n_sub = 0
    label_hits = 0
    label_checked = 0
    chunks: list[list[int]] = []
    per_source: dict[str, int] = {}

    for name, factory in sources:
        before = n_rows
        for text in factory():
            if n_rows >= max_tokens:
                break
            ids = shim.tokenize(text, add_special=False)
            if ids.size < 2:
                continue
            n_docs += 1
            take = min(ids.size - 1, max_tokens - n_rows, n_ctx - 1)
            if take <= 0:
                break
            doc = ids[:take + 1]
            store.require(start_row + n_rows + doc.size)

            shim.reset()
            argmax = np.empty(doc.size, dtype=np.int32)
            for start in range(0, doc.size, chunk):
                n = min(chunk, doc.size - start)
                a = shim.decode(doc[start:start + n], start_pos=start,
                                feat_ptr=store.row_ptr(start_row + n_rows + start),
                                want_logits=True)
                argmax[start:start + n] = a
                n_sub += 1

            store.write_tails(start_row + n_rows, doc[1:], argmax[1:])
            if in_set is not None:
                am = argmax[1:]
                label_hits += int(in_set[am].sum())
                label_checked += int(am.size)

            chunks.append([start_row + n_rows, int(take)])
            n_rows += int(take)
            n_chunks += 1
        per_source[name] = n_rows - before
        if n_rows >= max_tokens:
            break

    return {
        "n_tokens": n_rows,
        "n_docs": n_docs,
        "n_chunks": n_chunks,
        "n_sub_batches": n_sub,
        "chunks": chunks,
        "per_source": per_source,
        "label_coverage": (label_hits / label_checked) if label_checked else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="where to write the shard report JSON")
    ap.add_argument("--layers", default="2,18,33")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--n-ctx", type=int, default=0,
                    help="contiguous mode: context length; 0 = one independent chunk of chunk+1")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--capacity-gib", type=float, default=2.0)
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--text", action="append")
    ap.add_argument("--coig", action="append")
    ap.add_argument("--firefly", action="append")
    ap.add_argument("--wiki", action="append")
    ap.add_argument("--nlp", action="append", help="generated NLP JSONL (prompt + response)")
    ap.add_argument("--code", action="append", help="directory to mine for code")
    ap.add_argument("--chars-per-source", type=int, default=20_000_000)
    ap.add_argument("--skip-chars", type=int, default=0)
    ap.add_argument("--d2t", help="vocabulary id file, for reporting label coverage")
    ap.add_argument("--dump-rows", type=int, default=0, help="write this many raw rows")
    ap.add_argument("--dump-path", default="")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    sources = build_sources(args)
    if not sources:
        sys.exit("no input sources given")

    n_ctx = args.n_ctx if args.n_ctx > 0 else args.chunk + 1
    capacity = int(args.capacity_gib * 2**30)

    t0 = time.time()
    shim = Shim(args.model, n_gpu_layers=args.n_gpu_layers, n_ctx=n_ctx, n_batch=args.chunk)
    shim.set_layers(layers)
    store = FeatureStore(capacity, len(layers), shim.n_embd)
    store.require(args.max_tokens + 1)
    load_s = time.time() - t0

    in_set = None
    if args.d2t:
        ids = np.fromfile(args.d2t, dtype=np.int32).astype(np.int64)
        in_set = np.zeros(shim.n_vocab, dtype=bool)
        in_set[ids] = True

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    if args.n_ctx > 0:
        report = extract_contiguous_into(shim, store, sources, n_ctx=n_ctx,
                                         chunk=args.chunk, max_tokens=args.max_tokens,
                                         in_set=in_set)
    else:
        report = extract_into(shim, store, sources, chunk=args.chunk,
                              max_tokens=args.max_tokens, in_set=in_set)
    extract_s = time.time() - t1
    store.flush()

    n_rows = report["n_tokens"]
    report.update({
        "model": pathlib.Path(args.model).name,
        "layers": layers,
        "n_embd": shim.n_embd,
        "n_layer_feat": len(layers),
        "row_bytes": store.row_bytes,
        "chunk": args.chunk,
        "n_ctx": n_ctx,
        "contiguous": bool(args.n_ctx),
        "capacity_gib": args.capacity_gib,
        "load_s": round(load_s, 1),
        "extract_s": round(extract_s, 1),
        "tokens_per_s": round(n_rows / extract_s, 1) if extract_s > 0 else None,
        "bytes_used": n_rows * store.row_bytes,
    })

    if args.dump_rows and args.dump_path:
        want = min(args.dump_rows, n_rows)
        dump = np.frombuffer(store.buf, dtype=np.uint8,
                             count=want * store.row_bytes, offset=0)
        pathlib.Path(args.dump_path).write_bytes(dump.tobytes())
        del dump
        report["dumped_rows"] = want
        report["dump_path"] = args.dump_path

    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "chunks"}, indent=2))

    store.close()
    shim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

