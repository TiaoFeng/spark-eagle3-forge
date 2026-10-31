#!/usr/bin/env bash
# Head checkpoint -> quantised draft GGUF for llama.cpp's draft-eagle3 loader.
#
# Why this works: llama.cpp builds the eagle3 draft from plain create_tensor() +
# build_lora_mm(), exactly like a normal model, so the draft weights accept any
# GGML type; the only type assertion in src/models/eagle3.cpp is that `d2t` stays
# I64. bf16 was only ever our exporter's choice.
#
# Why Q8_0 is the default: measured on the S2b head over the 48-prompt NLP set
# (128 tokens each, temp 1.0, ctx 32768, greedy draft, n-max 1), speed relative to
# the same server without a draft head:
#   head      file on disk   accept   mean len   vs no head
#   bf16      500.0 MiB      0.4598   1.470      1.166x
#   Q8_0      268.2 MiB      0.4595   1.469      1.217x
#   Q4_K_M    182.2 MiB      0.4616   1.46       1.227x
# The argmax survives 8-bit quantisation untouched; the draft pass is cheap, so
# Q6_K/Q4_K_M buy under 1% more and are not worth the extra risk. Details in
# docs/notes.md §12.
#
# usage:
#   scripts/quantize_head.sh <ckpt.pt> <out.gguf> [QUANT]
#   QUANT=Q6_K scripts/quantize_head.sh checkpoints/s2b.pt out.gguf
#
# needs: .venv (torch/gguf), $LLAMACPP_ROOT (or LLAMA=<path to the llama CLI>),
#        --tokenizer-from defaults to $MODELS_DIR/spark-eagle3-v8.gguf
set -euo pipefail

CKPT="${1:?usage: $0 <ckpt.pt> <out.gguf> [QUANT]}"
OUT="${2:?usage: $0 <ckpt.pt> <out.gguf> [QUANT]}"
QUANT="${3:-${QUANT:-Q8_0}}"
TOK="${TOKENIZER_FROM:-${MODELS_DIR:-}/spark-eagle3-v8.gguf}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LLAMA="${LLAMA:-${LLAMACPP_ROOT:-}/build/bin/llama}"

[[ -x "$PY" ]]        || { echo "missing python: $PY" >&2; exit 1; }
[[ -x "$LLAMA" ]]     || { echo "missing llama CLI: $LLAMA (set LLAMACPP_ROOT or LLAMA)" >&2; exit 1; }
[[ -f "$CKPT" ]]      || { echo "missing checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$TOK" ]]       || { echo "missing tokenizer source: $TOK (set TOKENIZER_FROM)" >&2; exit 1; }

TMP="$OUT.bf16.tmp.gguf"
trap 'rm -f "$TMP"' EXIT

cd "$ROOT"
echo "[1/3] export $CKPT -> bf16 gguf"
PYTHONPATH="$ROOT" "$PY" -m spark_eagle3.export_gguf \
    --ckpt "$CKPT" --out "$TMP" --tokenizer-from "$TOK"

echo "[2/3] quantize -> $QUANT"
"$LLAMA" quantize "$TMP" "$OUT" "$QUANT"

echo "[3/3] verify"
PYTHONPATH="$ROOT" "$PY" - "$TMP" "$OUT" <<'PY'
import sys
import numpy as np
from gguf import GGUFReader

src = GGUFReader(sys.argv[1])
dst = GGUFReader(sys.argv[2])
a = {t.name: t for t in src.tensors}
b = {t.name: t for t in dst.tensors}
missing, extra = set(a) - set(b), set(b) - set(a)
assert not missing and not extra, (missing, extra)

d2t = np.array_equal(np.asarray(a["d2t"].data), np.asarray(b["d2t"].data))
assert d2t, "d2t changed"

kept, quantised = [], []
for name, t in a.items():
    src_type = t.tensor_type.name
    dst_type = b[name].tensor_type.name
    if t.data.ndim == 1:
        assert src_type == dst_type, (name, src_type, dst_type)
        kept.append(name)
    else:
        quantised.append((name, src_type, dst_type))

assert quantised, "nothing was quantised"
assert all(d != s for _, s, d in quantised), "a matrix kept its bf16 type"

print("  tensors: %d (%d vectors kept, %d matrices quantised)" %
      (len(a), len(kept), len(quantised)))
print("  d2t identical: %s" % d2t)
print("  e.g. %s: %s -> %s" % quantised[0])
PY

ls -l "$OUT" | awk '{printf "  %s  %.1f MiB\n", $9, $5/1048576}'
echo "done: $OUT"
