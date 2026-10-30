"""Stage 3: trained head -> GGUF for llama.cpp's draft-eagle3 loader.

Two conventions are not guessable from the tensor names, and both were settled by
diffing the published v8 GGUF against its safetensors:

* ``d2t`` is stored **absolute** (one target id per draft row). The safetensors
  stores it delta-encoded, so writing the delta would shift almost every draft id
  to a different token and quietly destroy acceptance.
* ``attn_q`` / ``attn_k`` are **rope-permuted over the output axis**, exactly the
  ``LlamaModel.permute()`` that the HF converter applies. Without it the file
  still loads, and the head simply drafts worse.

Everything else is a rename: ``model.py`` uses the safetensors parameter names on
purpose, so the map below is the whole conversion.

Usage:
  python -m spark_eagle3.export_gguf --ckpt checkpoints/s2a.pt \
      --out "$MODELS_DIR/spark-eagle3-s2a.gguf" \
      --tokenizer-from "$MODELS_DIR/spark-eagle3-v8.gguf"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from gguf import (GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter)

# checkpoint parameter -> GGUF tensor, and how it has to be transformed
TENSOR_MAP = (
    ("fc.weight", "fc.weight", "mat"),
    ("midlayer.input_layernorm.weight", "blk.0.attn_norm.weight", "norm"),
    ("midlayer.hidden_norm.weight", "blk.0.attn_norm_2.weight", "norm"),
    ("midlayer.self_attn.q_proj.weight", "blk.0.attn_q.weight", "perm_q"),
    ("midlayer.self_attn.k_proj.weight", "blk.0.attn_k.weight", "perm_k"),
    ("midlayer.self_attn.v_proj.weight", "blk.0.attn_v.weight", "mat"),
    ("midlayer.self_attn.o_proj.weight", "blk.0.attn_output.weight", "mat"),
    ("midlayer.post_attention_layernorm.weight", "blk.0.ffn_norm.weight", "norm"),
    ("midlayer.mlp.gate_proj.weight", "blk.0.ffn_gate.weight", "mat"),
    ("midlayer.mlp.up_proj.weight", "blk.0.ffn_up.weight", "mat"),
    ("midlayer.mlp.down_proj.weight", "blk.0.ffn_down.weight", "mat"),
    ("norm.weight", "output_norm.weight", "norm"),
    ("lm_head.weight", "output.weight", "mat"),
)

DEFAULT_TOKENIZER_FROM = "spark-eagle3-v8.gguf"


def permute(w: torch.Tensor, n_head: int, n_head_kv: int) -> torch.Tensor:
    """llama.cpp's rope permutation (conversion/llama.py LlamaModel.permute)."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
             .swapaxes(1, 2)
             .reshape(w.shape))


def bf16_bytes(t: torch.Tensor) -> np.ndarray:
    """[..] torch -> numpy uint16 carrying the bf16 pattern."""
    return t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()


def decode_gguf_tensor(t) -> np.ndarray:
    """GGUF tensor back to float32, whatever its dtype."""
    data = np.array(t.data, copy=True)
    if t.tensor_type == GGMLQuantizationType.BF16:
        if data.dtype == np.uint8:
            data = data.view(np.uint16)
        return (data.astype(np.uint32) << 16).view(np.float32)
    return data.astype(np.float32)


def copy_tokenizer(reader: GGUFReader, writer: GGUFWriter) -> list[str]:
    """Reuse the tokenizer of an existing head (or target) GGUF verbatim."""
    copied = []
    for name, field in reader.fields.items():
        if not name.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            writer.add_key_value(name, list(field.contents()), vtype, field.types[1])
        else:
            writer.add_key_value(name, field.contents(), vtype)
        copied.append(name)
    return copied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="training checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="output .gguf")
    ap.add_argument("--tokenizer-from", default="",
                    help="GGUF whose tokenizer.* metadata is reused "
                         "(default: %s)" % DEFAULT_TOKENIZER_FROM)
    ap.add_argument("--name", default="",
                    help="general.name; default: the output filename stem")
    ap.add_argument("--size-label", default="",
                    help="general.size_label; default: the parameter count of the checkpoint")
    ap.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    ap.add_argument("--skip-weights", action="store_true",
                    help="write an all-zero head (layout smoke test)")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    cfg = blob["config"]
    d2t = blob["d2t"].to(torch.int64).numpy()

    n_head = cfg["num_attention_heads"]
    n_head_kv = cfg["num_key_value_heads"]
    hidden = cfg["hidden_size"]
    n_ff = cfg["intermediate_size"]
    head_dim = cfg.get("head_dim", hidden // n_head)
    target_vocab = cfg["vocab_size"]
    layers = cfg["eagle_aux_hidden_state_layer_ids"]

    tok_from = args.tokenizer_from
    if not tok_from:
        import os
        models = os.environ.get("MODELS_DIR", "")
        tok_from = str(pathlib.Path(models) / DEFAULT_TOKENIZER_FROM) if models else ""
    if not tok_from or not pathlib.Path(tok_from).exists():
        sys.exit("no tokenizer source: pass --tokenizer-from <existing head/target gguf>")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_params = sum(int(v.numel()) for k, v in sd.items() if k != "d2t")
    name = args.name or out.stem
    size_label = args.size_label or ("%.1fM" % (n_params / 1e6))

    w = GGUFWriter(str(out), "eagle3")
    kv = {
        "general.type": (GGUFValueType.STRING, "model"),
        "general.name": (GGUFValueType.STRING, name),
        "general.size_label": (GGUFValueType.STRING, size_label),
        "general.file_type": (GGUFValueType.UINT32, 32 if args.dtype == "bf16" else 0),
        "general.quantization_version": (GGUFValueType.UINT32, 2),
        "eagle3.target_layers": (GGUFValueType.ARRAY, list(layers), GGUFValueType.INT32),
        "eagle3.target_hidden_size": (GGUFValueType.UINT32, hidden),
        "eagle3.norm_before_residual": (GGUFValueType.BOOL, False),
        "eagle3.norm_before_fc": (GGUFValueType.BOOL, False),
        "eagle3.block_count": (GGUFValueType.UINT32, 1),
        "eagle3.context_length": (GGUFValueType.UINT32,
                                  int(cfg.get("max_position_embeddings", 4096))),
        "eagle3.embedding_length": (GGUFValueType.UINT32, hidden),
        "eagle3.feed_forward_length": (GGUFValueType.UINT32, n_ff),
        "eagle3.attention.head_count": (GGUFValueType.UINT32, n_head),
        "eagle3.attention.head_count_kv": (GGUFValueType.UINT32, n_head_kv),
        "eagle3.rope.freq_base": (GGUFValueType.FLOAT32, float(cfg["rope_theta"])),
        "eagle3.attention.layer_norm_rms_epsilon": (
            GGUFValueType.FLOAT32, float(cfg.get("rms_norm_eps", 1e-6))),
        "eagle3.attention.key_length": (GGUFValueType.UINT32, head_dim),
        "eagle3.attention.value_length": (GGUFValueType.UINT32, head_dim),
        "eagle3.rope.dimension_count": (GGUFValueType.UINT32, head_dim),
        "eagle3.vocab_size": (GGUFValueType.UINT32, target_vocab),
    }
    for key, spec in kv.items():
        if spec[0] == GGUFValueType.ARRAY:
            w.add_key_value(key, spec[1], GGUFValueType.ARRAY, spec[2])
        else:
            w.add_key_value(key, spec[1], spec[0])

    with open(tok_from, "rb") as _fh:
        copied = copy_tokenizer(GGUFReader(tok_from), w)
    print("tokenizer fields copied: %d" % len(copied), flush=True)

    written = []
    for src, dst, how in TENSOR_MAP:
        if src not in sd:
            sys.exit("checkpoint is missing %s" % src)
        t = sd[src].detach().to(torch.float32)
        if args.skip_weights:
            t = torch.zeros_like(t)
        if how == "perm_q":
            t = permute(t, n_head, n_head)
        elif how == "perm_k":
            t = permute(t, n_head, n_head_kv)
        if how == "norm" or args.dtype == "float32":
            w.add_tensor(dst, t.contiguous().numpy().astype(np.float32))
        else:
            w.add_tensor(dst, bf16_bytes(t), raw_dtype=GGMLQuantizationType.BF16)
        written.append(dst)

    w.add_tensor("d2t", np.ascontiguousarray(d2t, dtype=np.int64))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s (%.0f MB)" % (out, out.stat().st_size / 2**20), flush=True)

    # -- verify: read back, compare every tensor with the checkpoint --------
    r = GGUFReader(str(out))
    got = {t.name: t for t in r.tensors}
    missing = [d for _s, d, _h in TENSOR_MAP if d not in got]
    if missing or "d2t" not in got:
        sys.exit("export is missing tensors: %s" % (missing + ([] if "d2t" in got else ["d2t"])))
    worst = 0.0
    worst_name = ""
    for src, dst, how in TENSOR_MAP:
        ref = sd[src].detach().to(torch.float32)
        if how == "perm_q":
            ref = permute(ref, n_head, n_head)
        elif how == "perm_k":
            ref = permute(ref, n_head, n_head_kv)
        if args.dtype == "bf16" and how != "norm":
            ref = ref.to(torch.bfloat16).to(torch.float32)
        t = got[dst]
        back = decode_gguf_tensor(t).reshape(tuple(int(v) for v in t.shape[::-1]))
        diff = float(np.abs(back - ref.numpy()).max())
        if diff > worst:
            worst, worst_name = diff, dst
    if not np.array_equal(np.array(got["d2t"].data, copy=True), d2t):
        sys.exit("d2t did not round-trip")
    print("round-trip: %d tensors, d2t exact, worst |diff| %.3e on %s"
          % (len(TENSOR_MAP) + 1, worst, worst_name), flush=True)
    print(json.dumps({"out": str(out), "tensors": len(r.tensors),
                      "name": name, "size_label": size_label,
                      "parameters": n_params,
                      "draft_vocab_size": int(d2t.size),
                      "file_type": "bf16" if args.dtype == "bf16" else "float32"},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
