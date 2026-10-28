"""EAGLE-3 draft head for Spark (one decoder layer + fc + reduced lm_head).

Parameter names mirror the v8 safetensors exactly, so a published checkpoint
loads with ``load_state_dict(..., strict=True)`` and an export can be written
back without a name-mapping table:

    fc.weight                              [hidden, 3 * hidden]
    midlayer.hidden_norm.weight            [hidden]
    midlayer.input_layernorm.weight        [hidden]
    midlayer.self_attn.{q,k,v,o}_proj.weight
    midlayer.post_attention_layernorm.weight
    midlayer.mlp.{gate,up,down}_proj.weight
    norm.weight                            [hidden]
    lm_head.weight                         [draft_vocab, hidden]

Shapes follow llama.cpp's eagle3 graph: the decoder consumes
``cat(input_layernorm(fc_out), hidden_norm(prev_hidden))`` (2 * hidden = 5120),
and the vocabulary axis is the *draft* vocabulary, mapped back to target ids by
``d2t``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Attention(nn.Module):
    def __init__(self, hidden: int, n_head: int, n_head_kv: int, head_dim: int,
                 rope_theta: float):
        super().__init__()
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(2 * hidden, n_head * head_dim, bias=False)
        self.k_proj = nn.Linear(2 * hidden, n_head_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(2 * hidden, n_head_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * head_dim, hidden, bias=False)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [b, n, heads, head_dim]
        d = self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, d, 2, device=x.device,
                                                           dtype=torch.float32) / d))
        freqs = torch.outer(positions.to(torch.float32), inv_freq)  # [n, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)                     # [n, d]
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        return x * cos + rotate_half(x) * sin

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.n_head, self.head_dim)
        k = self.k_proj(x).view(b, n, self.n_head_kv, self.head_dim)
        v = self.v_proj(x).view(b, n, self.n_head_kv, self.head_dim)

        q = self._rope(q, positions)
        k = self._rope(k, positions)

        # broadcast kv heads over query heads
        rep = self.n_head // self.n_head_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [b, h, n, d]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None)
        out = out.transpose(1, 2).reshape(b, n, self.n_head * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg["hidden_size"]
        self.hidden_norm = RMSNorm(hidden, cfg["rms_norm_eps"])
        self.input_layernorm = RMSNorm(hidden, cfg["rms_norm_eps"])
        self.self_attn = Attention(hidden, cfg["num_attention_heads"],
                                   cfg["num_key_value_heads"], cfg["head_dim"],
                                   cfg["rope_theta"])
        self.post_attention_layernorm = RMSNorm(hidden, cfg["rms_norm_eps"])
        self.mlp = MLP(hidden, cfg["intermediate_size"])

    def forward(self, input_embeds: torch.Tensor, hidden_states: torch.Tensor,
                positions: torch.Tensor, attn_mask: torch.Tensor | None = None):
        residual = input_embeds
        x = self.input_layernorm(input_embeds)
        h = self.hidden_norm(hidden_states)
        x = torch.cat((x, h), dim=-1)
        x = self.self_attn(x, positions, attn_mask)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Eagle3Head(nn.Module):
    """features: [b, n, n_layer_feat * hidden] from the target's layers 2/18/33."""

    def __init__(self, cfg: dict):
        super().__init__()
        hidden = cfg["hidden_size"]
        self.n_layer_feat = len(cfg["eagle_aux_hidden_state_layer_ids"])
        self.fc = nn.Linear(self.n_layer_feat * hidden, hidden, bias=False)
        self.midlayer = DecoderLayer(cfg)
        self.norm = RMSNorm(hidden, cfg["rms_norm_eps"])
        self.lm_head = nn.Linear(hidden, cfg["draft_vocab_size"], bias=False)

    def forward(self, features: torch.Tensor, hidden_states: torch.Tensor,
                positions: torch.Tensor, attn_mask: torch.Tensor | None = None):
        x = self.fc(features)
        x = self.midlayer(x, hidden_states, positions, attn_mask)
        return self.norm(x)


def config_from_json(cfg: dict) -> dict:
    out = dict(cfg)
    out.setdefault("head_dim", out["hidden_size"] // out["num_attention_heads"])
    out.setdefault("rms_norm_eps", 1e-6)
    out.setdefault("rope_theta", 1_000_000.0)
    return out
