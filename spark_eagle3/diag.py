"""Offline replay helpers for the P0 diagnostics (`scripts/p0_diag.py`).

None of these helpers train anything. They replay a recorded path through the
draft head in PyTorch the same way `eval_p0.py` does for protocol P, so the
numbers are hardware independent and comparable with the published gates.

Runtime conventions, both confirmed against `src/models/eagle3.cpp` in the
patched llama.cpp checkout:

  * `fc` lives in the **encoder** graph. The encoder runs once per verified
    token on the three target layers and stores `g_embd`; the **decoder** graph
    consumes `g_embd` directly, so a self-fed draft step feeds the head's prenorm
    hidden *without* `fc`.
  * the exported `t_h_nextn` is the decoder's prenorm hidden (after the decoder
    layer, before `output_norm`/`lm_head`), and the draft loop feeds it back as
    `g_embd` for the next drafted token, together with the target tokenizer's
    embedding of the token the draft itself chose.

Three quantities are produced here:

``alpha``
    ``sum_x min(p, q)`` between the target's deployment sampling law and the
    head's distribution. It upper-bounds *probabilistic* speculative decoding
    and is the only quantity that can exceed the argmax ceiling (0.869 on the
    D1 mix, `docs/notes.md` §11f).

``p1``
    the runtime's second draft step, decomposed into the four token/hidden
    variants so the cost of never training on the head's own hidden state
    (§12d) becomes measurable instead of inferred from `mean len`.

``context drift``
    p0 as a function of session context length: features are extracted with a
    16k window, and the deployment context is 131072.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "softmax_probs", "truncate_support", "TargetSampler", "generate",
    "collect_rows", "head_pass", "token_embeds", "alpha_of_row",
    "find_marker", "thinking_marker", "long_prompt",
]


# ---------------------------------------------------------------------------
# the deployment sampling law
# ---------------------------------------------------------------------------
def softmax_probs(logits, temp: float = 1.0) -> np.ndarray:
    """float64 softmax of ``logits / temp`` (one row or a 2-D batch)."""
    x = np.asarray(logits, dtype=np.float64)
    if temp != 1.0:
        x = x / float(temp)
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def truncate_support(probs, top_p: float = 0.95, min_p: float = 0.05):
    """llama.cpp's ``top_p`` then ``min_p``, renormalised.

    `common/common.h` orders the chain ``top_p -> min_p -> temperature``. With
    ``temp = 1.0`` the temperature stage is a no-op, so the retained set is the
    shortest descending prefix whose cumulative probability reaches ``top_p``
    (`llama_sampler_top_p_apply`, crossing token included) intersected with
    ``p >= min_p * p_max`` (`llama_sampler_min_p_apply`, always keeping one).
    """
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs)
    sp = probs[order]
    k = int(np.searchsorted(np.cumsum(sp), top_p, side="left")) + 1
    k = min(max(k, 1), sp.size)
    ids, p = order[:k], sp[:k]
    if min_p > 0.0:
        keep = p >= min_p * p[0]
        keep[0] = True
        ids, p = ids[keep], p[keep]
    return ids.astype(np.int32), p / p.sum()


class TargetSampler:
    """The deployment sampler: temp 1.0, top_p 0.95, min_p 0.05, top_k off.

    `scripts/d0_matrix.py` sends exactly these values, and `docs/notes.md` §8b
    is the reason: the daily server and the gate harness had drifted apart once
    already, so the diagnostics reproduce the deployed law instead of inventing
    one.
    """

    def __init__(self, temp: float = 1.0, top_p: float = 0.95,
                 min_p: float = 0.05, seed: int = 1234):
        self.temp = float(temp)
        self.top_p = float(top_p)
        self.min_p = float(min_p)
        self.rng = np.random.default_rng(seed)

    def support(self, logits):
        return truncate_support(softmax_probs(logits, self.temp),
                                self.top_p, self.min_p)

    def sample(self, logits):
        """Return (token, support_ids, support_probs)."""
        ids, p = self.support(logits)
        return int(self.rng.choice(ids, p=p)), ids, p


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def generate(shim, store, prompt_ids, n_gen: int, sampler: TargetSampler | None = None,
             collect_support: bool = False):
    """Decode a path and fill ``store`` with one feature row per token.

    ``sampler is None`` means greedy, which is protocol P. The prompt is
    prefilled in ``n_batch`` chunks with logits requested only for the last one:
    the context allocates ``n_vocab`` floats per batch entry, so asking for the
    whole prompt at once would be GiB at 128k vocabulary.

    Returns a dict with ``tokens`` (prompt + generated), ``argmax`` (the target's
    own argmax at each position), ``n_prompt``, ``n_total`` and ``support``, the
    per-row ``(ids, probs)`` of the target's sampling law when sampling.
    """
    ids = np.asarray(prompt_ids, dtype=np.int32)
    n_prompt = int(ids.size)
    n_total = min(n_prompt + int(n_gen), int(shim.n_ctx))
    if n_prompt < 2:
        raise ValueError("prompt too short")
    if n_total <= n_prompt:
        raise ValueError("prompt fills the context (%d tokens)" % n_prompt)

    toks = np.zeros(n_total, dtype=np.int32)
    toks[:n_prompt] = ids[:n_prompt]
    argmax = np.full(n_total, -1, dtype=np.int32)

    shim.reset()
    store.require(n_total)

    step = max(1, int(shim.n_batch))
    last_beg = ((n_prompt - 1) // step) * step
    for beg in range(0, n_prompt, step):
        end = min(beg + step, n_prompt)
        if end == n_prompt:
            argmax[beg:end] = shim.decode(toks[beg:end], beg,
                                          feat_ptr=store.row_ptr(beg),
                                          want_logits=True)
        else:
            # prompt rows are never scored, and argmax needs logits
            shim.decode(toks[beg:end], beg, feat_ptr=store.row_ptr(beg),
                        want_logits=False)

    support: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if sampler is None:
        nxt = int(argmax[n_prompt - 1])
    else:
        nxt, ids_s, p_s = sampler.sample(shim.logits(n_prompt - 1 - last_beg))
        if collect_support:
            support[n_prompt - 1] = (ids_s, p_s)

    n_done = n_prompt
    while n_done < n_total:
        toks[n_done] = nxt
        a = shim.decode(toks[n_done:n_done + 1], n_done,
                        feat_ptr=store.row_ptr(n_done), want_logits=True)
        argmax[n_done] = a[0]
        n_done += 1
        if shim.is_eog(nxt) or n_done >= n_total:
            break
        if sampler is None:
            nxt = int(argmax[n_done - 1])
        else:
            nxt, ids_s, p_s = sampler.sample(shim.logits(0))
            if collect_support:
                support[n_done - 1] = (ids_s, p_s)

    return {"tokens": toks, "argmax": argmax, "n_prompt": n_prompt,
            "n_total": n_done, "support": support}


def collect_rows(store, shim, n_total: int, first_p: int, last_p: int,
                 n_layer_feat: int):
    """Copy the scored rows out of the store.

    Row ``p`` holds ``feature[p]`` and the shift-left tail ``(token[p+1],
    argmax[p+1])``, so it is scoreable when ``last_p <= n_total - 2``.
    """
    rows = np.arange(first_p, last_p + 1, dtype=np.int64)
    stride = store.row_bytes
    feat_bytes = n_layer_feat * shim.n_embd * 2
    raw = np.frombuffer(store.buf, dtype=np.uint8,
                        count=n_total * stride, offset=0)
    view = raw.reshape(n_total, stride)
    feats = np.array(view[rows, :feat_bytes], dtype=np.uint8, copy=True)
    tok_ids = np.array(view[rows, feat_bytes:feat_bytes + 4].view("<i4").ravel(),
                       dtype=np.int64)
    labels = np.array(view[rows, feat_bytes + 4:feat_bytes + 8].view("<i4").ravel(),
                      dtype=np.int64)
    return rows, feats, tok_ids, labels


def token_embeds(shim, ids) -> np.ndarray:
    """F32 rows of the *target* token embedding table (what the runtime feeds)."""
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    out = np.empty((ids.size, shim.n_embd), dtype=np.float32)
    shim.tok_embd_rows(ids, out)
    return out


def head_pass(head, shim, feats_bf16, tok_ids, dtype, device):
    """One causal pass over the scored rows, returning every intermediate.

    ``g`` is the fused target feature (`fc` output), ``h`` the prenorm hidden the
    runtime feeds back for a second draft step, and ``logits`` the draft-vocab
    scores.
    """
    import torch

    R = int(feats_bf16.shape[0])
    f = torch.from_numpy(np.ascontiguousarray(feats_bf16)).view(torch.bfloat16)
    f = f.reshape(1, R, -1).to(device=device, dtype=dtype)
    e = torch.from_numpy(token_embeds(shim, tok_ids)).to(device=device, dtype=dtype)
    e = e.reshape(1, R, -1)
    pos = torch.arange(R, device=device)
    with torch.no_grad():
        g = head.fc(f)
        h = head.midlayer(e, g, pos)
        logits = head.lm_head(head.norm(h))[0].float()
    return {"f": f, "e": e, "g": g, "h": h, "logits": logits}


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def alpha_of_row(q_row, p_ids, p_probs, t2d, keep_idx=None) -> float:
    """``sum_x min(p, q)`` for one position.

    ``q_row`` is already normalised over the draft vocabulary; ``p`` only has
    support on its truncated set, and everywhere else ``min`` is 0, so the sum
    runs over that support. Tokens the draft vocabulary cannot express
    (`t2d == -1`) contribute 0 by construction.

    ``keep_idx`` restricts ``q`` to those draft indices and renormalises, which
    is what the runtime's ``top_k(10)`` draft chain does. The ordering of the
    draft logits does not depend on temperature, so callers compute the indices
    once and reuse them for the whole sweep.
    """
    didx = t2d[p_ids]
    qv = np.zeros(len(p_ids), dtype=np.float64)
    if keep_idx is None:
        ok = didx >= 0
        qv[ok] = np.asarray(q_row, dtype=np.float64)[didx[ok]]
    else:
        sel = np.asarray(keep_idx)
        qs = np.asarray(q_row, dtype=np.float64)[sel]
        qs = qs / qs.sum()
        lookup = dict(zip(sel.tolist(), qs.tolist()))
        for i, d in enumerate(didx.tolist()):
            if d >= 0:
                qv[i] = lookup.get(d, 0.0)
    return float(np.minimum(np.asarray(p_probs, dtype=np.float64), qv).sum())


def find_marker(shim, gen_ids, marker: bytes):
    """Locate ``marker`` in the generated token ids.

    Returns ``(first, end)`` token indices covering the marker, or ``(-1, -1)``.
    Detecting at the byte level avoids depending on how the tokenizer splits the
    marker at a chunk boundary.
    """
    text = b""
    ends = []
    for tok in gen_ids:
        text += shim.token_to_piece(int(tok))
        ends.append(len(text))
    at = text.find(marker)
    if at < 0:
        return -1, -1
    first = next(i for i, e in enumerate(ends) if e > at)
    end = next((i for i, e in enumerate(ends) if e >= at + len(marker)), first)
    return first, end


def thinking_marker(model_path: str) -> str:
    """The literal the chat template puts between the thinking and answer channels.

    Rendered instead of hard-coded: the template is read from the GGUF, so a
    template change cannot silently split the channels in the wrong place.
    """
    import jinja2
    from gguf import GGUFReader

    reader = GGUFReader(model_path)
    field = reader.fields.get("tokenizer.chat_template")
    if field is None:
        raise RuntimeError("model has no tokenizer.chat_template")
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)

    def _raise(msg):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = _raise
    tmpl = env.from_string(str(field.contents()))
    out = tmpl.render(
        messages=[
            {"role": "user", "content": "PROBE_USER"},
            {"role": "assistant", "reasoning_content": "PROBE_REASON",
             "content": "PROBE_ANSWER"},
        ],
        add_generation_prompt=False,
        enable_thinking=True,
    )
    a = out.find("PROBE_REASON")
    b = out.find("PROBE_ANSWER")
    if a < 0 or b <= a:
        raise RuntimeError("could not locate the channel marker in the template")
    return out[a + len("PROBE_REASON"):b]


def long_prompt(shim, text: str, n_tokens: int) -> np.ndarray:
    """Repeat a document until it fills ``n_tokens`` (context-drift probe)."""
    ids = shim.tokenize(text, add_special=False)
    if ids.size == 0:
        raise ValueError("empty document")
    reps = int(np.ceil(n_tokens / ids.size))
    return np.tile(ids, reps)[:n_tokens].astype(np.int32)
