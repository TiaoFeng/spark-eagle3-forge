# Core facts (do not re-derive)

This is the long form of `AGENTS.md` §2: model properties, the cost model, the vocabulary ceiling,
and what llama.cpp needs. All of it is portable; none of it needs to be worked out again.

---

## 2. Core facts

### 2.1 Model and runtime

```
spark2_5: block 36 / hidden 2560 / ffn 10240 / heads 16 / kv_heads 4 / head_dim 256
          sliding_window 512, 3:1 → 27 sliding layers + 9 full layers
          rope.base 5e6 (global rotation on only 64 dims), rope_freq_base_swa 1e4
          tied embedding (the GGUF has no output.weight) / vocab 131072 / ctx 1048576 (model maximum;
          deployment keeps 131072 and must not shrink it)
GGUF: 290 tensors = 2 + 36×8; attn_qkv is fused as [2560, 6144]
```

Note that `attn_qkv` is fused `[2560, 6144]` (q 4096 / k 1024 / v 1024) and `attn_gate` is
per-head `[2560, 16]`. **If a conversion back to HF safetensors is ever needed, this is the only
thing that requires hand work.**

The design target is a single machine with a **12 GiB-class GPU and ordinary desktop RAM**. The
portable constraints that follow:

| Constraint | Consequence |
|---|---|
| VRAM ≤ 12 GiB | the target (~6 GiB during extraction, ~8 GiB near full context) and the head in training state (~4 GiB) **cannot be resident together** — the desktop and CUDA context push the total past 12 GiB → two stages |
| tmpfs defaults to 50% of RAM | the feature buffer **must not** rely on `/dev/shm` |
| often no swap | overcommit goes straight to the OOM killer (no clean error) → check capacity up front |
| CUDA often not on PATH | builds must be told where `nvcc` is |

---

### 2.2 Cost model (the core formula)

The verification pass costs a measured **+12%**:

$$\text{speed-up} = \frac{1+p_0}{1.12} \qquad \text{break-even } p_0 \approx 0.12$$

| p0 | 0.05 | 0.12 | 0.25 | 0.33 | 0.43 | 0.55 |
|---|---:|---:|---:|---:|---:|---:|
| speed-up | 0.94x | 1.00x | 1.12x | 1.19x | 1.28x | 1.38x |

Verified conclusions about the inference parameters:

- **`n-max` is always 1**: 2 and 3 are worse on every workload
- `--spec-draft-p-min` 0.3/0.6/0.8 has no net benefit (it drags chat-without-thinking back up
  while eating the thinking gain)
- `probabilistic` vs `greedy` is within noise; **greedy is the default**, probabilistic is only a
  control at temp 1.0
- the `draft-eagle3` draft **always takes an argmax** (`common/speculative.cpp:869`)
  → **the training target must be the target model's argmax**

---

### 2.3 The hard ceiling on p0: a 32K draft vocabulary

The head emits one value per draft-vocabulary id (32000 for v8/S2a, 48000 for S2b); the runtime
scatters them back into the 131072 target vocabulary with `d2t` and everything else is
`-INFINITY`. Therefore

$$p_0 \le \text{coverage} \equiv \Pr[\text{target argmax} \in \text{draft vocab}]$$

v8's 32K vocabulary is almost entirely non-Chinese: a 171-token Chinese prompt hits it only 4.1% of
the time, and `，` / `的` / `在` are all missing. Over 2002 real workload steps, Chinese tokens are
52.0% of the output and hit 1.1% of the time; pure-ASCII tokens are 48.0% and hit 49.9% — an
aggregate **0.245**, below the 0.33 that G4 needs.

Those same steps use only 722 distinct tokens, so **32K is ample; the problem is which tokens were
picked**. A rebuilt 32K vocabulary (84.8% of its ids contain CJK) lifts coverage **0.245 → 0.939**.

Measured afterwards (`notes.md` §7d): the head trained on that vocabulary reaches p0 **0.3159** on
the same eight prompts. On protocol P its coverage is 0.882, so it converts ≈ 0.36 of that ceiling
(0.3159 / 0.882), against v8's ≈ 0.41 (0.1480 / 0.3590). Coverage is a
necessary condition, not a promise.

`d2t`'s `ne[0]` sets the draft vocabulary size (`src/models/eagle3.cpp:47`), so changing it **needs
no runtime change**, but the rows of `lm_head` change meaning → no warm start for those rows, they
have to be retrained.

Measurement method and per-workload numbers are in `notes.md` §2, §2b, §2c.

---

### 2.4 llama.cpp changes required

- `spark2_5` never sets `res->t_layer_inp[il]`, so EAGLE-3 aborts on a `GGML_ASSERT`
  → add `res->t_layer_inp[il] = inpL;` in `src/models/spark2-5.cpp` (local branch only)
- `llama_set/get_embeddings_layer_inp()` are declared in `src/llama-ext.h`, a **private header**,
  and the declarations are not inside `extern "C"` → the symbols are C++-mangled
  → the `csrc/` shim must be compiled as **C++**, declare the prototypes by hand, and export a
  plain C ABI
- the buffer is **F32**, sized `n_embd × n_batch`, one per layer (`src/llama-context.cpp:2409`)
- `llama_set_embeddings_layer_inp()` arms `sched_need_reserve` → all layers must be set before the
  first decode
