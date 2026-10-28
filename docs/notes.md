# Design notes

Portable derivations, measurements and conclusions only. Machine-specific figures (absolute paths,
t/s, capacity) live in `AGENTS.local.md`.

Read `AGENTS.md` first for the plan; this file carries the evidence behind it. Sections are in the
order the work happened, so the numbering is not sequential — §2x belongs to the vocabulary
investigation, §3–§5 to the shim and pipeline.

---

## Baseline: v8 in Chinese registers

Six fixed workloads, 256 tokens each. `p0` is the share of first drafted tokens the target
accepts; it is hardware independent, so it compares across machines. Noise is about ±0.03.

| Workload | p0 @ temp 0.7 | p0 @ temp 1.0 |
|---|---:|---:|
| Chinese prose, no thinking | 0.013 | 0.012 |
| Chinese prose, thinking | 0.257 | 0.058 |
| Chinese chat, no thinking | 0.072 | 0.072 |
| Chinese chat, thinking | **0.328** | **0.171** |
| Code | 0.324 | 0.356 |
| Rewriting an existing file | 0.429 | 0.416 |

Three immediately useful conclusions:

- On the same prompts, **dropping temp 1.0 → 0.7 almost doubles `chat_think` p0** (0.171 → 0.328)
  → lowering the temperature is a free speed-up.
- Prose without thinking (0.013) is far below the 0.12 break-even → enabling speculative decoding
  there is **slower**.
- `n-max` 2 and 3 are worse on every workload; `--spec-draft-p-min` 0.3/0.6/0.8 does nothing.

On the inference side (the hardware-independent part): decode is **purely bandwidth-bound** (each
token reads `3.379 GB of weights + n_past × 36 KiB of KV`, counting only the 9 full-attention
layers; the 27 sliding layers need a 512 window). `GGML_CUDA_FORCE_MMQ=1` / `FORCE_CUBLAS=1` made
no difference; `--spec-type ngram-simple` gives only +3~9% (a free gain, but not this project's
focus), and `ngram-mod` is worse.

---

## 2026-10-28: v8 checkpoint verification, and the shim passing G0

### 1. Upstream artefacts (two earlier claims corrected)

| Item | Previously written | Measured |
|---|---|---|
| Head weight precision | "the head weights are BF16" | all **14/14 tensors are F32** and `config.json` says `torch_dtype = float32`; bf16 only appears in the converted GGUF (`--outtype bf16`) |
| Draft vocabulary | not mentioned | `draft_vocab_size = 32000`, `lm_head.weight = [32000, 2560]` |

Supporting confirmation:

- `fc.weight` is `[2560, 7680]` in HF (GGUF `ne0 = 7680`, same thing), i.e. 3 layers × 2560.
- `d2t` is **delta-encoded**: `target[i] = d2t[i] + i`; all 32000 ids are distinct and within
  `[0, 131071]`.
- `t2d` fills all 99,072 out-of-set ids with **0** (a placeholder, not `-1`). llama.cpp **never
  reads `t2d`**; it only scatters with `d2t` via `ggml_set_rows`, setting non-draft logits to
  `-INFINITY` (`src/models/eagle3.cpp:307`).
- The official `llama.cpp-spark2_5-layer-inputs.patch` is **semantically identical** to the local
  branch change (ours carries three extra comment lines), so the patch itself needs no further
  scrutiny.

### 2. Hard ceiling: the draft vocabulary is 32K and holds almost no Chinese

The EAGLE-3 head has 32000 output dimensions and the verifier only ever sees those candidates:

$$p_0 \le \text{coverage} \equiv \Pr\big[\text{target argmax} \in \text{draft vocab}\big]$$

Measured with `sp_gen_cli`, which generates 256 tokens through the **model's own chat template**
(temp 0.7, top-p 0.9) and checks per token whether the sampled token and the argmax fall inside the
32K set. Cross-checked with a static measurement (`sp_feat_cli --text`) over the generated text;
the two agree within 0.02.

| Workload (256 generated tokens) | sampled in-set | argmax in-set |
|---|---:|---:|
| chat_think (maths, heavy ASCII/LaTeX) | 0.8008 | 0.7891 |
| chat_nothink (Chinese small talk) | 0.0769 | 0.0705 |
| prose_think (Chinese prose, thinking) | 0.0391 | 0.0391 |
| prose_nothink (Chinese prose, no thinking) | 0.0465 | 0.0407 |
| code | 0.7734 | 0.7695 |
| copy | 0.5430 | 0.5508 |

The cause is not a register gap, it is that **the vocabulary contains no Chinese**:

- A 171-token Chinese prose prompt has **7** tokens inside the 32K set (4.1%), all whitespace or
  ASCII.
- `，`(280), `的`(276), `在`(397), `。`(302) and `有`(5053) are **all absent**.
- The ids we reconstruct with `target[i] = d2t[i] + i` are **element-wise identical to the `d2t`
  shipped in the official GGUF**, so this is not a decoding mistake.
- The earlier static "Chinese coverage 0.30" was misleading: `AGENTS.md` is full of English, paths,
  tables and code, and those were the hits.

Implications (this changed the project's direction):

- The p0 ceiling for Chinese prose is **≈ 0.04** (measured 0.013, already close to it).
- `chat_think`'s 0.79 is an artefact of a maths problem dominated by ASCII; a genuinely Chinese
  thinking trace should land around 0.05–0.15.
- In other words, **fine-tuning the existing head cannot meaningfully improve Chinese**. G4
  (`chat_think` p0 ≥ 0.33) is unreachable without changing the vocabulary.
- The only effective lever is **rebuilding `d2t` so it covers CJK**.

### 2b. The real workload (Simplified Chinese prompts, mixed-language thinking)

The probe set above was my own construction and sits at one end of the register space. A second
measurement used 8 prompts close to actual use (thinking on, 256 tokens each, temp 0.7):

| Workload | coverage | CJK share of output |
|---|---:|---:|
| zh_tech (explain the KV cache) | **0.813** | 0.0% |
| zh_math (filling a pool) | 0.495 | 14.3% |
| zh_code (write a script) | 0.215 | 55.5% |
| zh_translate (English → Chinese) | 0.191 | 49.6% |
| zh_summary (three-point summary) | 0.109 | 69.5% |
| zh_advice (career advice) | 0.094 | 70.7% |
| zh_rewrite (polish a sentence) | 0.082 | 71.1% |
| zh_daily (small talk) | **0.004** | 78.9% |
| **aggregate (2002 steps)** | **0.245** | **52.0%** |

Split by token class over the same steps:

| Class | Share | Hit rate inside the 32K set |
|---|---:|---:|
| Tokens containing CJK | 52.0% | **1.1%** |
| Pure ASCII tokens | 48.0% | **49.9%** |

Two conclusions:

1. **v8's 32K vocabulary covers only half of ASCII** (49.9%). It is not an "English vocabulary" but
   a slice of one specific corpus — presumably high-frequency tokens from the author's own English
   and code data.
2. Chinese tokens are essentially absent (1.1%). So when the model happens to think in English the
   result is good (`zh_tech` emits 0% CJK, coverage 0.813), and the moment thinking turns Chinese it
   collapses to 0.004–0.11.

→ The realistic p0 ceiling is about **0.245**, i.e. **1.11x**, still short of G4's 1.19x.
But 2002 steps used only **722 distinct tokens** (the top 522 cover 90%), which proves the 32K
**capacity** is more than enough and the problem is purely **which tokens were chosen**: a rebuilt
vocabulary should reach 0.9 or better.

Cost structure of changing the vocabulary (important):

| Option | `lm_head` shape | What can warm start |
|---|---|---|
| Keep 32K and the old d2t | 32000×2560 | everything |
| Re-pick 32K from our own corpora | 32000×2560 | fc / decoder / norm yes; `lm_head` retrained (rows change meaning) |
| Widen to 64K | 64000×2560 (+81.9M parameters) | same as above |

llama.cpp takes the draft vocabulary size from `d2t`'s `ne[0]` (`src/models/eagle3.cpp:47`), so the
**size is variable** and no runtime change is needed.

### 2c. Verifying the rebuilt vocabulary (table lookup only, no training)

Corpora come from `scripts/fetch_corpora.sh` (into `datasets/raw/`, untracked). All three files
match the LFS sha256 published by HuggingFace and parse cleanly:

| Source | Size | Sampling budget | Tokens |
|---|---:|---:|---:|
| COIG-CQIA (`m-a-p/COIG-CQIA`) | 149 MB | 40.0M chars | 23.5M |
| firefly-train-1.1M (`YeungNLP/…`) | 1.17 GB | 40.0M chars | 25.6M |
| zh-wikipedia (`wikimedia/wikipedia`, shard 1/6) | 587 MB | 40.0M chars | 28.5M |
| llama.cpp sources | — | 23.3M chars | 8.3M |

`Chinese-FineWeb-Edu` and `CCI3-Data` require an HF login (401 / gated), so they were not used.

Method: an **equal character budget per domain** (so a long-form dump cannot swamp the dialogue
data), tokenised with **the target's own tokenizer**, top 32K kept, bos/eos forced in. Of the
32,000 ids in the new vocabulary, **27,134 contain CJK (84.8%)**.

Scored on the 2002 steps of the 8 realistic probes (pure table lookup, no model needed):

| | v8 32K | rebuilt 32K |
|---|---:|---:|
| Chinese token hit rate | 0.011 | **0.974** |
| ASCII token hit rate | 0.499 | **0.900** |
| Overall | 0.245 | **0.939** |
| Speed-up ceiling | 1.11x | **1.73x** |

The only regression is `zh_tech` (which thinks entirely in English), 0.813 → 0.797; everything else
improves substantially.

This is a **ceiling**, not p0: v8 currently converts coverage into acceptance at about 40%
efficiency (code 0.77 → p0 0.32). A head with the same efficiency would sit near p0 0.37 →
**1.22x**, above G4's 1.19x.

### 2d. Warm-start checkpoint for the new vocabulary

The parameter names in `spark_eagle3/model.py` match the v8 safetensors exactly (`fc` /
`midlayer.*` / `norm` / `lm_head`), so a released checkpoint loads with `strict=True` and an export
needs no name-mapping table.

`scripts/init_head.py` copies v8's 13 parameter tensors verbatim and rebuilds only `lm_head`: tokens
present in both vocabularies keep their v8 row, the rest start from small random weights.

| Item | Value |
|---|---:|
| `lm_head` rows kept from v8 | 4,243 / 32,000 (13.3%) |
| Re-initialised | 27,757 |
| Parameters | 218.2M |
| Forward smoke | `[2,8,7680]` → `[2,8,2560]` → logits `[2,8,32000]`, all finite |

A 13% overlap is expected: v8's 32K is almost entirely ASCII while the new vocabulary is 84.8%
CJK. But those 4,243 rows already cover 90% of the ASCII tokens in the workload, so code and
English behaviour does not start from zero.

### 3. G0: feature self-consistency (passed)

`layer_inp[0]` is the hidden state right after embedding and before the first attention block, so
it must equal the same row from `llama_model_get_tok_embd_rows()`:

| Metric | Value |
|---|---:|
| max \|diff\| | 2.95e-4 |
| mean \|diff\| | 2.71e-5 |
| Regression slope (feature ÷ embd) | 1.000004 |

The error magnitude is exactly bf16 rounding → **layer indexing, token-major layout and the bf16
cast are all correct at once**, and Spark does **not** scale its embeddings (no `sqrt(n_embd)`
factor). The minimum pairwise max|diff| across the three layers is 0.71, confirming they are not
degenerate.

G0 is two orders of magnitude cheaper than G1: no training, no online comparison, one decode.

### 4. Runtime semantics the implementation must follow

- `common/speculative.cpp:747` states it plainly: the draft at memory position P consumes
  **(token[P+1], g_embd[P])**. → records must be **shift-left**: `feature[p]` pairs with
  `token[p+1]`, and the label is `argmax[p+1]`.
- `llama_set_embeddings_layer_inp()` sets `sched_need_reserve = true` (the source comment notes that
  acceptance drops to zero without that reserve) → all layers must be set before the first
  `llama_decode()`.
- `llama_get_embeddings_layer_inp()` synchronises internally.
- `n_outputs_max` defaults to `n_batch` → the whole batch's logits can be fetched at once, costing
  `n_vocab × n_batch × 4 B`. So **the argmax is computed inside the shim** and logits never cross
  the boundary.
- `embeddings_layer_inp` is unrelated to `cparams.embeddings`: the output buffer is allocated even
  with `embeddings` off.

### 5. Shim (`csrc/`) state

- It must be compiled as **C++**: `llama_get/set_embeddings_layer_inp` have C++ linkage (`nm -D`
  shows `_Z30llama_get_embeddings_layer_inpP13llama_contextj`), so a plain C translation unit cannot
  link against them.
- Approach: declare the two private prototypes by hand (not including `src/llama-ext.h`, which
  drags in `llama-arch.h`) and export a plain C ABI.
- Implemented: `sp_create / sp_destroy / sp_set_layers / sp_tokenize / sp_decode / sp_reset /
  sp_row_stride / sp_tok_embd_rows`.
  `sp_decode()` performs decode, multi-layer extraction (F32 → BF16 RNE), per-row argmax and writes
  the `[bf16 × n_layer_feat][int32 token][int32 argmax]` layout in one call.
- `sp_feat_cli` is the M1/G0 harness: prints JSON, supports `--dump-argmax` / `--dump-feat`, and
  computes coverage directly when given `--d2t`.

### 5b. Extraction pipeline (S1)

`spark_eagle3/extract.py`:

1. corpus → `tokenize()` (`corpus.py` provides COIG / firefly / wikipedia / plain-text readers)
2. each chunk takes `chunk+1` tokens (the extra one gives the last row its `p+1` label)
3. `shim.decode(slice, feat_ptr=store.row_ptr(n))` ← features go **straight into the memfd**, never
   through Python
4. `store.write_tails()` writes `token[p+1]` / `argmax[p+1]` in one shot through a strided numpy view

`FeatureStore` in `feats.py` uses `os.memfd_create` + `mmap`, so features are pure RAM, never touch
NVMe, and are not bounded by the `/dev/shm` quota; capacity is checked with `require()` before
writing, so an oversize run errors out instead of being OOM-killed.

**Known trade-off**: every chunk is decoded independently (KV cleared, positions restart at 0).
That keeps extraction bounded and simple, at the cost of the first few hundred tokens of a chunk
seeing less context than a live long-context session would give them. If G1 shows a mismatch, the
first thing to change is to decode contiguously within one document.

S1 smoke (32,768 tokens, chunk 2048, 225 chunks): label coverage **0.967**, the four sampled rows
all finite, minimum pairwise max|diff| across layers 2.25 (layer indexing correct), per-layer
std = [0.038, 0.132, 0.894] (the residual stream grows with depth, as expected).

### 5c. G1: offline p0 ≈ online p0 (passed)

Method: the same prompts and the same v8 head, greedy (temp 0) on both sides.

- Offline: `spark_eagle3/eval_p0.py` generates a greedy sequence with libllama while writing each
  row's features into the memfd, then replays the draft forward in PyTorch and maps the 32K draft
  argmax back to target ids via `d2t`.
- Online: `scripts/g1_online.py` starts `llama-server` with
  `--spec-type draft-eagle3 --spec-draft-n-max 1`, posts the same prompts to `/completion`, and
  parses the `draft acceptance` line.

| prompt | offline p0 | online p0 | diff | coverage |
|---|---:|---:|---:|---:|
| chat_think | 0.2891 | 0.3263 | +0.037 | 0.828 |
| chat_nothink | 0.0391 | 0.0244 | −0.015 | 0.063 |
| prose_think | 0.0547 | 0.0413 | −0.013 | 0.117 |
| code | 0.1172 | 0.1053 | −0.012 | 0.625 |
| copy | 0.2344 | 0.1651 | −0.069 | 0.742 |
| **mean** | **0.1469** | **0.1325** | **−0.014** | 0.475 |

Gate (mean difference < 0.03): **passed** — aggregate difference **0.0144**, mean absolute
per-prompt difference 0.0293.

Sources of the gap (they do not change the conclusion, but are worth remembering):

1. llama.cpp verifies one draft per step and the accepted bonus token never counts towards the draft
   statistics, so the online sample is ~generated tokens ÷ mean accepted length (95 steps / 126
   tokens), while the offline replay scores every position.
2. Greedy paths diverge slightly between fp32 on CPU and bf16 on GPU, and once a sequence forks the
   remaining positions are no longer comparable.
3. `copy` is verbatim repetition, where the draft is unusually strong, so small divergences get
   amplified; it is the only outlier (−0.069).

Conclusion: **the chain from feature extraction through head forward, `d2t` and label semantics is
correct**. G1 exists to catch fatal mistakes — wrong layer index, shifted labels, `d2t` wired
backwards — all of which would drive p0 to zero or to chance; the ordering and magnitude across the
five registers now match the online measurement.

---

## Open questions

1. ~~**Draft vocabulary**~~ — **resolved in §2c**: the rebuilt 32K vocabulary reaches 0.939
   coverage on the recorded steps, and `checkpoints/head_init.pt` already carries its `lm_head`
   rows. Changing the vocabulary again would invalidate that checkpoint, so the decision is
   frozen unless coverage turns out to be the binding constraint after training.
2. **Feature precision bf16 vs F32**: v8's weights are F32, bf16 is purely a capacity trade
   (2.79M vs 1.4M tokens). Train two short runs on the same data and compare p0.
3. **Contiguous KV within a chunk**: keeping it is closer to live long-context behaviour, but the
   nine full-attention layers slow down with depth; resetting per chunk truncates the first 512
   tokens of context.

## Next step

S2a: implement `spark_eagle3/train.py`, extract 2.79M tokens of features, train 8k steps, then
measure G3 (`chat_think` ≥ 0.25) and G3b (`code` ≥ 0.356) with `eval_p0.py`.
