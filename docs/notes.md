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

## 6. Gate protocol for S2a (fixed before training)

`AGENTS.md` §8 quoted G3 (`chat_think` 0.17 → 0.25) and G3b (`code` 0.356) from the temp 1.0 online
table, but the gate is checked with `eval_p0.py`, which replays **greedily**. Those two numbers were
never measured on the protocol they gate, so they are replaced by the ones below. They were taken
**before any training run**, so the outcome cannot be reinterpreted afterwards.

**Protocol P.** `spark_eagle3.eval_p0`, `--dtype bfloat16 --n-gen 256`, greedy (the target decodes by
argmax, the draft is scored at every generated position), on prompts rendered with the model's own
chat template. Two prompt sets: `rendered` (the six fixed workloads) and `realistic` (the eight
Chinese thinking prompts of §2b). Offline and greedy, so the numbers are hardware independent.

**v8 baseline on P** (`logs/p0_v8_greedy_256.json`, `logs/p0_v8_greedy_realistic_256.json`):

| Workload | p0 | coverage |
|---|---:|---:|
| `chat_think` | 0.2461 | 0.7344 |
| `chat_nothink` | 0.0441 | 0.0662 |
| `prose_think` | 0.0469 | 0.0938 |
| `prose_nothink` | 0.0000 | 0.0298 |
| `code` | 0.1992 | 0.5898 |
| `copy` | 0.1562 | 0.5469 |

| realistic prompt | p0 | coverage |
|---|---:|---:|
| zh_tech | 0.3203 | 0.7930 |
| zh_math | 0.3205 | 0.6026 |
| zh_code | 0.2930 | 0.6797 |
| zh_translate | 0.1250 | 0.5972 |
| zh_rewrite | 0.0508 | 0.0703 |
| zh_summary | 0.0469 | 0.0742 |
| zh_advice | 0.0273 | 0.0547 |
| zh_daily | 0.0000 | 0.0000 |
| **mean of 8** | **0.1480** | **0.3590** |

Note that the single `chat_think` prompt measures 0.2461 at `--n-gen 256` and 0.2891 at
`--n-gen 128`: the scorer samples a different set of positions, and the spread (0.043) is larger than
the ±0.03 noise quoted elsewhere. `--n-gen 256` is the standard here because it scores twice as many
positions.

**Gates, restated as absolute targets on P:**

| Gate | Condition | Meaning if it fails |
|---|---|---|
| G3 probe | `chat_think` **≥ 0.25** | below v8's own 0.2461 → the recipe damaged the register v8 already serves; stop |
| G3b no forgetting | `code` **≥ 0.18** and `copy` **≥ 0.13** | v8 is 0.1992 / 0.1562; a drop of more than ~0.02–0.03 is forgetting, because only 4,243 of v8's 32,000 `lm_head` rows survive the vocabulary change |
| G3c the point | realistic mean **≥ 0.25** | v8 is 0.1480 there and the rebuilt vocabulary lifts the coverage ceiling from 0.245 to 0.939; a head that cannot turn that into 0.25 has not learned the new rows |
| G4 ship | `chat_think` **≥ 0.33** (unchanged) | 1.19x, the number `AGENTS.md` §8 has always carried |

G3 and G3b are now *no-regression* checks against v8 on the same protocol; G3c is the gain the
project exists for. G4 keeps its original value, so the ship criterion did not get easier.

---

## 7. S2a: the training run and the gates

### 7a. What actually ran

`spark_eagle3/train.py`, one process, eight segments, 1000 optimiser steps each:

| Item | Value |
|---|---|
| corpus | COIG-CQIA 35% / firefly 35% / zh-wikipedia 20% / llama.cpp `src` 10% (by token quota) |
| unique tokens | 8 × 1,000,000 = **8.0M** |
| chunk | 2048, four domains extracted separately with disjoint character offsets per segment |
| sample | 2 sequences × 1024 tokens per step = 2048 tokens/step, windows never cross a chunk |
| steps | 8000 → 16.4M token updates, **2.05 epochs per segment** |
| optimiser | AdamW, lr 1e-4 cosine to 10%, betas (0.9, 0.95), wd 0.01, grad clip 1.0, bf16 autocast |
| loss | `CE(argmax) + 0.5 · CE(training-time test)` |
| label coverage | 0.944–0.993 per domain-segment; masked fraction during training 0.7–2.9% |
| checkpoint | every 500 steps (parameters only), full state after each segment |
| wall clock | extraction ≈ 2/3 of the run and training ≈ 1/3; absolute times in `AGENTS.local.md` |
| train curve | CE 8.37 → 2.14, top-1 0.168 → 0.614 |

`lambda = 0.5` was chosen because the published head's `train_log.json` carries no
hyper-parameters at all (only `steps`, `params`, `coverage`, `history`), so there was nothing to
read it from.

**Why eight segments and not four.** The first attempt used four segments of 700K tokens, i.e. 5.9
epochs per segment. Within 2000 steps the training CE fell to 0.15 at top-1 0.97 while the held-out
CE of that segment rose 5.45 → 6.33: the head had memorised 700K tokens and stopped generalising. It
also died between segments (the VRAM bug in §7b), which made re-cutting the run cheap. The same
8000 steps over 8M unique tokens instead of 2.8M end at CE 2.14 / top-1 0.614, which is a healthy
distance from the ceiling, and `p0` on the real prompts kept improving to the last checkpoint.

### 7b. Two bugs the run exposed

**Scratch row overrun (fixed).** `sp_decode()` writes one row per *decoded* token, i.e. `take + 1`
rows for a chunk, but `extract_into()` reserved only `take`. The released CLI never noticed because
`--capacity-gib 2.0` always left slack; a segment sized to exactly `tokens_per_segment` rows walked
one row off the end of the mmap and took SIGSEGV. Capacity is now `tokens + 8` rows and the check
asks for `take + 1`.

**VRAM handover (fixed).** Extraction needs the 4B target (~6 GiB) and training needs the head plus
two AdamW moments (~4 GiB); on a 12 GiB card they cannot be resident together. Segment 0 was fine
because at that point only the parameters existed, so segment 1 died in `llama_init_from_model`. The
driver now parks the head and the optimiser state on the CPU for the duration of each extraction and
recreates the embedding table from the shim afterwards.

### 7c. Checkpoint curve

Four prompts, protocol P (`--n-gen 256`, greedy), the same checkpoints the run saved:

| step | chat_think | code | zh_daily | zh_tech |
|---|---:|---:|---:|---:|
| v8 (v8 vocab) | 0.2461 | 0.1992 | 0.0000 | 0.3203 |
| 500 | 0.3242 | 0.2422 | 0.1602 | 0.3594 |
| 1500 | 0.3008 | 0.2109 | 0.1953 | 0.3477 |
| 3000 | 0.3164 | 0.2031 | 0.1641 | 0.3438 |
| 5000 | 0.2930 | 0.2500 | 0.2461 | 0.3672 |
| 8000 (final) | 0.3281 | 0.2070 | 0.2891 | 0.3516 |

Two things to read out of it:

- **Almost all of the gain arrives by step 500.** `chat_think` jumps 0.246 → 0.324 within one
  segment, then sits inside the ±0.03 noise band for the remaining 7500 steps. The 27,757
  re-initialised `lm_head` rows are the whole story: once they have seen a few hundred steps of
  their own language, the head can use them.
- **Only `zh_daily` keeps climbing** (0.16 → 0.29), which is the most purely Chinese register of the
  four. Registers that already worked (code, `zh_tech`, English-ish thinking) do not improve further.

The within-segment validation loss is *not* a usable selection signal here: it drifts up inside every
segment (4.40 → 4.60 CE in segment 0, and the same shape seven more times) even though `p0` on the
real prompts ends higher than it started. Each segment validates against its own held-out slice, so
only the prompt-driven `p0` is comparable across segments.

### 7d. Gates (protocol P, `--n-gen 256`, greedy)

`rendered`, the six fixed workloads:

| workload | v8 | S2a | change | coverage (v8 → S2a) |
|---|---:|---:|---:|---|
| chat_think | 0.2461 | **0.3281** | +0.082 | 0.734 → 0.789 |
| chat_nothink | 0.0441 | **0.2353** | +0.191 | 0.066 → 0.956 |
| prose_think | 0.0469 | **0.3086** | +0.262 | 0.094 → 0.957 |
| prose_nothink | 0.0000 | **0.2024** | +0.202 | 0.030 → 0.982 |
| code | 0.1992 | **0.2070** | +0.008 | 0.590 → 0.668 |
| copy | 0.1562 | **0.2031** | +0.047 | 0.547 → 0.793 |
| **mean** | **0.1154** | **0.2474** | +0.132 | 0.343 → 0.858 |

`realistic`, the eight Chinese thinking prompts:

| prompt | v8 | S2a | change | coverage (v8 → S2a) |
|---|---:|---:|---:|---|
| zh_advice | 0.0273 | **0.2500** | +0.223 | 0.055 → 0.984 |
| zh_code | 0.2930 | 0.2852 | −0.008 | 0.680 → 0.688 |
| zh_daily | 0.0000 | **0.2891** | +0.289 | 0.000 → 0.996 |
| zh_math | 0.3205 | **0.3675** | +0.047 | 0.603 → 0.842 |
| zh_rewrite | 0.0508 | **0.3750** | +0.324 | 0.070 → 0.953 |
| zh_summary | 0.0469 | **0.4141** | +0.367 | 0.074 → 0.996 |
| zh_tech | 0.3203 | **0.3516** | +0.031 | 0.793 → 0.801 |
| zh_translate | 0.1250 | **0.1944** | +0.069 | 0.597 → 0.799 |
| **mean of 8** | **0.1480** | **0.3159** | +0.168 | 0.359 → 0.882 |

Verdicts against §6:

| Gate | Threshold | Measured | Verdict |
|---|---|---:|---|
| G3 probe | `chat_think` ≥ 0.25 | 0.3281 | **pass** |
| G3b no forgetting | `code` ≥ 0.18, `copy` ≥ 0.13 | 0.2070 / 0.2031 | **pass** |
| G3c the point | realistic mean ≥ 0.25 | 0.3159 | **pass** |
| G4 ship | `chat_think` ≥ 0.33 | 0.3281 | **miss by 0.002** |

G4 is a genuine miss, not a rounding argument: the pre-registered number for `chat_think` on
protocol P is 0.33 and the head reaches 0.3281, which is 1.186x rather than 1.19x. It is inside the
±0.03 noise of the measurement, so the same head would clear it on a rerun about half the time — but
the gate was fixed before the run and it is recorded as missed.

### 7e. G2 export, and an online A/B against v8

`spark_eagle3/export_gguf.py` writes `spark-eagle3-s2a.gguf` (422 MB, 14 tensors, bf16 matmuls and
F32 norms). The round-trip check compares every tensor against the checkpoint and `d2t` exactly; the
two conventions it has to reproduce (`d2t` absolute, `attn_q`/`attn_k` rope-permuted) are documented
in the module. llama.cpp loads it without an assert (`draft acceptance` is printed), so **G2 passes**.

The same five prompts, both heads, `--spec-type draft-eagle3 --spec-draft-n-max 1`, greedy, 128
tokens, 4096 context:

| prompt | v8 online | S2a online | change |
|---|---:|---:|---:|
| zh_daily | 0.0080 | **0.2857** | +0.278 |
| zh_rewrite | 0.0328 | **0.4157** | +0.383 |
| zh_summary | 0.0328 | **0.4270** | +0.394 |
| zh_tech | 0.3404 | 0.3125 | −0.028 |
| chat_think | 0.3263 | 0.2990 | −0.027 |
| **mean of 5** | **0.1481** | **0.3480** | +0.200 |

Wall clock over the same 5 × 128 tokens: **12.6 ms/token → 10.6 ms/token, 1.19x**. That is the first
end-to-end confirmation on this machine that the p0 gains convert into decode speed, and it agrees
with $(1+0.348)/1.12 = 1.20$.

Offline versus online for the new head at the matched generation length (128): `chat_think` 0.3359 vs
0.2990, `zh_tech` 0.3516 vs 0.3125, `zh_daily` 0.2656 vs 0.2857 → mean absolute difference 0.032,
the same order as v8's 0.029 in §5c. The export reproduces the PyTorch head; the residual gap is the
offline scorer counting every position while llama.cpp counts verification steps.

### 7f. What the numbers say

- **The vocabulary was the whole problem, exactly as §2 predicted.** Every register whose ceiling
  was limited by coverage moved by roughly what the coverage gain allows; the two registers that were
  already at 0.79–0.80 coverage (`zh_tech`, `zh_code`) did not move.
- **The cost is a small regression on English-heavy thinking.** `chat_think` and `zh_tech` are the
  only prompts that lost p0 (−0.027/−0.028 online), and they are the two where v8's ASCII vocabulary
  was already well suited. Trading ~0.03 there for +0.28…+0.39 on Chinese is the right trade for
  this project, but it means the new head should not be shipped to a purely English workload.
- **`code` is flat** (0.1992 → 0.2070). It was not the target, and only 10% of the tokens were code;
  the earlier hope that the 4,243 shared `lm_head` rows would preserve it exactly holds, but no more.
- **More steps buy Chinese small talk, not thinking.** `zh_daily` is the only held-out register that
  improved monotonically, which points at S2b (Chinese thinking traces) rather than more steps of the
  same data for a further gain.

---

## 8. D0: the deployment-matched acceptance matrix

### 8a. Why the gate number and the daily number differed

`scripts/g1_online.py` sends `temperature = 0.0`, so every S2a gate reports
greedy acceptance. A real session samples. D0 measures the same prompts under
three server configurations:

| Mode | Server | What it measures |
|---|---|---|
| baseline | no draft head | post-sampling top-1 probability and top-10 mass |
| greedy | `draft-eagle3`, n-max 1 | the head's argmax acceptance |
| probabilistic | the same + `--spec-draft-sampling probabilistic` | sampled drafts verified by rejection |

The baseline is a separate process because llama.cpp does not report
post-sampling probabilities while speculation is enabled.

### 8b. Protocol

- 46 prompts in six registers (chat, chat without thinking, technical Q&A,
  code, structured output, long context), rendered with the model's own chat
  template by `scripts/make_prompts.py --set daily`.
- temperatures 0.6 / 0.7 / 0.85 / 1.0, `top_p 0.95`, `min_p 0.05`, 128 tokens,
  seed 1234, `cache_prompt = false`.
- acceptance is token weighted: `sum(accepted) / sum(generated)`.
- the baseline asks for `n_probs = 10` with `post_sampling_probs = true`; the
  per-token maximum is the ceiling for a perfect greedy draft, and the top-10
  sum is the theoretical ceiling for a perfect probabilistic draft.
- `scripts/d0_matrix.py` writes one JSON line per request and is resumable;
  `scripts/d0_report.py` aggregates.
- the prompt set is synthetic and public; the long-context prompts use
  `docs/notes.md` and `docs/facts.md` as context.

### 8c. Results

Overall (46 prompts per temperature):

| temp | ceiling top-1 | greedy p0 | p0 / ceiling | probabilistic p0 | vs greedy |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.879 | 0.292 | 0.332 | 0.292 | 1.00 |
| 0.70 | 0.858 | 0.291 | 0.339 | 0.291 | 1.00 |
| 0.85 | 0.833 | 0.285 | 0.342 | 0.285 | 1.00 |
| 1.00 | 0.804 | 0.291 | 0.362 | 0.291 | 1.00 |

At the deployment temperature (`temp 1.0`), per register:

| register | ceiling top-1 | greedy p0 | p0 / ceiling | coverage |
|---|---:|---:|---:|---:|
| chat | 0.754 | 0.258 | 0.342 | 0.958 |
| chat, no thinking | 0.757 | 0.276 | 0.364 | 0.975 |
| technical Q&A | 0.761 | 0.238 | 0.313 | 0.771 |
| code | 0.779 | 0.249 | 0.319 | 0.749 |
| structured output | 0.882 | 0.407 | 0.462 | 0.895 |
| long context | 0.913 | 0.346 | 0.379 | 0.819 |

Per-prompt efficiency at temp 1.0 ranges from 0.19 (`daily_code_03`, a C++
ring buffer) to 0.67 (`daily_struct_04`, a Markdown table).

### 8d. What the numbers say

1. **The head is the bottleneck, not the ceiling.** The target's top-1
   probability is 0.80 at temp 1.0 and the head converts about a third of it
   (0.29). The training top-1 of S2a was 0.61, so the gap is the deployment
   distribution, not the optimiser: the head does much better on the training
   mix than on the model's own thinking and chat output.
2. **Temperature is not a lever.** With the daily sampler (`min_p 0.05`) the
   ceiling only falls from 0.879 to 0.804 between temp 0.6 and 1.0, and the
   absolute acceptance is flat (0.285-0.292). Lowering the temperature is not a
   free speed-up for this head. The earlier v8 probe reported a large
   0.7 -> 1.0 drop; that does not reproduce here, so it should not be assumed
   for the current head.
3. **Probabilistic drafting currently adds nothing.** An offline replay of the
   head gives a mean top-1 probability of 0.40 (median 0.31), so the
   distribution is not a point mass. The host-sampler pass does draft tokens
   outside the argmax (the verbose log shows a token outside the top 3), yet
   the accepted/generated counts are identical to greedy for all 184
   prompt x temperature pairs. The head is uncertain exactly where it is wrong:
   the extra samples land on positions the target rejects anyway, while at
   confident positions (p ~ 0.99) the sample is the argmax.
   - The default `--spec-draft-backend-sampling` builds a backend chain with
     `top_k(10)` only, so the draft is the argmax whatever
     `--spec-draft-sampling` says. The flag only takes effect with
     `--no-spec-draft-backend-sampling`.
4. **The vocabulary caps the ASCII-heavy registers.** Coverage of the target's
   top-1 is 0.75 (code), 0.77 (technical Q&A) and 0.82 (long context), against
   0.96-0.98 for Chinese chat. The rebuilt 32K vocabulary is 84.8% CJK, so it
   is the mirror image of v8: Chinese is covered, code and technical ASCII are
   not. That is a hard ceiling for those registers, independent of training.

### 8e. Decision for the next stage

- S2b/S3 are justified and should be weighted toward the free-form Chinese
  registers, which sit at 0.31-0.34 (technical Q&A, code, chat); structured
  output is already the strongest register (0.46).
- Before S3, rebuild the draft vocabulary at 48K-64K with a code/ASCII quota if
  code and technical text are a large share of daily tokens; code coverage is
  0.75 and technical Q&A 0.77, so those registers are capped regardless of the
  head.
- Do not ship `--spec-draft-sampling probabilistic` expecting a gain until the
  head is calibrated (soft labels or a post-hoc `output.weight` scale); the
  current distribution has no useful mass on the target's alternatives.
- Re-run D0 after S2b with the same matrix; it is the gate.

---

## 9. D1: the NLP workload matrix

### 9a. Why

The daily workload is natural-language work: document organisation, Chinese-
English translation, summarisation and chat. D0 covered chat, technical Q&A,
code, structured output and long context, but not translation, and its mix was
tuned for a code-heavy session. D1 measures the actual NLP mix.

### 9b. Protocol

- 48 prompts in seven registers: chat (6, two multi-turn), Chinese -> English
  translation (8), English -> Chinese translation (8), Chinese summary (6),
  English document -> Chinese summary (6), document organisation (8), long
  context (6).
- Sources are public repo docs (`README.md`, `README.cn.md`, `docs/notes.md`,
  `docs/facts.md`) or invented notes; rendered with the model's chat template by
  `scripts/make_prompts.py --set nlp`.
- temp 1.0 only: D0 showed that temperature is not a lever, so the matrix runs at
  the deployment setting. Baseline / greedy / probabilistic and the aggregation
  are the D0 harness (`scripts/d0_matrix.py`, `scripts/d0_report.py`).
- 144 requests, no failures.

### 9c. Results

| register | ceiling top-1 | greedy p0 | p0 / ceiling | coverage |
|---|---:|---:|---:|---:|
| chat | 0.769 | 0.276 | 0.359 | 0.977 |
| Chinese -> English | 0.915 | 0.449 | 0.491 | 0.852 |
| English -> Chinese | 0.913 | 0.325 | 0.356 | 0.814 |
| Chinese summary | 0.832 | 0.333 | 0.400 | 0.960 |
| English -> Chinese summary | 0.844 | 0.277 | 0.328 | 0.892 |
| document organisation | 0.892 | 0.421 | 0.471 | 0.935 |
| long context | 0.880 | 0.298 | 0.339 | 0.810 |
| **all** | **0.869** | **0.344** | **0.396** | **0.888** |

Probabilistic drafting is identical to greedy for all 48 prompts, the same
result as D0 (§8d): the backend sampler is top-k only, and the host sampler
still adds no accepted tokens.

### 9d. What the numbers say

1. **Translation is the strongest register, not the weakest.** Chinese -> English
   reaches 0.449 at half of its ceiling (0.491), and English -> Chinese 0.325 at
   0.356. The source text constrains the output, so the target is predictable and
   the head converts that well.
2. **Summarisation is the weakest.** English -> Chinese summary 0.277, Chinese
   summary 0.333, long-context summary/organisation 0.298, efficiency 0.33-0.40.
   A summary is a paraphrase, so the target's own choice is less constrained and
   the head has less to latch onto.
3. **Organisation is strong** (0.421, 0.471), consistent with D0's structured
   output (0.407, 0.462): constrained formats are where this head wins.
4. **Chat stays head-limited** (0.276 at coverage 0.977), exactly as in D0.
5. **The vocabulary binds the English-heavy registers.** Coverage is 0.81-0.89
   for both translation directions, English -> Chinese summary and long context,
   against 0.96-0.98 for chat and Chinese summary. The 32K vocabulary is 84.8%
   CJK, so the bilingual workload has a structural cap there.

### 9e. Decision

- **Rebuild `d2t` at 48K as a union, not a fresh build**: keep the current
  32,000 ids and append the top 16,000 English/ASCII target ids from a bilingual
  corpus. A fresh build would move the Chinese rows and throw away the S2a
  `lm_head` alignment; the union keeps all 32K rows.
- **Warm-start from `checkpoints/s2a.pt`, not v8**: copy the 32K `lm_head` rows
  and randomly initialise the 16,000 new English rows. `init_head.py` currently
  starts from v8, so a `--from-checkpoint` path is needed.
- **Then S2b with the 48K head**, weighted to the weak registers: summarisation
  30%, translation 25% (more English -> Chinese), organisation 20%, chat 20%,
  other 5%; long-context shards 30% of tokens.
- **Gates**: NLP mix mean p0 >= 0.40; English -> Chinese >= 0.40; English ->
  Chinese summary >= 0.35; coverage >= 0.90 on every register; online wall clock
  >= 1.3x. Re-run D1 after S2b.

---

## 10. D2: the 48K bilingual union vocabulary

### 10a. Why

D1 measured 0.81-0.89 coverage on both translation directions, English ->
Chinese summary and long context: the 32K vocabulary is 84.8% CJK while the daily
work is bilingual. The next structural step was a 48K vocabulary that keeps the
32K Chinese ids and adds English ones.

### 10b. Method

- Record greedy steps for the 48 NLP prompts (`scripts/record_steps.py`), in the
  same `[register, token, argmax, cjk]` layout `eval_vocab.py` consumes.
- Build the union (`scripts/build_union_vocab.py`): keep the 32,000 base ids in
  their original order, append the 16,000 most frequent missing target ids from
  one English Wikipedia shard (20M chars) plus the recorded steps at weight 50.
  Keeping the base order is what lets the S2a lm_head rows survive.
- Warm start from `checkpoints/s2a.pt`, not v8 (`scripts/init_head.py
  --from-checkpoint`): copy the 32,000 trained rows and random-initialise the
  16,000 new rows at 0.5x the old row std.
- Export and load the head to confirm the runtime accepts a 48K draft vocabulary
  (llama.cpp takes the size from `d2t.ne[0]`).

### 10c. Results

- union: 48,000 ids, 27,134 CJK base + 32 CJK new; the 16,000 new rows are
  almost entirely English/ASCII.
- in-sample (the steps used for selection): coverage 1.000. This is expected and
  is not evidence of generalisation.
- held-out realistic workload (`logs/real/steps.i32`, 2,002 steps from different
  prompts): 0.9386 -> 0.9710 overall; ASCII 0.9000 -> 0.9594; CJK 0.9741 ->
  0.9818; `zh_tech` 0.7969 -> 0.9492.
- held-out English Wikipedia slice (2M chars after the selection window, 508,553
  tokens): 0.6835 -> 0.9604.
- warm start: 32,000 rows kept from S2a, 16,000 new; 259.2M parameters.
- 48K GGUF: 500 MB, 14 tensors, `d2t` exact; llama-server loads it and reports
  draft acceptance. The smoke p0 is unchanged because the Chinese argmax stays
  inside the preserved 32K rows.

### 10d. Decision

- S2b/S3 use `datasets/vocab/d2t_v2_48k.i32` and
  `checkpoints/head_init_48k.pt`. The English rows are new and must be trained;
  the S2b mixture (summarisation 30% / translation 25% / organisation 20% /
  chat 20%) supplies the English data.
- The 48K vocabulary removes the 0.81-0.89 coverage cap measured in D1; the
  remaining gap is head quality, as in D0/D1.
- Keep the D1 matrix as the gate: re-run it with the 48K head after S2b.

---

## 11. S2b: the NLP mixture and the contiguous extractor

### 11a. Why

D1/D2 settled two things: the head is the binding constraint (it converts
about a third of the target's top-1 ceiling), and the 48K vocabulary removes
the English coverage cap. S2b therefore trains on the target's own output for
the actual workload instead of the public Chinese corpora.

### 11b. Data generation

`scripts/gen_nlp_data.py` starts a draft-free llama-server with four slots and
writes one JSONL record per request:

| task | jobs | prompt | effort |
|---|---:|---|---|
| sum | 600 | Chinese/English wiki article, 200-character summary | high 2/3, low 1/3 |
| trans | 1200 | wiki article, Chinese <-> English | high 2/3, low 1/3 |
| org | 400 | wiki article -> outline / table / JSON | high 2/3, low 1/3 |
| chat | 1200 | COIG-CQIA instruction | high |
| long | 120 | 24k-character document -> integrated summary / outline | high 2/3, low 1/3 |

The document is the rendered chat prompt plus the target's continuation, so
training sees the prompt distribution and the thinking/answer register. The
`long` records are the long-context shards (~30% of the token budget). The
generator is resumable and writes `datasets/nlp/*.jsonl`.

### 11c. Training

`scripts/train_s2b.py` differs from `train.py` in two places:

- **contiguous extraction**: `extract_contiguous_into()` decodes each document
  in 2048-token sub-batches without resetting the KV cache, up to a 16k
  context, so a feature at a response position sees the whole document prefix.
  The old `extract_into()` resets the KV per 2048-token chunk.
- **mixed-token TTT**: at each position the second pass feeds the draft's own
  argmax with probability `mix_rate` and the target token otherwise, matching
  the runtime's accept/reject, instead of a lagged self-fed chain.

The segment plan is 4 x 1M tokens, weights sum 0.20 / trans 0.20 / org 0.15 /
chat 0.15 / long 0.30, starting from `head_init_48k.pt`; 2000 steps of
2 x 1024 tokens, lr 1e-4 cosine with a 200-step warmup.

### 11d. Smoke

A 20k-token, 20-step run on a 1% generated sample:

- extraction 2,299 tok/s, 0.3 GiB resident;
- label coverage (48K vocabulary): chat 0.948, long 0.991;
- training 20 steps in 2.2 s, no failures;
- `s2b.pt` loads and the 48K logits are produced.

The full generation and training run is in progress; the measured p0 will be
recorded here when it finishes.

---

## Open questions

1. ~~**Draft vocabulary**~~ — **decided in §9**: the NLP mix measures 0.81-0.89 coverage on
   both translation directions, English -> Chinese summary and long context. The 48K union
   rebuild (keep the 32K ids, append 16K English ids, warm-start from S2a) is scheduled before
   S2b; the code/ASCII quota is no longer the target.
2. **Feature precision bf16 vs F32**: v8's weights are F32, bf16 is purely a capacity trade
   (2.79M vs 1.4M tokens). Train two short runs on the same data and compare p0.
3. **Contiguous KV within a chunk**: keeping it is closer to live long-context behaviour, but the
   nine full-attention layers slow down with depth; resetting per chunk truncates the first 512
   tokens of context.
4. **How much of the remaining gap is data volume?** 2.05 epochs per segment is still repetition;
   §7c shows the gain saturating by step 500, so the next marginal step is *more unique tokens*
   (S3, 12–16M) rather than more steps.
5. **`lambda` for the training-time test**: 0.5 was a guess (§7a). A/B it now that the pipeline is
   cheap: two 2000-step runs with 0 and 1.0 would bracket the useful range.

## Next step

D0 (§8), D1 (§9) and D2 (§10) settle the measurement and vocabulary questions: the head converts
about a third of the target's top-1 ceiling, temperature and probabilistic drafting change
nothing, and the 48K bilingual vocabulary lifts held-out coverage from 0.939 to 0.971 overall
(0.684 to 0.960 on English text). The next stage:

1. **S2b (register)**: the generator and contiguous trainer are in place; the full run is in
   progress on `d2t_v2_48k.i32` and `head_init_48k.pt` (summarisation 30% / translation 25% /
   organisation 20% / chat 20%, long-context shards 30%). A recipe check, not the final run.
2. **S3 (scale)**: 12-16M unique tokens with the S2b mixture and a last-mile pass on the daily
   mix; D1 is the gate.

Probabilistic drafting and the sampling temperature are not training levers at present; they stay
parked until the head is calibrated.
