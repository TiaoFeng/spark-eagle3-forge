# 設計筆記

只記錄**可移植**的推導、實測與結論。機器相關數據（絕對路徑、t/s、容量）見 `AGENTS.local.md`。

---

## 2026-10-28：v8 檢查點核實 + shim 的 G0 通過

### 1. 上游檔案核實（推翻兩處既有描述）

| 項目 | 原本描述 | 實測 |
|---|---|---|
| 頭權重精度 | 「該頭權重本身是 BF16」 | safetensors 內 **14/14 張量全為 F32**，`config.json` 的 `torch_dtype = float32`；GGUF 才是轉檔時 `--outtype bf16` |
| draft 詞表 | 未提及 | `draft_vocab_size = 32000`，`lm_head.weight = [32000, 2560]` |

補充確認：

- `fc.weight` 在 HF 是 `[2560, 7680]`（GGUF 的 `ne0 = 7680`，同一個東西），對應 3 層 × 2560。
- `d2t` 是 **delta 儲存**：`target[i] = d2t[i] + i`；32000 個 id 互異、全部落在 `[0, 131071]`。
- `t2d` 對集合外的 99,072 個 id **全部填 0**（佔位值，不是 `-1`）。llama.cpp **不讀 `t2d`**，只用 `d2t` 走 `ggml_set_rows` 散射，非 draft 詞表的 logits 一律設 `-INFINITY`（`src/models/eagle3.cpp:307`）。
- 官方 `llama.cpp-spark2_5-layer-inputs.patch` 與本機分支的改動**語意相同**（本機多 3 行註解），patch 正確性不必再懷疑。

### 2. 硬性上限：draft 詞表只有 32K，而且幾乎不含中文

EAGLE-3 head 的輸出維度是 32000，驗證端只會看到這 32000 個候選。因此

$$p_0 \le \text{coverage} \equiv \Pr\big[\text{target argmax} \in \text{draft vocab}\big]$$

量測方式：`sp_gen_cli` 用**模型自己的 chat template** 生成 256 token（temp 0.7、top-p 0.9），
逐 token 比較「實際取樣到的 token」與「argmax」是否落在 32K 集合內；
再用 `sp_feat_cli --text` 對生成出的文字做一次靜態量測交叉驗證（兩者差異 < 0.02）。

| 工作負載（生成 256 token） | sampled in-set | argmax in-set |
|---|---:|---:|
| chat_think（數學推理，ASCII/LaTeX 多） | 0.8008 | 0.7891 |
| chat_nothink（中文閒聊） | 0.0769 | 0.0705 |
| prose_think（中文散文，thinking） | 0.0391 | 0.0391 |
| prose_nothink（中文散文，不 thinking） | 0.0465 | 0.0407 |
| code | 0.7734 | 0.7695 |
| copy | 0.5430 | 0.5508 |

根本原因不是「語域差距」，是**詞表本身不含中文**：

- 一段 171 token 的中文散文 prompt，只有 **7 個** token 在 32K 集合內（4.1%），而且全是空白與 ASCII。
- `，`(280)、`的`(276)、`在`(397)、`。`(302)、`有`(5053) **全部不在**集合內。
- 我們用 `target[i] = d2t[i] + i` 重建的 id 與 **官方 GGUF 內的 `d2t` 完全相同**（逐元素比對通過），所以不是解讀錯誤。
- 早期用靜態文字量到的「中文 0.30」是誤導：`AGENTS.md` 內含大量英文、路徑、表格與程式碼，這些才是命中的那一批。

推論（這改變了專案方向）：

- 中文散文的 p0 天花板 **≈ 0.04**（實測 0.013，已接近上限）。
- `chat_think` 的 0.79 是**數學題偏 ASCII** 造成的假象；真正的中文 thinking trace 預期也在 0.05–0.15。
- 也就是說：**只微調現有 head，中文幾乎不可能改善**。G4（`chat_think` p0 ≥ 0.33）在不換詞表的前提下達不到。
- 唯一有效的槓桿是**重建 `d2t` 讓它涵蓋 CJK**。

### 2b. 真實使用情境（簡體中文提問、thinking 中英混合）

上面那組探針是我自己造的，語域偏一端。另外用 8 個貼近實際用法的簡體中文 prompt
（thinking 全開、各生成 256 token、temp 0.7）再量一次：

| 工作負載 | coverage | 輸出中 CJK token 佔比 |
|---|---:|---:|
| zh_tech（解釋 KV cache） | **0.813** | 0.0% |
| zh_math（注水題） | 0.495 | 14.3% |
| zh_code（寫腳本） | 0.215 | 55.5% |
| zh_translate（英翻中） | 0.191 | 49.6% |
| zh_summary（三點總結） | 0.109 | 69.5% |
| zh_advice（職涯建議） | 0.094 | 70.7% |
| zh_rewrite（潤色句子） | 0.082 | 71.1% |
| zh_daily（閒聊） | **0.004** | 78.9% |
| **加總（2002 step）** | **0.245** | **52.0%** |

按 token 種類分解（同一批 step）：

| 類別 | 佔比 | 落在 32K 集合內的比例 |
|---|---:|---:|
| 含 CJK 的 token | 52.0% | **1.1%** |
| 純 ASCII 的 token | 48.0% | **49.9%** |

兩個結論：

1. **v8 的 32K 詞表連 ASCII 都只覆蓋一半**（49.9%）——它不是「英文詞表」，而是某個特定
   語料的子集（推測是作者自己的英文／程式碼語料的高頻 token）。
2. 中文 token 幾乎完全不在集合內（1.1%）。所以「模型剛好用英文思考」時表現很好
   （zh_tech 的輸出 0% CJK、coverage 0.813），一旦 thinking 轉成中文就掉到 0.004–0.11。

→ 真實使用下的 p0 天花板約 **0.245**，對應 **1.11x**，仍低於 G4 的 1.19x。
但 2002 個 step 只用到 **722 個相異 token**（前 522 個就涵蓋 90%），
證明 32K 的**容量**綽綽有餘，問題純粹是**選錯了 token**：重建詞表後覆蓋率預期可達 0.9 以上。

換詞表的成本結構（重要）：


| 方案 | `lm_head` 形狀 | 可 warm-start 的部分 |
|---|---|---|
| 維持 32K + 舊 d2t | 32000×2560 | 全部 |
| 以自有語料重挑 32K | 32000×2560 | fc / decoder / norm 可；`lm_head` 需重訓（行語意改變） |
| 擴到 64K | 64000×2560（+81.9M 參數） | 同上 |

llama.cpp 由 `d2t` 張量的 `ne[0]` 決定 draft vocab 大小（`src/models/eagle3.cpp:47`），所以**詞表大小是可變的**，不需要改 runtime。

### 2c. 重建詞表的驗證（純查表，未訓練）

語料由 `scripts/fetch_corpora.sh` 取得（`datasets/raw/`，未入庫）。三份檔案的大小與 sha256
都與 HuggingFace 公布的 LFS oid 相符，格式也都能正常解析：

| 來源 | 檔案大小 | 抽樣預算 | token |
|---|---:|---:|---:|
| COIG-CQIA（`m-a-p/COIG-CQIA`） | 149 MB | 40.0M 字 | 23.5M |
| firefly-train-1.1M（`YeungNLP/…`） | 1.17 GB | 40.0M 字 | 25.6M |
| zh-wikipedia（`wikimedia/wikipedia` 1/6 分片） | 587 MB | 40.0M 字 | 28.5M |
| llama.cpp 原始碼 | — | 23.3M 字 | 8.3M |

`Chinese-FineWeb-Edu` 與 `CCI3-Data` 需要 HF 登入（401 / gated），因此未使用。

做法：每域**等字數預算**（避免長文壓過對話語料），用**目標模型自己的 tokenizer** 統計，
取前 32K，bos/eos 強制保留。新詞表 32,000 個 id 中有 **27,134 個含 CJK（84.8%）**。

在 8 個真實負載探針的 2002 個 step 上評估（純查表，不需載入模型）：

| | 現行 v8 32K | 重建 32K |
|---|---:|---:|
| 中文 token 命中率 | 0.011 | **0.974** |
| ASCII token 命中率 | 0.499 | **0.900** |
| 整體 | 0.245 | **0.939** |
| 加速上限 | 1.11x | **1.73x** |

唯一退步是 `zh_tech`（整段用英文思考）0.813 → 0.797，其餘全部大幅提升。

這是**上限**而非 p0：v8 目前把覆蓋率轉成接受率的效率約 40%（code 0.77 → p0 0.32）。
若新頭維持同樣效率，p0 約 0.37 → **1.22x**，已超過 G4 的 1.19x。

### 2d. 新詞表的 warm-start 檢查點

`spark_eagle3/model.py` 的參數名稱與 v8 safetensors 完全一致（`fc` / `midlayer.*` / `norm` / `lm_head`），
所以發布的檢查點可以用 `strict=True` 載入，匯出時也不需要名稱對照表。

`scripts/init_head.py` 把 v8 的 13 個參數張量原樣搬進新模型，只重建 `lm_head`：
新舊詞表重疊的 token 沿用 v8 的列，其餘以小隨機權重初始化。

| 項目 | 值 |
|---|---:|
| 沿用 v8 的 `lm_head` 列 | 4,243 / 32,000（13.3%） |
| 重新初始化 | 27,757 |
| 參數量 | 218.2M |
| 前向 smoke | `[2,8,7680]` → `[2,8,2560]` → logits `[2,8,32000]`，全 finite |

重疊只有 13% 是預期結果：v8 的 32K 幾乎全是 ASCII，新詞表 84.8% 是 CJK。
但這 4,243 列已覆蓋真實負載中 90% 的 ASCII token，所以 code／英文能力不會歸零。

### 3. G0：特徵自洽（通過）

`layer_inp[0]` 就是 embedding 之後、第一層 attention 之前的 hidden state，因此必須等於 `llama_model_get_tok_embd_rows()` 的同一列：

| 指標 | 值 |
|---|---:|
| max \|diff\| | 2.95e-4 |
| mean \|diff\| | 2.71e-5 |
| 回歸斜率（feature ÷ embd） | 1.000004 |

誤差量級正好等於 bf16 捨入 → **層索引、token-major 佈局、bf16 轉換三者同時正確**，且 Spark **不縮放** embedding（無 `sqrt(n_embd)` 因子）。三層兩兩之間的最小 max|diff| = 0.71，確認非退化。

G0 比 G1 便宜兩個數量級：不必訓練、不必比對線上 p0，一次 decode 就能判定。

### 4. 執行期語意（實作必須遵守）

- `common/speculative.cpp:747` 明文：draft 在記憶體位置 P 吃的是 **(token[P+1], g_embd[P])**。→ 訓練記錄必須 **shift-left**：`feature[p]` 配 `token[p+1]`、標籤配 `argmax[p+1]`。
- `llama_set_embeddings_layer_inp()` 會設 `sched_need_reserve = true`（原始碼註解：少了這個 reserve，接受率會歸零）→ 必須在第一次 `llama_decode()` 前設完。
- `llama_get_embeddings_layer_inp()` 內部自帶 `synchronize()`。
- `n_outputs_max` 預設等於 `n_batch` → 可以一次取得全 batch 的 logits，成本 `n_vocab × n_batch × 4 B`。因此 **argmax 在 shim 內算完**，不回傳 logits。
- `embeddings_layer_inp` 與 `cparams.embeddings` 無關：不必開 `embeddings` 也會配置輸出緩衝。

### 5. shim（`csrc/`）狀態

- 必須用 **C++** 編譯：`llama_get/set_embeddings_layer_inp` 是 C++ linkage（`nm -D` 可見 `_Z30llama_get_embeddings_layer_inpP13llama_contextj`），純 C TU 連不上。
- 做法：手動宣告兩個私有原型（不 include `src/llama-ext.h`，避免拉進 `llama-arch.h`），對外只導出純 C ABI。
- 已完成：`sp_create / sp_destroy / sp_set_layers / sp_tokenize / sp_decode / sp_reset / sp_row_stride / sp_tok_embd_rows`。
  `sp_decode()` 一次做完 decode、多層特徵抽取（F32→BF16 RNE）、逐列 argmax，並依 §5 佈局寫出 `[bf16 × n_layer_feat][int32 token][int32 argmax]`。
- `sp_feat_cli` 是 M1/G0 harness：印 JSON、可 `--dump-argmax` / `--dump-feat`、可帶 `--d2t` 直接算 coverage。

---

## 待決策（建議在 S2 前定案）

1. **draft 詞表**（會決定 `lm_head` 形狀，必須在訓練前定）
   - 先用 S2a 語料算 coverage 的三個變體：現有 32K / 頻率重挑 32K / 64K。
   - 判準：若 chat_think 語域的 coverage 明顯 < 0.33，直接換詞表，不要浪費一次訓練。
2. **特徵精度 bf16 vs F32**：v8 原始權重是 F32，bf16 只是為了容量（2.79M vs 1.4M token）。在 S1 用同一批資料各訓一次短程，比 p0 再定。
3. **chunk 內是否保留連續 KV**：保留較接近真實長上下文，但 9 層 full attention 會隨深度變慢；每 chunk reset 則前 512 token 的上下文被截斷。

## 下一步

S1：`spark_eagle3/llamacpp.py`（ctypes binding）+ `feats.py`（memfd、容量前置檢查）→ 32K token 端到端 → G1。
