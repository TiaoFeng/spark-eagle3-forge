// spark_feat.cpp — C ABI shim over libllama for EAGLE-3 feature extraction.
//
// Why a shim at all: the two functions we need are declared in src/llama-ext.h,
// which is a *private* header (it pulls in llama-arch.h and C++ containers) and,
// unlike include/llama.h, its declarations are NOT inside an `extern "C"` block.
// They therefore have C++ linkage and mangled symbols:
//
//   $ nm -D --defined-only build/bin/libllama.so | grep layer_inp
//   000000000022ab90 T _Z30llama_get_embeddings_layer_inpP13llama_contextj
//   000000000022ab10 T _Z30llama_set_embeddings_layer_inpP13llama_contextjb
//
// So this file is compiled as C++, declares the two prototypes by hand (cheap and
// stable: they are the mangled names above), and exports a plain C ABI that Python
// can load with ctypes without guessing any struct layout.
//
// Extraction-isolation rule: the shim never touches disk or tmpfs. It writes into
// caller-provided memory; the memfd/shard bookkeeping lives in Python.

#include <llama.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// private llama.cpp API (src/llama-ext.h) — C++ linkage, deliberately re-declared
// ---------------------------------------------------------------------------
void    llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);
float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// returns n_tokens * n_embd floats for the given token rows; 0 on error
uint32_t llama_model_get_tok_embd_rows(
        const struct llama_model * model,
        const llama_token * tokens,
        uint32_t n_tokens,
        float * out);

// ---------------------------------------------------------------------------
// public C ABI
// ---------------------------------------------------------------------------
#define SP_API extern "C" __attribute__((visibility("default")))

typedef struct sp_handle sp_handle;

typedef struct sp_params {
    const char * model_path;   // required
    int32_t      n_gpu_layers; // 99 = all
    uint32_t     n_ctx;        // >= n_batch
    uint32_t     n_batch;
    uint32_t     n_ubatch;     // 0 = n_batch
    int32_t      verbose;
} sp_params;

struct sp_handle {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;

    std::vector<int32_t> layers;
    int32_t n_embd  = 0;
    int32_t n_vocab = 0;
    int32_t n_layer = 0;
    int32_t n_ctx   = 0;
    int32_t n_batch = 0;

    std::vector<float>   embd_scratch; // G0 helper
    std::string          err;
};

// bytes per emitted token row: n_layer_feat * n_embd * 2 (bf16) + int32 token + int32 argmax
static inline int32_t row_stride_for(const sp_handle * h) {
    return (int32_t) h->layers.size() * h->n_embd * 2 + 8;
}

// round-to-nearest-even F32 -> BF16
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb; // NaN -> NaN (may canonicalise payload, harmless here)
    return (uint16_t) (u >> 16);
}

SP_API const char * sp_version(void) { return "spark_feat 0.1.0"; }

SP_API sp_handle * sp_create(const sp_params * p) {
    if (!p || !p->model_path) {
        return nullptr;
    }

    static bool backend_ready = false;
    if (!backend_ready) {
        llama_backend_init();
        backend_ready = true;
    }

    auto * h = new sp_handle();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = p->n_gpu_layers;

    h->model = llama_model_load_from_file(p->model_path, mparams);
    if (!h->model) {
        h->err = "llama_model_load_from_file failed: ";
        h->err += p->model_path;
        return h; // handle kept so the caller can read sp_last_error()
    }

    llama_context_params cparams = llama_context_default_params();
    const uint32_t n_batch = p->n_batch ? p->n_batch : 512;
    cparams.n_ctx     = p->n_ctx ? p->n_ctx : n_batch;
    cparams.n_batch   = n_batch;
    cparams.n_ubatch  = p->n_ubatch ? p->n_ubatch : n_batch;
    cparams.n_seq_max = 1;

    h->ctx = llama_init_from_model(h->model, cparams);
    if (!h->ctx) {
        h->err = "llama_init_from_model failed";
        return h;
    }

    h->n_embd  = llama_model_n_embd(h->model);
    h->n_layer = llama_model_n_layer(h->model);
    h->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(h->model));
    h->n_ctx   = (int32_t) cparams.n_ctx;
    h->n_batch = (int32_t) cparams.n_batch;

    return h;
}

SP_API void sp_destroy(sp_handle * h) {
    if (!h) {
        return;
    }
    if (h->ctx)   { llama_free(h->ctx); }
    if (h->model) { llama_model_free(h->model); }
    delete h;
}

SP_API const char * sp_last_error(const sp_handle * h) {
    return h ? h->err.c_str() : "null handle";
}

SP_API int32_t sp_ok(const sp_handle * h) {
    return (h && h->ctx && h->model && h->err.empty()) ? 1 : 0;
}

SP_API int32_t sp_n_embd (const sp_handle * h) { return h ? h->n_embd  : 0; }
SP_API int32_t sp_n_vocab(const sp_handle * h) { return h ? h->n_vocab : 0; }
SP_API int32_t sp_n_layer(const sp_handle * h) { return h ? h->n_layer : 0; }
SP_API int32_t sp_n_ctx  (const sp_handle * h) { return h ? h->n_ctx   : 0; }
SP_API int32_t sp_n_batch(const sp_handle * h) { return h ? h->n_batch : 0; }

SP_API int32_t sp_n_layer_feat(const sp_handle * h) {
    return h ? (int32_t) h->layers.size() : 0;
}

SP_API const int32_t * sp_layers(const sp_handle * h) {
    return (h && !h->layers.empty()) ? h->layers.data() : nullptr;
}

SP_API int32_t sp_row_stride(const sp_handle * h) {
    return h ? row_stride_for(h) : 0;
}

// Must be called before the first sp_decode(): set_embeddings_layer_inp() flips
// `sched_need_reserve`, and the extracted buffer is sized from these flags when
// the output buffer is laid out.
SP_API int32_t sp_set_layers(sp_handle * h, const int32_t * layers, int32_t n) {
    if (!h || !h->ctx) {
        return -1;
    }
    if (!layers || n <= 0) {
        h->err = "sp_set_layers: empty layer list";
        return -1;
    }
    for (int32_t i = 0; i < n; ++i) {
        if (layers[i] < 0 || layers[i] > h->n_layer) {
            h->err = "sp_set_layers: layer id out of range (0..n_layer)";
            return -1;
        }
    }
    h->layers.assign(layers, layers + n);
    for (int32_t lid : h->layers) {
        llama_set_embeddings_layer_inp(h->ctx, (uint32_t) lid, true);
    }
    return n;
}

SP_API void sp_reset(sp_handle * h) {
    if (!h || !h->ctx) {
        return;
    }
    llama_memory_clear(llama_get_memory(h->ctx), true);
}

SP_API void sp_sync(sp_handle * h) {
    if (h && h->ctx) {
        llama_synchronize(h->ctx);
    }
}

SP_API int32_t sp_tokenize(
        sp_handle * h, const char * text, int32_t text_len,
        int32_t * out, int32_t cap, bool add_special) {
    if (!h || !h->model) {
        return -1;
    }
    // first call with a generous buffer is not available; grow on demand
    std::vector<llama_token> buf((size_t) (cap > 0 ? cap : 4096));
    int32_t n = llama_tokenize(
            llama_model_get_vocab(h->model), text, text_len,
            buf.data(), (int32_t) buf.size(), add_special, /*parse_special*/ true);
    if (n < 0) { // buffer too small: n is the required size (negative)
        buf.resize((size_t) (-n));
        n = llama_tokenize(
                llama_model_get_vocab(h->model), text, text_len,
                buf.data(), (int32_t) buf.size(), add_special, true);
    }
    if (n < 0) {
        h->err = "llama_tokenize failed";
        return -1;
    }
    const int32_t n_copy = (out && cap > 0) ? (n < cap ? n : cap) : 0;
    if (n_copy > 0) {
        std::memcpy(out, buf.data(), (size_t) n_copy * sizeof(int32_t));
    }
    return n; // required size regardless of cap
}

// G0 helper: raw token embedding rows, F32, n * n_embd floats.
SP_API int32_t sp_tok_embd_rows(
        sp_handle * h, const int32_t * toks, int32_t n, float * out) {
    if (!h || !h->model || !out) {
        return -1;
    }
    std::vector<llama_token> lt((size_t) n);
    for (int32_t i = 0; i < n; ++i) {
        lt[(size_t) i] = (llama_token) toks[i];
    }
    const uint32_t nw = llama_model_get_tok_embd_rows(
            h->model, lt.data(), (uint32_t) n, out);
    return (int32_t) (nw / (uint32_t) h->n_embd);
}

// Decode `n` tokens at positions start_pos..start_pos+n-1 and, if feat_out is not
// null, emit one row per token:
//   [bf16  n_layer_feat * n_embd][int32 token][int32 argmax]
// Rows are laid out contiguously with sp_row_stride() spacing.
// argmax is written to argmax_out[i] (or -1 when want_logits == 0).
// Returns the number of rows written, or -1 on error.
SP_API int32_t sp_decode(
        sp_handle * h,
        const int32_t * toks, int32_t n, int32_t start_pos,
        void * feat_out, int32_t * argmax_out, float * argmax_p_out,
        int32_t want_logits) {
    if (!h || !h->ctx) {
        return -1;
    }
    if (!toks || n <= 0) {
        h->err = "sp_decode: empty batch";
        return -1;
    }
    if (n > h->n_batch) {
        h->err = "sp_decode: batch larger than n_batch";
        return -1;
    }
    if (start_pos + n > h->n_ctx) {
        h->err = "sp_decode: batch exceeds n_ctx";
        return -1;
    }
    if (feat_out && h->layers.empty()) {
        h->err = "sp_decode: sp_set_layers() not called";
        return -1;
    }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]    = (llama_token) toks[i];
        batch.pos[i]      = start_pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (want_logits && (feat_out || argmax_out)) ? 1 : 0;
    }
    batch.n_tokens = n;

    const int32_t rc = llama_decode(h->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        h->err = "llama_decode failed rc=" + std::to_string(rc);
        return -1;
    }

    const int32_t n_feat = (int32_t) h->layers.size();

    if (feat_out) {
        const int32_t stride = row_stride_for(h);
        auto * base = (uint8_t *) feat_out;

        // one get_embeddings_layer_inp() per layer, then scatter every token row
        for (int32_t k = 0; k < n_feat; ++k) {
            const float * src = llama_get_embeddings_layer_inp(
                    h->ctx, (uint32_t) h->layers[(size_t) k]);
            if (!src) {
                h->err = "layer input " + std::to_string(h->layers[(size_t) k]) +
                         " not extracted (patch missing?)";
                return -1;
            }
            const size_t dst_layer_off = (size_t) k * h->n_embd * 2;
            for (int32_t i = 0; i < n; ++i) {
                uint8_t * row = base + (size_t) i * stride + dst_layer_off;
                const float * s = src + (size_t) i * h->n_embd;
                for (int32_t e = 0; e < h->n_embd; ++e) {
                    const uint16_t b = f32_to_bf16(s[e]);
                    std::memcpy(row + (size_t) e * 2, &b, 2);
                }
            }
        }
        const size_t tail = (size_t) n_feat * h->n_embd * 2;
        for (int32_t i = 0; i < n; ++i) {
            uint8_t * row = base + (size_t) i * stride + tail;
            const int32_t tok = toks[i];
            const int32_t am  = (argmax_out && want_logits) ? argmax_out[i] : -1;
            std::memcpy(row + 0, &tok, 4);
            std::memcpy(row + 4, &am, 4);
        }
    }

    if (want_logits) {
        llama_synchronize(h->ctx);
        for (int32_t i = 0; i < n; ++i) {
            const float * lg = llama_get_logits_ith(h->ctx, i);
            if (!lg) {
                h->err = "llama_get_logits_ith returned null";
                return -1;
            }
            int32_t best = 0;
            float   bv   = lg[0];
            for (int32_t v = 1; v < h->n_vocab; ++v) {
                if (lg[v] > bv) { bv = lg[v]; best = v; }
            }
            if (argmax_out)  { argmax_out[i] = best; }
            if (argmax_p_out) { argmax_p_out[i] = bv; }
        }
        // feature rows may have been written before argmax was known: patch the tail
        if (feat_out && argmax_out) {
            const int32_t stride = row_stride_for(h);
            auto * base = (uint8_t *) feat_out;
            const size_t tail = (size_t) n_feat * h->n_embd * 2;
            for (int32_t i = 0; i < n; ++i) {
                std::memcpy(base + (size_t) i * stride + tail + 4, &argmax_out[i], 4);
            }
        }
    }

    return n;
}
