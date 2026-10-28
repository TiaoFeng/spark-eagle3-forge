// sp_feat_cli.cpp — M1 / G0 harness for the EAGLE-3 feature shim.
//
// G0 (cheap self-consistency gate, far cheaper than the p0 gate G1):
//   layer_inp[0] must equal the token embedding row of the same token.
//   If it does, the layer indexing, the token-major layout and the bf16 cast are
//   all confirmed at once. If it is off by a constant factor, Spark scales the
//   embeddings (also fine, just record the factor).
//
// Usage:
//   sp_feat_cli --model M.gguf --text FILE [--ctx 1024] [--ngl 99]
//               [--tokens N] [--layers 0,2,18,33] [--add-bos]
//               [--d2t d2t_target_ids.i32] [--dump-argmax out.i32]
//               [--dump-feat out.bin]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---- C ABI mirror of spark_feat.cpp ----
typedef struct sp_handle sp_handle;
typedef struct sp_params {
    const char * model_path;
    int32_t      n_gpu_layers;
    uint32_t     n_ctx;
    uint32_t     n_batch;
    uint32_t     n_ubatch;
    int32_t      verbose;
} sp_params;

extern "C" {
sp_handle * sp_create(const sp_params *);
void        sp_destroy(sp_handle *);
const char * sp_last_error(const sp_handle *);
int32_t     sp_ok(const sp_handle *);
int32_t     sp_n_embd(const sp_handle *);
int32_t     sp_n_vocab(const sp_handle *);
int32_t     sp_n_layer(const sp_handle *);
int32_t     sp_set_layers(sp_handle *, const int32_t *, int32_t);
int32_t     sp_n_layer_feat(const sp_handle *);
const int32_t * sp_layers(const sp_handle *);
int32_t     sp_row_stride(const sp_handle *);
int32_t     sp_tokenize(sp_handle *, const char *, int32_t, int32_t *, int32_t, bool);
int32_t     sp_tok_embd_rows(sp_handle *, const int32_t *, int32_t, float *);
int32_t     sp_decode(sp_handle *, const int32_t *, int32_t, int32_t,
                      void *, int32_t *, float *, int32_t);
void        sp_reset(sp_handle *);
}

static inline float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static bool read_file(const char * path, std::string & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { return false; }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < 0) { std::fclose(f); return false; }
    out.resize((size_t) n);
    const size_t got = n > 0 ? std::fread(&out[0], 1, (size_t) n, f) : 0;
    std::fclose(f);
    return got == (size_t) n;
}

static std::vector<int32_t> parse_layers(const char * s) {
    std::vector<int32_t> v;
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        const long x = std::strtol(p, &end, 10);
        if (end == p) { break; }
        v.push_back((int32_t) x);
        p = end;
        if (*p == ',') { ++p; }
    }
    return v;
}

int main(int argc, char ** argv) {
    const char * model = nullptr;
    const char * textf = nullptr;
    const char * d2tf  = nullptr;
    const char * dump_argmax = nullptr;
    const char * dump_feat  = nullptr;
    const char * dump_tokens = nullptr;
    int32_t ngl   = 99;
    uint32_t nctx = 1024;
    int32_t limit = 0;
    bool add_bos  = false;
    std::vector<int32_t> layers = {0, 2, 18, 33};

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--model")      { model = next("--model"); }
        else if (a == "--text")       { textf = next("--text"); }
        else if (a == "--ctx")        { nctx = (uint32_t) std::atoi(next("--ctx")); }
        else if (a == "--ngl")        { ngl = std::atoi(next("--ngl")); }
        else if (a == "--tokens")     { limit = std::atoi(next("--tokens")); }
        else if (a == "--layers")     { layers = parse_layers(next("--layers")); }
        else if (a == "--d2t")        { d2tf = next("--d2t"); }
        else if (a == "--dump-argmax"){ dump_argmax = next("--dump-argmax"); }
        else if (a == "--dump-feat")  { dump_feat  = next("--dump-feat"); }
        else if (a == "--dump-tokens"){ dump_tokens = next("--dump-tokens"); }
        else if (a == "--add-bos")    { add_bos = true; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (!model || !textf) {
        std::fprintf(stderr, "usage: sp_feat_cli --model M.gguf --text FILE [--ctx N] [--ngl N] "
                             "[--tokens N] [--layers 0,2,18,33] [--add-bos] [--d2t f.i32]\n");
        return 2;
    }

    std::string text;
    if (!read_file(textf, text)) {
        std::fprintf(stderr, "cannot read %s\n", textf);
        return 1;
    }

    sp_params p{};
    p.model_path   = model;
    p.n_gpu_layers = ngl;
    p.n_ctx        = nctx;
    p.n_batch      = nctx;
    p.n_ubatch     = nctx;
    p.verbose      = 0;

    auto t0 = std::chrono::steady_clock::now();
    sp_handle * h = sp_create(&p);
    auto t1 = std::chrono::steady_clock::now();
    if (!h || !sp_ok(h)) {
        std::fprintf(stderr, "sp_create failed: %s\n", h ? sp_last_error(h) : "null");
        if (h) { sp_destroy(h); }
        return 1;
    }
    const double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const int32_t n_embd  = sp_n_embd(h);
    const int32_t n_vocab = sp_n_vocab(h);

    // tokenize
    int32_t need = sp_tokenize(h, text.data(), (int32_t) text.size(), nullptr, 0, add_bos);
    if (need < 0) {
        std::fprintf(stderr, "tokenize failed: %s\n", sp_last_error(h));
        sp_destroy(h);
        return 1;
    }
    std::vector<int32_t> toks((size_t) need);
    sp_tokenize(h, text.data(), (int32_t) text.size(), toks.data(), need, add_bos);

    int32_t n = (int32_t) toks.size();
    int32_t truncated = 0;
    if (limit > 0 && n > limit) { n = limit; truncated = 1; }
    if (n > (int32_t) nctx)     { n = (int32_t) nctx; truncated = 1; }
    if (n <= 0) {
        std::fprintf(stderr, "no tokens\n");
        sp_destroy(h);
        return 1;
    }

    if (sp_set_layers(h, layers.data(), (int32_t) layers.size()) < 0) {
        std::fprintf(stderr, "sp_set_layers failed: %s\n", sp_last_error(h));
        sp_destroy(h);
        return 1;
    }

    const int32_t stride   = sp_row_stride(h);
    const int32_t n_feat   = sp_n_layer_feat(h);
    std::vector<uint8_t> feat((size_t) n * stride, 0);
    std::vector<int32_t> argmax((size_t) n, -1);
    std::vector<float>   argmax_v((size_t) n, 0.0f);

    auto t2 = std::chrono::steady_clock::now();
    const int32_t got = sp_decode(h, toks.data(), n, 0, feat.data(),
                                 argmax.data(), argmax_v.data(), 1);
    auto t3 = std::chrono::steady_clock::now();
    if (got != n) {
        std::fprintf(stderr, "sp_decode failed: %s\n", sp_last_error(h));
        sp_destroy(h);
        return 1;
    }
    const double decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ---- G0: layer 0 input must equal the token embedding row ----
    int g0_layer = -1;
    for (int32_t k = 0; k < n_feat; ++k) {
        if (sp_layers(h)[k] == 0) { g0_layer = k; break; }
    }
    double g0_max = -1.0, g0_mean = -1.0, g0_scale = -1.0;
    if (g0_layer >= 0) {
        std::vector<float> emb((size_t) n * n_embd, 0.0f);
        if (sp_tok_embd_rows(h, toks.data(), n, emb.data()) != n) {
            std::fprintf(stderr, "tok_embd_rows failed\n");
        } else {
            double mx = 0.0, sum = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
            for (int32_t i = 0; i < n; ++i) {
                for (int32_t e = 0; e < n_embd; ++e) {
                    const uint8_t * row = feat.data() + (size_t) i * stride;
                    const uint16_t b = (uint16_t) (row[(size_t) e * 2] |
                                                  (row[(size_t) e * 2 + 1] << 8));
                    const double a = (double) bf16_to_f32(b);
                    const double r = (double) emb[(size_t) i * n_embd + e];
                    const double d = std::fabs(a - r);
                    if (d > mx) { mx = d; }
                    sum += d;
                    dot += a * r;
                    na  += a * a;
                    nb  += r * r;
                }
            }
            const double cnt = (double) n * n_embd;
            g0_max   = mx;
            g0_mean  = sum / cnt;
            g0_scale = (na > 0 && nb > 0) ? dot / nb : -1.0; // regress feature ~ scale * emb
        }
    }

    // ---- sanity: the three target layers must not be identical ----
    double sep_min = -1.0;
    if (n_feat >= 2) {
        double worst = 1e30;
        for (int32_t a = 0; a < n_feat; ++a) {
            for (int32_t b = a + 1; b < n_feat; ++b) {
                double mx = 0.0;
                for (int32_t i = 0; i < n; ++i) {
                    const uint8_t * ra = feat.data() + (size_t) i * stride + (size_t) a * n_embd * 2;
                    const uint8_t * rb = feat.data() + (size_t) i * stride + (size_t) b * n_embd * 2;
                    for (int32_t e = 0; e < n_embd; ++e) {
                        const uint16_t ba = (uint16_t) (ra[(size_t) e * 2] | (ra[(size_t) e * 2 + 1] << 8));
                        const uint16_t bb = (uint16_t) (rb[(size_t) e * 2] | (rb[(size_t) e * 2 + 1] << 8));
                        const double d = std::fabs((double) bf16_to_f32(ba) - (double) bf16_to_f32(bb));
                        if (d > mx) { mx = d; }
                    }
                }
                if (mx < worst) { worst = mx; }
            }
        }
        sep_min = worst;
    }

    // ---- label coverage vs the draft vocabulary (32k d2t subset) ----
    double coverage = -1.0;
    if (d2tf) {
        std::string raw;
        if (read_file(d2tf, raw) && raw.size() % 4 == 0) {
            const int32_t nd = (int32_t) (raw.size() / 4);
            std::vector<uint8_t> in((size_t) n_vocab, 0);
            int32_t bad = 0;
            for (int32_t i = 0; i < nd; ++i) {
                int32_t id;
                std::memcpy(&id, raw.data() + (size_t) i * 4, 4);
                if (id >= 0 && id < n_vocab) { in[(size_t) id] = 1; } else { ++bad; }
            }
            int32_t hit = 0;
            for (int32_t i = 0; i < n; ++i) {
                if (argmax[i] >= 0 && in[(size_t) argmax[i]]) { ++hit; }
            }
            coverage = (double) hit / (double) n;
            if (bad) { std::fprintf(stderr, "warn: %d d2t ids out of range\n", bad); }
        } else {
            std::fprintf(stderr, "cannot read d2t file %s (expect raw int32)\n", d2tf);
        }
    }

    if (dump_argmax) {
        FILE * f = std::fopen(dump_argmax, "wb");
        if (f) { std::fwrite(argmax.data(), 4, (size_t) n, f); std::fclose(f); }
    }
    if (dump_feat) {
        FILE * f = std::fopen(dump_feat, "wb");
        if (f) { std::fwrite(feat.data(), 1, feat.size(), f); std::fclose(f); }
    }
    if (dump_tokens) {
        FILE * f = std::fopen(dump_tokens, "wb");
        if (f) { std::fwrite(toks.data(), 4, toks.size(), f); std::fclose(f); }
    }

    std::printf("{\n");
    std::printf("  \"n_embd\": %d, \"n_vocab\": %d, \"n_layer\": %d,\n",
                n_embd, n_vocab, sp_n_layer(h));
    std::printf("  \"layers\": [");
    for (int32_t k = 0; k < n_feat; ++k) {
        std::printf("%s%d", k ? ", " : "", sp_layers(h)[k]);
    }
    std::printf("],\n");
    std::printf("  \"row_stride\": %d,\n", stride);
    std::printf("  \"n_text_tokens\": %d, \"n_decoded\": %d, \"truncated\": %d,\n",
                (int32_t) toks.size(), n, truncated);
    std::printf("  \"load_ms\": %.1f, \"decode_ms\": %.1f, \"tok_per_s\": %.2f,\n",
                load_ms, decode_ms, decode_ms > 0 ? 1000.0 * n / decode_ms : 0.0);
    std::printf("  \"g0_layer0_index\": %d, \"g0_max_abs_diff\": %.6g, "
                "\"g0_mean_abs_diff\": %.6g, \"g0_scale_vs_embd\": %.6f,\n",
                g0_layer, g0_max, g0_mean, g0_scale);
    std::printf("  \"layer_pair_min_max_abs_diff\": %.6g,\n", sep_min);
    std::printf("  \"label_coverage_32k\": %.4f,\n", coverage);
    std::printf("  \"first_argmax\": [");
    for (int32_t i = 0; i < n && i < 8; ++i) {
        std::printf("%s%d", i ? ", " : "", argmax[i]);
    }
    std::printf("]\n}\n");

    sp_destroy(h);
    return 0;
}
