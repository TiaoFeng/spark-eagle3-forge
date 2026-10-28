// sp_gen_cli.cpp — measure how much of the target's own output the 32k draft
// vocabulary can represent.
//
// Why this exists: the eagle3 head can only propose one of the 32k tokens that
// `d2t` maps into the target vocabulary; every other logit is -inf at runtime
// (src/models/eagle3.cpp:307). So for any workload
//
//     p0  <=  Pr[target picks a token that is in the draft vocabulary]
//
// and no amount of fine-tuning lifts a workload above that. Measuring the bound
// needs the *sampled* token (for temp > 0) and the argmax (for greedy), which is
// exactly what this harness records while generating text with the target model.
//
// Usage:
//   sp_gen_cli --model M.gguf --d2t d2t_target_ids.i32 \
//              --prompt-file p1.txt [--prompt-file p2.txt ...] \
//              [--n-gen 256] [--temp 0.7] [--top-p 0.9] [--top-k 0] [--seed 1] \
//              [--ctx 2048] [--ngl 99] [--out-dir DIR]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
int32_t     sp_n_vocab(const sp_handle *);
int32_t     sp_tokenize(sp_handle *, const char *, int32_t, int32_t *, int32_t, bool);
int32_t     sp_decode(sp_handle *, const int32_t *, int32_t, int32_t,
                      void *, int32_t *, float *, int32_t);
const float * sp_logits(sp_handle *, int32_t);
int32_t     sp_token_to_piece(sp_handle *, int32_t, char *, int32_t);
int32_t     sp_is_eog(sp_handle *, int32_t);
void        sp_reset(sp_handle *);
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

struct rng_t {
    uint32_t s;
    explicit rng_t(uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float uniform() { return (float) (next() >> 8) * (1.0f / 16777216.0f); }
};

// returns the chosen token id; also reports the argmax so the caller can track
// how often greedy and sampled decoding agree
static int32_t sample_token(const float * lg, int32_t n_vocab,
                            float temp, float top_p, int32_t top_k,
                            rng_t & rng, int32_t & argmax_out) {
    int32_t best = 0;
    float   bv   = lg[0];
    for (int32_t v = 1; v < n_vocab; ++v) {
        if (lg[v] > bv) { bv = lg[v]; best = v; }
    }
    argmax_out = best;
    if (temp <= 0.0f) {
        return best;
    }

    const float inv_t = 1.0f / temp;
    float mx = bv;
    std::vector<std::pair<float, int32_t>> cand;
    cand.reserve((size_t) n_vocab);
    double sum = 0.0;
    for (int32_t v = 0; v < n_vocab; ++v) {
        // skip values that cannot survive underflow; keeps the sort small
        if (lg[v] < mx - 40.0f * temp) { continue; }
        const float p = std::exp((lg[v] - mx) * inv_t);
        cand.emplace_back(p, v);
        sum += p;
    }
    std::sort(cand.begin(), cand.end(),
              [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                  return a.first > b.first;
              });

    size_t keep = cand.size();
    if (top_k > 0 && (size_t) top_k < keep) { keep = (size_t) top_k; }
    if (top_p > 0.0f && top_p < 1.0f) {
        double acc = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            acc += (double) cand[i].first;
            if (acc / sum >= (double) top_p) { keep = i + 1; break; }
        }
    }
    double acc = 0.0;
    for (size_t i = 0; i < keep; ++i) { acc += (double) cand[i].first; }
    const double u = (double) rng.uniform() * acc;
    double run = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        run += (double) cand[i].first;
        if (run >= u) { return cand[i].second; }
    }
    return cand[keep ? keep - 1 : 0].second;
}

int main(int argc, char ** argv) {
    const char * model = nullptr;
    const char * d2tf  = nullptr;
    const char * outdir = nullptr;
    std::vector<const char *> prompts;
    int32_t ngl    = 99;
    uint32_t nctx  = 2048;
    int32_t n_gen  = 256;
    float temp     = 0.7f;
    float top_p    = 0.9f;
    int32_t top_k  = 0;
    uint32_t seed  = 1234;
    bool add_special = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if      (a == "--model")       { model = next("--model"); }
        else if (a == "--d2t")         { d2tf = next("--d2t"); }
        else if (a == "--prompt-file") { prompts.push_back(next("--prompt-file")); }
        else if (a == "--out-dir")     { outdir = next("--out-dir"); }
        else if (a == "--n-gen")       { n_gen = std::atoi(next("--n-gen")); }
        else if (a == "--ctx")         { nctx = (uint32_t) std::atoi(next("--ctx")); }
        else if (a == "--ngl")         { ngl = std::atoi(next("--ngl")); }
        else if (a == "--temp")        { temp = (float) std::atof(next("--temp")); }
        else if (a == "--top-p")       { top_p = (float) std::atof(next("--top-p")); }
        else if (a == "--top-k")       { top_k = std::atoi(next("--top-k")); }
        else if (a == "--seed")        { seed = (uint32_t) std::strtoul(next("--seed"), nullptr, 10); }
        else if (a == "--add-special") { add_special = true; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (!model || !d2tf || prompts.empty()) {
        std::fprintf(stderr, "usage: sp_gen_cli --model M.gguf --d2t ids.i32 "
                             "--prompt-file p.txt [--prompt-file q.txt] ...\n");
        return 2;
    }

    std::string raw;
    if (!read_file(d2tf, raw) || raw.size() % 4 != 0) {
        std::fprintf(stderr, "cannot read d2t id file %s\n", d2tf);
        return 1;
    }

    sp_params p{};
    p.model_path   = model;
    p.n_gpu_layers = ngl;
    p.n_ctx        = nctx;
    p.n_batch      = nctx;
    p.n_ubatch     = nctx;

    sp_handle * h = sp_create(&p);
    if (!h || !sp_ok(h)) {
        std::fprintf(stderr, "sp_create failed: %s\n", h ? sp_last_error(h) : "null");
        return 1;
    }
    const int32_t n_vocab = sp_n_vocab(h);

    std::vector<uint8_t> in_set((size_t) n_vocab, 0);
    for (size_t i = 0; i < raw.size() / 4; ++i) {
        int32_t id;
        std::memcpy(&id, raw.data() + i * 4, 4);
        if (id >= 0 && id < n_vocab) { in_set[(size_t) id] = 1; }
    }

    std::printf("[\n");
    bool first_out = true;
    rng_t rng(seed);

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        std::string text;
        if (!read_file(prompts[pi], text)) {
            std::fprintf(stderr, "cannot read %s\n", prompts[pi]);
            continue;
        }
        sp_reset(h);

        int32_t need = sp_tokenize(h, text.data(), (int32_t) text.size(), nullptr, 0, add_special);
        if (need <= 0) { std::fprintf(stderr, "tokenize failed\n"); continue; }
        std::vector<int32_t> toks((size_t) need);
        sp_tokenize(h, text.data(), (int32_t) text.size(), toks.data(), need, add_special);

        const int32_t n_prompt = (int32_t) toks.size();
        if (n_prompt + n_gen + 2 > (int32_t) nctx) {
            std::fprintf(stderr, "prompt too long for ctx %u\n", nctx);
            continue;
        }

        // prefill everything but the last prompt token, then step one token at a
        // time so every generated step has its own logits
        if (n_prompt > 1) {
            if (sp_decode(h, toks.data(), n_prompt - 1, 0, nullptr, nullptr, nullptr, 0) != n_prompt - 1) {
                std::fprintf(stderr, "prefill failed: %s\n", sp_last_error(h));
                continue;
            }
        }

        int32_t cur   = toks[(size_t) n_prompt - 1];
        int32_t pos   = n_prompt - 1;
        int32_t n_step = 0, sampled_in = 0, argmax_in = 0, agree = 0;
        std::string out;
        char piece[256];

        auto t0 = std::chrono::steady_clock::now();
        for (int32_t s = 0; s < n_gen; ++s) {
            if (sp_decode(h, &cur, 1, pos, nullptr, nullptr, nullptr, 1) != 1) {
                std::fprintf(stderr, "decode failed: %s\n", sp_last_error(h));
                break;
            }
            const float * lg = sp_logits(h, -1);
            if (!lg) { std::fprintf(stderr, "no logits\n"); break; }

            int32_t argmax = -1;
            const int32_t tok = sample_token(lg, n_vocab, temp, top_p, top_k, rng, argmax);

            ++n_step;
            if (tok >= 0 && in_set[(size_t) tok]) { ++sampled_in; }
            if (argmax >= 0 && in_set[(size_t) argmax]) { ++argmax_in; }
            if (tok == argmax) { ++agree; }

            const int32_t nb = sp_token_to_piece(h, tok, piece, (int32_t) sizeof(piece));
            if (nb > 0) { out.append(piece, (size_t) nb); }

            ++pos;
            cur = tok;
            if (sp_is_eog(h, tok)) { break; }
        }
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (outdir) {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/gen_%zu.txt", outdir, pi);
            FILE * f = std::fopen(path, "wb");
            if (f) { std::fwrite(out.data(), 1, out.size(), f); std::fclose(f); }
        }

        if (!first_out) { std::printf(",\n"); }
        first_out = false;
        std::printf("  {\"prompt_file\": \"%s\", \"n_prompt\": %d, \"n_gen\": %d, "
                    "\"sampled_in_set\": %d, \"argmax_in_set\": %d, \"agree\": %d, "
                    "\"sampled_rate\": %.4f, \"argmax_rate\": %.4f, \"ms\": %.1f, "
                    "\"chars\": %zu}",
                    prompts[pi], n_prompt, n_step,
                    sampled_in, argmax_in, agree,
                    n_step ? (double) sampled_in / n_step : 0.0,
                    n_step ? (double) argmax_in / n_step : 0.0,
                    ms, out.size());
        std::fprintf(stderr, "[%s] %s\n\n", prompts[pi], out.c_str());
    }

    std::printf("\n]\n");
    sp_destroy(h);
    return 0;
}
