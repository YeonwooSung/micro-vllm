#include "official_metal.hpp"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(MVLLM_OFFICIAL_METAL_HOST)
extern "C" {
#include "backend_metal.h"
#include "h3_gpu.h"
}
#endif

namespace mvllm {
namespace official_metal {
namespace {

bool g_coli = false;
bool g_h3 = false;
char g_st[256] = "off";
#if defined(MVLLM_OFFICIAL_METAL_HOST)
h3_gpu *g_h3g = nullptr;

bool coli_gemm(float *y, const float *x, const float *w, int S, int I, int O) {
    if (coli_metal_gemm(y, x, w, nullptr, 0, S, I, O, 0))
        return true;
    std::vector<float> ones(static_cast<size_t>(O), 1.f);
    ColiMetalTensor *t = nullptr;
    const int ok = coli_metal_matmul(&t, y, x, w, ones.data(), 0, S, I, O, 0);
    if (t)
        coli_metal_tensor_free(t);
    return ok != 0;
}

struct H3Bag {
    std::vector<h3_gpu_tensor *> ts;
    h3_gpu_tensor *add(h3_gpu_tensor *t) {
        if (t)
            ts.push_back(t);
        return t;
    }
    ~H3Bag() {
        for (h3_gpu_tensor *t : ts)
            h3_gpu_tensor_free(t);
    }
};

struct H3Cmd {
    bool open;
    explicit H3Cmd() : open(g_h3g && h3_gpu_begin(g_h3g) != 0) {}
    ~H3Cmd() {
        if (open && g_h3g)
            (void)h3_gpu_submit(g_h3g);
    }
    bool submit() {
        if (!open || !g_h3g)
            return false;
        open = false;
        return h3_gpu_submit(g_h3g) != 0;
    }
};

void apply_rope_host(float *q, float *k, const float *cos, const float *sin, int T, int I,
                     int heads, int hd) {
    const int half = 48;
    for (int t = 0; t < T; ++t) {
        const float *c = cos + static_cast<size_t>(t) * static_cast<size_t>(half);
        const float *s = sin + static_cast<size_t>(t) * static_cast<size_t>(half);
        for (int h = 0; h < heads; ++h) {
            float *qq = q + (static_cast<size_t>(t) * static_cast<size_t>(I) +
                             static_cast<size_t>(h) * static_cast<size_t>(hd));
            float *kk = k + (static_cast<size_t>(t) * static_cast<size_t>(I) +
                             static_cast<size_t>(h) * static_cast<size_t>(hd));
            for (int d = 0; d < half; ++d) {
                const float qa = qq[d], qb = qq[d + half];
                qq[d] = qa * c[d] - qb * s[d];
                qq[d + half] = qa * s[d] + qb * c[d];
                const float ka = kk[d], kb = kk[d + half];
                kk[d] = ka * c[d] - kb * s[d];
                kk[d + half] = ka * s[d] + kb * c[d];
            }
        }
    }
}

bool split_qkv(h3_gpu *gpu, h3_gpu_tensor *q, h3_gpu_tensor *k, h3_gpu_tensor *v,
               const h3_gpu_tensor *qkv, int T, int I) {
    const size_t Iu = static_cast<size_t>(I);
    for (int t = 0; t < T; ++t) {
        const size_t dst = static_cast<size_t>(t) * Iu;
        const size_t src = static_cast<size_t>(t) * 3u * Iu;
        if (!h3_gpu_copy_f32(gpu, q, dst, qkv, src, Iu) ||
            !h3_gpu_copy_f32(gpu, k, dst, qkv, src + Iu, Iu) ||
            !h3_gpu_copy_f32(gpu, v, dst, qkv, src + 2u * Iu, Iu))
            return false;
    }
    return true;
}
#endif

} // namespace

bool init() {
#if !defined(MVLLM_OFFICIAL_METAL_HOST)
    std::snprintf(g_st, sizeof(g_st), "off");
    return true;
#else
    g_coli = coli_metal_init() != 0;
    char err[256] = {};
#if defined(MVLLM_VENDOR_METAL_DIR)
    const char *sp = MVLLM_VENDOR_METAL_DIR "/h3_shaders.metal";
#else
    const char *sp = "src/gpu/vendor/h3_shaders.metal";
#endif
    g_h3g = h3_gpu_create(sp, err, sizeof(err));
    g_h3 = g_h3g != nullptr;
    if (g_coli && g_h3)
        std::snprintf(g_st, sizeof(g_st), "coli+h3");
    else if (g_coli)
        std::snprintf(g_st, sizeof(g_st), "coli");
    else if (g_h3)
        std::snprintf(g_st, sizeof(g_st), "h3");
    else
        std::snprintf(g_st, sizeof(g_st), "err: %s", err[0] ? err : "init");
    return true;
#endif
}

void shutdown() {
#if defined(MVLLM_OFFICIAL_METAL_HOST)
    if (g_h3g) {
        h3_gpu_free(g_h3g);
        g_h3g = nullptr;
    }
    coli_metal_shutdown();
#endif
    g_coli = false;
    g_h3 = false;
    std::snprintf(g_st, sizeof(g_st), "off");
}

bool coli_available() { return g_coli; }

bool h3_available() { return g_h3; }

const char *status() { return g_st; }

bool rmsnorm(float *y, const float *x, const float *w, int nrows, int D, float eps) {
#if !defined(MVLLM_OFFICIAL_METAL_HOST)
    (void)y;
    (void)x;
    (void)w;
    (void)nrows;
    (void)D;
    (void)eps;
    return false;
#else
    if (!g_coli || !y || !x || !w || nrows < 1 || D < 1)
        return false;
    if (y != x)
        std::memcpy(y, x, static_cast<size_t>(nrows) * static_cast<size_t>(D) * sizeof(float));
    return coli_metal_rmsnorm(y, w, nrows, D, eps) != 0;
#endif
}

bool add(float *y, const float *a, size_t n) {
#if !defined(MVLLM_OFFICIAL_METAL_HOST)
    (void)y;
    (void)a;
    (void)n;
    return false;
#else
    if (!g_coli || !y || !a)
        return false;
    return coli_metal_add(y, a, n) != 0;
#endif
}

bool silu_mul(float *g, const float *u, size_t n) {
#if !defined(MVLLM_OFFICIAL_METAL_HOST)
    (void)g;
    (void)u;
    (void)n;
    return false;
#else
    if (!g_coli || !g || !u)
        return false;
    return coli_metal_silu_mul(g, u, n) != 0;
#endif
}

bool gemm_f32(float *y, const float *x, const float *w, int S, int I, int O) {
#if !defined(MVLLM_OFFICIAL_METAL_HOST)
    (void)y;
    (void)x;
    (void)w;
    (void)S;
    (void)I;
    (void)O;
    return false;
#else
    if (!g_coli || !y || !x || !w || S < 1 || I < 1 || O < 1)
        return false;
    return coli_gemm(y, x, w, S, I, O);
#endif
}

bool layer_residual(float *x, const float *attn, const float *post_ln, float *nrm, int D,
                    float eps) {
#if !defined(MVLLM_OFFICIAL_METAL_HOST)
    (void)x;
    (void)attn;
    (void)post_ln;
    (void)nrm;
    (void)D;
    (void)eps;
    return false;
#else
    if (!g_coli || !x || !attn || !post_ln || !nrm || D < 1)
        return false;
    std::vector<float> xs(x, x + D);
    if (coli_metal_add(xs.data(), attn, static_cast<size_t>(D)) == 0)
        return false;
    std::vector<float> ns = xs;
    if (coli_metal_rmsnorm(ns.data(), post_ln, 1, D, eps) == 0)
        return false;
    std::memcpy(x, xs.data(), static_cast<size_t>(D) * sizeof(float));
    std::memcpy(nrm, ns.data(), static_cast<size_t>(D) * sizeof(float));
    return true;
#endif
}

bool moe_block_f32(int nb, int D, int Iinter, const float *const *g, const float *const *u,
                   const float *const *d, const float *xg, const int *xoff, const int *nr,
                   const int *rows, const float *rw, float *out, int S) {
#if !defined(MVLLM_OFFICIAL_METAL_HOST)
    (void)nb;
    (void)D;
    (void)Iinter;
    (void)g;
    (void)u;
    (void)d;
    (void)xg;
    (void)xoff;
    (void)nr;
    (void)rows;
    (void)rw;
    (void)out;
    (void)S;
    return false;
#else
    if (!g_coli || nb < 1 || D < 1 || !out || !xg || !xoff || !nr)
        return false;
    if (coli_metal_moe_block(nb, D, Iinter, 0, 0, reinterpret_cast<const void *const *>(g),
                             reinterpret_cast<const void *const *>(u),
                             reinterpret_cast<const void *const *>(d), nullptr, nullptr, nullptr,
                             xg, xoff, nr, rows, rw, out, S))
        return true;
    if (Iinter < 1 || !g || !u || !d)
        return false;
    int total = 0;
    for (int e = 0; e < nb; ++e) {
        if (nr[e] < 0)
            return false;
        total += nr[e];
    }
    std::vector<float> packed(static_cast<size_t>(total) * static_cast<size_t>(D), 0.f);
    int base = 0;
    for (int e = 0; e < nb; ++e) {
        const int nre = nr[e];
        if (nre <= 0)
            continue;
        if (!g[e] || !u[e] || !d[e])
            return false;
        const float *xe = xg + static_cast<size_t>(xoff[e]) * static_cast<size_t>(D);
        std::vector<float> gate(static_cast<size_t>(nre) * static_cast<size_t>(Iinter), 0.f);
        std::vector<float> up(static_cast<size_t>(nre) * static_cast<size_t>(Iinter), 0.f);
        if (!coli_gemm(gate.data(), xe, g[e], nre, D, Iinter) ||
            !coli_gemm(up.data(), xe, u[e], nre, D, Iinter))
            return false;
        if (coli_metal_silu_mul(gate.data(), up.data(),
                                static_cast<size_t>(nre) * static_cast<size_t>(Iinter)) == 0)
            return false;
        if (!coli_gemm(packed.data() + static_cast<size_t>(base) * static_cast<size_t>(D),
                       gate.data(), d[e], nre, Iinter, D))
            return false;
        base += nre;
    }
    if (!rows || !rw)
        return true;
    base = 0;
    for (int e = 0; e < nb; ++e) {
        const int nre = nr[e];
        for (int r = 0; r < nre; ++r) {
            const int dest = rows[base + r];
            if (dest < 0 || (S > 0 && dest >= S))
                continue;
            const float w = rw[base + r];
            const float *hr = packed.data() + static_cast<size_t>(base + r) * static_cast<size_t>(D);
            float *orow = out + static_cast<size_t>(dest) * static_cast<size_t>(D);
            for (int i = 0; i < D; ++i)
                orow[i] += w * hr[i];
        }
        base += nre;
    }
    return true;
#endif
}

bool dit_residual(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes, int64_t fc1_bytes,
                  int64_t fc2_bytes, int hidden, int inner, int ffn, int head_dim, float *x,
                  int tokens, float eps, const float *adaln_mod, const float *q_norm,
                  const float *k_norm, const float *rope_cos, const float *rope_sin,
                  const uint32_t *row_map, int adaln_groups, const float *norm1,
                  const float *norm2) {
#if !defined(MVLLM_OFFICIAL_METAL_HOST)
    (void)blob;
    (void)qkv_bytes;
    (void)out_bytes;
    (void)fc1_bytes;
    (void)fc2_bytes;
    (void)hidden;
    (void)inner;
    (void)ffn;
    (void)head_dim;
    (void)x;
    (void)tokens;
    (void)eps;
    (void)adaln_mod;
    (void)q_norm;
    (void)k_norm;
    (void)rope_cos;
    (void)rope_sin;
    (void)row_map;
    (void)adaln_groups;
    (void)norm1;
    (void)norm2;
    return false;
#else
    if (!g_h3 || !g_h3g || !blob || !x || hidden <= 0 || inner <= 0 || ffn <= 0 || tokens <= 0)
        return false;
    if (inner > INT_MAX / 3 || ffn > INT_MAX / 2)
        return false;
    const int T = tokens;
    const int H = hidden;
    const int I = inner;
    int hd = head_dim > 0 ? head_dim : I;
    int heads = I / hd;
    if (heads < 1) {
        heads = 1;
        hd = I;
    }
    if (heads * hd != I)
        return false;
    const int64_t qkv_n = qkv_bytes / 2;
    const int64_t out_n = out_bytes / 2;
    const int64_t fc1_n = fc1_bytes / 2;
    const int64_t fc2_n = fc2_bytes / 2;
    const int64_t need_qkv = static_cast<int64_t>(3) * I * H;
    const int64_t need_out = static_cast<int64_t>(H) * I;
    const int64_t need_fc1 = static_cast<int64_t>(2) * ffn * H;
    const int64_t need_fc2 = static_cast<int64_t>(H) * ffn;
    if (qkv_n < need_qkv || out_n < need_out || fc1_n < need_fc1 || fc2_n < need_fc2)
        return false;
    const int groups = adaln_groups > 0 ? adaln_groups : 1;
    const bool do_rope = rope_cos && rope_sin && hd >= 96;
    const uint32_t rope_half = do_rope ? 48u : 0u;
    const bool use_qkv_rope = q_norm && k_norm;

    H3Bag bag;
    auto *tx = bag.add(h3_gpu_tensor_from_f32(g_h3g, x, static_cast<size_t>(T) * H));
    auto *xn = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * H));
    std::vector<float> one_host(static_cast<size_t>(H > hd ? H : hd), 1.f);
    auto *ones = bag.add(h3_gpu_tensor_from_f32(g_h3g, one_host.data(), one_host.size()));
    h3_gpu_tensor *n1t = ones;
    h3_gpu_tensor *n2t = ones;
    if (norm1)
        n1t = bag.add(h3_gpu_tensor_from_f32(g_h3g, norm1, static_cast<size_t>(H)));
    if (norm2)
        n2t = bag.add(h3_gpu_tensor_from_f32(g_h3g, norm2, static_cast<size_t>(H)));
    std::vector<uint32_t> map(static_cast<size_t>(T), 0u);
    if (row_map) {
        for (int t = 0; t < T; ++t) {
            const int g = static_cast<int>(row_map[t]);
            map[static_cast<size_t>(t)] = (g >= 0 && g < groups) ? static_cast<uint32_t>(g) : 0u;
        }
    }
    auto *rm = bag.add(h3_gpu_tensor_from_u32(g_h3g, map.data(), static_cast<size_t>(T)));
    auto *qkv_b = bag.add(h3_gpu_tensor_from_bf16(
        g_h3g, reinterpret_cast<const uint16_t *>(blob), static_cast<size_t>(need_qkv)));
    auto *out_b = bag.add(h3_gpu_tensor_from_bf16(
        g_h3g, reinterpret_cast<const uint16_t *>(blob + qkv_bytes), static_cast<size_t>(need_out)));
    auto *fc1_b = bag.add(
        h3_gpu_tensor_from_bf16(g_h3g, reinterpret_cast<const uint16_t *>(blob + qkv_bytes + out_bytes),
                                static_cast<size_t>(need_fc1)));
    auto *fc2_b = bag.add(h3_gpu_tensor_from_bf16(
        g_h3g, reinterpret_cast<const uint16_t *>(blob + qkv_bytes + out_bytes + fc1_bytes),
        static_cast<size_t>(need_fc2)));
    auto *wqkv = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(need_qkv)));
    auto *wout = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(need_out)));
    auto *wfc1 = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(need_fc1)));
    auto *wfc2 = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(need_fc2)));
    auto *qkv = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * 3u * I));
    auto *q = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * I));
    auto *k = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * I));
    auto *v = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * I));
    auto *ctx = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * I));
    auto *attn = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * H));
    auto *h1 = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * 2u * ffn));
    auto *gated = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * ffn));
    auto *down = bag.add(h3_gpu_tensor_new_f32(g_h3g, static_cast<size_t>(T) * H));
    h3_gpu_tensor *mod = nullptr;
    if (adaln_mod)
        mod = bag.add(h3_gpu_tensor_from_f32(g_h3g, adaln_mod,
                                             static_cast<size_t>(groups) * 6u * static_cast<size_t>(H)));
    h3_gpu_tensor *qn = nullptr;
    h3_gpu_tensor *kn = nullptr;
    if (q_norm)
        qn = bag.add(h3_gpu_tensor_from_f32(g_h3g, q_norm, static_cast<size_t>(hd)));
    if (k_norm)
        kn = bag.add(h3_gpu_tensor_from_f32(g_h3g, k_norm, static_cast<size_t>(hd)));
    h3_gpu_tensor *rc = nullptr;
    h3_gpu_tensor *rs = nullptr;
    if (do_rope) {
        rc = bag.add(h3_gpu_tensor_from_f32(g_h3g, rope_cos, static_cast<size_t>(T) * 48u));
        rs = bag.add(h3_gpu_tensor_from_f32(g_h3g, rope_sin, static_cast<size_t>(T) * 48u));
    } else {
        rc = bag.add(h3_gpu_tensor_new_f32(g_h3g, 1));
        rs = bag.add(h3_gpu_tensor_new_f32(g_h3g, 1));
    }
    if (!tx || !xn || !ones || !n1t || !n2t || !rm || !qkv_b || !out_b || !fc1_b || !fc2_b ||
        !wqkv || !wout || !wfc1 || !wfc2 || !qkv || !q || !k || !v || !ctx || !attn || !h1 ||
        !gated || !down || !rc || !rs || (adaln_mod && !mod) || (q_norm && !qn) || (k_norm && !kn))
        return false;

    H3Cmd cmd;
    if (!cmd.open)
        return false;
    if (!h3_gpu_cast_bf16_to_f32(g_h3g, wqkv, qkv_b, static_cast<uint32_t>(need_qkv)) ||
        !h3_gpu_cast_bf16_to_f32(g_h3g, wout, out_b, static_cast<uint32_t>(need_out)) ||
        !h3_gpu_cast_bf16_to_f32(g_h3g, wfc1, fc1_b, static_cast<uint32_t>(need_fc1)) ||
        !h3_gpu_cast_bf16_to_f32(g_h3g, wfc2, fc2_b, static_cast<uint32_t>(need_fc2)))
        return false;

    if (adaln_mod) {
        if (!h3_gpu_adaln_f32(g_h3g, xn, tx, n1t, mod, rm, static_cast<uint32_t>(T),
                              static_cast<uint32_t>(H), 6u, 0u, 1u, eps))
            return false;
    } else if (!h3_gpu_rms_norm_f32(g_h3g, xn, tx, n1t, static_cast<uint32_t>(T),
                                    static_cast<uint32_t>(H), eps)) {
        return false;
    }
    if (!h3_gpu_linear_f32(g_h3g, qkv, xn, wqkv, nullptr, static_cast<uint32_t>(T),
                           static_cast<uint32_t>(H), static_cast<uint32_t>(3 * I)))
        return false;

    if (use_qkv_rope) {
        if (!h3_gpu_qkv_rope_f32(g_h3g, q, k, v, qkv, qn, kn, rc, rs, static_cast<uint32_t>(T),
                                 static_cast<uint32_t>(heads), static_cast<uint32_t>(hd), rope_half,
                                 eps))
            return false;
    } else {
        if (!split_qkv(g_h3g, q, k, v, qkv, T, I))
            return false;
        const uint32_t qk_rows = static_cast<uint32_t>(T) * static_cast<uint32_t>(heads);
        if (qn && !h3_gpu_rms_norm_f32(g_h3g, q, q, qn, qk_rows, static_cast<uint32_t>(hd), eps))
            return false;
        if (kn && !h3_gpu_rms_norm_f32(g_h3g, k, k, kn, qk_rows, static_cast<uint32_t>(hd), eps))
            return false;
        if (do_rope) {
            if (!cmd.submit())
                return false;
            std::vector<float> qh(static_cast<size_t>(T) * I), kh(static_cast<size_t>(T) * I);
            if (!h3_gpu_tensor_read_f32(q, qh.data(), qh.size()) ||
                !h3_gpu_tensor_read_f32(k, kh.data(), kh.size()))
                return false;
            apply_rope_host(qh.data(), kh.data(), rope_cos, rope_sin, T, I, heads, hd);
            if (!h3_gpu_tensor_write_f32(q, qh.data(), qh.size()) ||
                !h3_gpu_tensor_write_f32(k, kh.data(), kh.size()))
                return false;
            if (!g_h3g || h3_gpu_begin(g_h3g) == 0)
                return false;
            cmd.open = true;
        }
    }

    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    if (!h3_gpu_sdpa_f32(g_h3g, ctx, q, k, v, static_cast<uint32_t>(T), static_cast<uint32_t>(heads),
                         static_cast<uint32_t>(hd), scale) ||
        !h3_gpu_linear_f32(g_h3g, attn, ctx, wout, nullptr, static_cast<uint32_t>(T),
                           static_cast<uint32_t>(I), static_cast<uint32_t>(H)))
        return false;
    if (adaln_mod) {
        if (!h3_gpu_gate_f32(g_h3g, tx, tx, attn, mod, rm, static_cast<uint32_t>(T),
                             static_cast<uint32_t>(H), 6u, 2u))
            return false;
        if (!h3_gpu_adaln_f32(g_h3g, xn, tx, n2t, mod, rm, static_cast<uint32_t>(T),
                              static_cast<uint32_t>(H), 6u, 3u, 4u, eps))
            return false;
    } else {
        if (!h3_gpu_add_scaled_f32(g_h3g, tx, tx, attn, 1.f, 1.f,
                                   static_cast<uint32_t>(T) * static_cast<uint32_t>(H)) ||
            !h3_gpu_rms_norm_f32(g_h3g, xn, tx, n2t, static_cast<uint32_t>(T),
                                 static_cast<uint32_t>(H), eps))
            return false;
    }
    if (!h3_gpu_linear_f32(g_h3g, h1, xn, wfc1, nullptr, static_cast<uint32_t>(T),
                           static_cast<uint32_t>(H), static_cast<uint32_t>(2 * ffn)) ||
        !h3_gpu_swiglu_f32(g_h3g, gated, h1, static_cast<uint32_t>(T), static_cast<uint32_t>(ffn)) ||
        !h3_gpu_linear_f32(g_h3g, down, gated, wfc2, nullptr, static_cast<uint32_t>(T),
                           static_cast<uint32_t>(ffn), static_cast<uint32_t>(H)))
        return false;
    if (adaln_mod) {
        if (!h3_gpu_gate_f32(g_h3g, tx, tx, down, mod, rm, static_cast<uint32_t>(T),
                             static_cast<uint32_t>(H), 6u, 5u))
            return false;
    } else if (!h3_gpu_add_scaled_f32(g_h3g, tx, tx, down, 1.f, 1.f,
                                      static_cast<uint32_t>(T) * static_cast<uint32_t>(H))) {
        return false;
    }
    if (!cmd.submit())
        return false;
    return h3_gpu_tensor_read_f32(tx, x, static_cast<size_t>(T) * H) != 0;
#endif
}

} // namespace official_metal
} // namespace mvllm
