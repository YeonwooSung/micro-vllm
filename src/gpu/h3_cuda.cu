#include "h3_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Device kernels for H3 DiT residual. Semantics match h3_dit_block_cpu.
// Scores live in device memory [T,T]; no tokens<=256 cap.

namespace {

bool ck(cudaError_t e) { return e == cudaSuccess; }

template <typename T>
T *dalloc(size_t n) {
    T *p = nullptr;
    if (n == 0)
        n = 1;
    if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess)
        return nullptr;
    return p;
}

void dfree(void *p) {
    if (p)
        cudaFree(p);
}

dim3 grid1(int n) { return dim3(static_cast<unsigned>((n + 63) / 64)); }

dim3 grid2(int gx, int gy) {
    return dim3(static_cast<unsigned>((gx + 15) / 16), static_cast<unsigned>((gy + 15) / 16), 1);
}

struct DitWs {
    uint8_t *blob;
    float *x, *xn, *qkv, *ctx, *attn, *h1, *gated, *down, *scores, *mod, *qn, *kn, *cos, *sin;
    int cap_T, cap_H, cap_I, cap_ffn, cap_hd;
    int64_t cap_blob;
};
DitWs g_ws;

void ws_release(DitWs &w) {
    dfree(w.blob);
    dfree(w.x);
    dfree(w.xn);
    dfree(w.qkv);
    dfree(w.ctx);
    dfree(w.attn);
    dfree(w.h1);
    dfree(w.gated);
    dfree(w.down);
    dfree(w.scores);
    dfree(w.mod);
    dfree(w.qn);
    dfree(w.kn);
    dfree(w.cos);
    dfree(w.sin);
    w = DitWs{};
}

void ws_free() { ws_release(g_ws); }

size_t ws_bytes() {
    if (!g_ws.blob && !g_ws.x)
        return 0;
    const size_t T = static_cast<size_t>(g_ws.cap_T);
    const size_t H = static_cast<size_t>(g_ws.cap_H);
    const size_t I = static_cast<size_t>(g_ws.cap_I);
    const size_t ffn = static_cast<size_t>(g_ws.cap_ffn);
    const size_t hd = static_cast<size_t>(g_ws.cap_hd);
    size_t n = static_cast<size_t>(g_ws.cap_blob);
    n += sizeof(float) * T * H;       // x
    n += sizeof(float) * T * H;       // xn
    n += sizeof(float) * T * 3 * I;   // qkv
    n += sizeof(float) * T * I;       // ctx
    n += sizeof(float) * T * H;       // attn
    n += sizeof(float) * T * 2 * ffn; // h1
    n += sizeof(float) * T * ffn;     // gated
    n += sizeof(float) * T * H;       // down
    n += sizeof(float) * T * T;       // scores
    n += sizeof(float) * 6 * H;       // mod
    n += sizeof(float) * hd;          // qn
    n += sizeof(float) * hd;          // kn
    n += sizeof(float) * T * 48;      // cos
    n += sizeof(float) * T * 48;      // sin
    return n;
}

// One transaction: alloc new first; on any failure free nxt and keep g_ws.
bool ws_ensure(int T, int H, int I, int ffn, int hd, int64_t blob_n) {
    if (T <= g_ws.cap_T && H <= g_ws.cap_H && I <= g_ws.cap_I && ffn <= g_ws.cap_ffn &&
        hd <= g_ws.cap_hd && blob_n <= g_ws.cap_blob && g_ws.blob)
        return true;

    const int nT = T > g_ws.cap_T ? T : g_ws.cap_T;
    const int nH = H > g_ws.cap_H ? H : g_ws.cap_H;
    const int nI = I > g_ws.cap_I ? I : g_ws.cap_I;
    const int nffn = ffn > g_ws.cap_ffn ? ffn : g_ws.cap_ffn;
    const int nhd = hd > g_ws.cap_hd ? hd : g_ws.cap_hd;
    const int64_t nblob = blob_n > g_ws.cap_blob ? blob_n : g_ws.cap_blob;

    DitWs nxt{};
    nxt.blob = dalloc<uint8_t>(static_cast<size_t>(nblob));
    nxt.x = dalloc<float>(static_cast<size_t>(nT) * nH);
    nxt.xn = dalloc<float>(static_cast<size_t>(nT) * nH);
    nxt.qkv = dalloc<float>(static_cast<size_t>(nT) * 3 * nI);
    nxt.ctx = dalloc<float>(static_cast<size_t>(nT) * nI);
    nxt.attn = dalloc<float>(static_cast<size_t>(nT) * nH);
    nxt.h1 = dalloc<float>(static_cast<size_t>(nT) * 2 * nffn);
    nxt.gated = dalloc<float>(static_cast<size_t>(nT) * nffn);
    nxt.down = dalloc<float>(static_cast<size_t>(nT) * nH);
    nxt.scores = dalloc<float>(static_cast<size_t>(nT) * nT);
    nxt.mod = dalloc<float>(static_cast<size_t>(6) * nH);
    nxt.qn = dalloc<float>(static_cast<size_t>(nhd));
    nxt.kn = dalloc<float>(static_cast<size_t>(nhd));
    nxt.cos = dalloc<float>(static_cast<size_t>(nT) * 48);
    nxt.sin = dalloc<float>(static_cast<size_t>(nT) * 48);

    const bool ok = nxt.blob && nxt.x && nxt.xn && nxt.qkv && nxt.ctx && nxt.attn && nxt.h1 &&
                    nxt.gated && nxt.down && nxt.scores && nxt.mod && nxt.qn && nxt.kn && nxt.cos &&
                    nxt.sin;
    if (!ok) {
        ws_release(nxt);
        return false;
    }
    nxt.cap_T = nT;
    nxt.cap_H = nH;
    nxt.cap_I = nI;
    nxt.cap_ffn = nffn;
    nxt.cap_hd = nhd;
    nxt.cap_blob = nblob;
    ws_release(g_ws);
    g_ws = nxt;
    return true;
}

// RMSNorm with ones, then y = y * (1 + scale) + shift when has_mod.
__global__ void k_adaln(const float *x, const float *mod, float *y, int T, int H, float eps,
                        int has_mod, int scale_slot, int shift_slot) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T)
        return;
    const float *xs = x + static_cast<size_t>(t) * H;
    float *ys = y + static_cast<size_t>(t) * H;
    float ss = 0.f;
    for (int i = 0; i < H; ++i)
        ss += xs[i] * xs[i];
    float inv = rsqrtf(ss / static_cast<float>(H) + eps);
    if (has_mod) {
        const float *s = mod + static_cast<size_t>(scale_slot) * H;
        const float *b = mod + static_cast<size_t>(shift_slot) * H;
        for (int i = 0; i < H; ++i)
            ys[i] = xs[i] * inv * (1.f + s[i]) + b[i];
    } else {
        for (int i = 0; i < H; ++i)
            ys[i] = xs[i] * inv;
    }
}

// y[S,O] = x[S,I] @ W_bf16[O,I]^T, IEEE f32 accumulate. Tile 16x16, BK=16.
__global__ void k_gemm_bf16_tiled(const float *x, const uint16_t *w, float *y, int S, int I, int O) {
    constexpr int BM = 16, BN = 16, BK = 16;
    const int s0 = static_cast<int>(blockIdx.y) * BM;
    const int o0 = static_cast<int>(blockIdx.x) * BN;
    const int ty = static_cast<int>(threadIdx.y);
    const int tx = static_cast<int>(threadIdx.x);
    const int s = s0 + ty;
    const int o = o0 + tx;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    float acc = 0.f;
    for (int k0 = 0; k0 < I; k0 += BK) {
        const int ka = k0 + tx;
        if (s < S && ka < I)
            As[ty][tx] = x[static_cast<size_t>(s) * I + ka];
        else
            As[ty][tx] = 0.f;
        const int kb = k0 + ty;
        if (o < O && kb < I)
            Bs[tx][ty] = __uint_as_float(static_cast<unsigned>(w[static_cast<size_t>(o) * I + kb])
                                         << 16);
        else
            Bs[tx][ty] = 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k)
            acc += As[ty][k] * Bs[tx][k];
        __syncthreads();
    }
    if (s < S && o < O)
        y[static_cast<size_t>(s) * O + o] = acc;
}

__global__ void k_qk_rmsnorm(float *qkv, const float *q_norm, const float *k_norm, int T, int I,
                             int hd, int heads, int do_q, int do_k, float eps) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= heads || t >= T)
        return;
    if (do_q) {
        float *q = qkv + static_cast<size_t>(t) * (3 * I) + h * hd;
        float ss = 0.f;
        for (int i = 0; i < hd; ++i)
            ss += q[i] * q[i];
        float inv = rsqrtf(ss / static_cast<float>(hd) + eps);
        for (int i = 0; i < hd; ++i)
            q[i] = q[i] * inv * q_norm[i];
    }
    if (do_k) {
        float *k = qkv + static_cast<size_t>(t) * (3 * I) + I + h * hd;
        float ss = 0.f;
        for (int i = 0; i < hd; ++i)
            ss += k[i] * k[i];
        float inv = rsqrtf(ss / static_cast<float>(hd) + eps);
        for (int i = 0; i < hd; ++i)
            k[i] = k[i] * inv * k_norm[i];
    }
}

// First 48 dims; pair d with d+48. Applied when hd >= 96.
__global__ void k_rope_qk(float *qkv, const float *cos, const float *sin, int T, int I, int hd,
                          int heads) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= heads || t >= T)
        return;
    if (hd < 96)
        return;
    const int half = 48;
    const float *c = cos + static_cast<size_t>(t) * half;
    const float *s = sin + static_cast<size_t>(t) * half;
    float *q = qkv + static_cast<size_t>(t) * (3 * I) + h * hd;
    float *k = qkv + static_cast<size_t>(t) * (3 * I) + I + h * hd;
    for (int d = 0; d < half; ++d) {
        float qa = q[d], qb = q[d + half];
        q[d] = qa * c[d] - qb * s[d];
        q[d + half] = qa * s[d] + qb * c[d];
        float ka = k[d], kb = k[d + half];
        k[d] = ka * c[d] - kb * s[d];
        k[d + half] = ka * s[d] + kb * c[d];
    }
}

// Full (non-causal) scores for one head into scores[T,T].
__global__ void k_attn_scores(const float *qkv, float *scores, int T, int I, int hd, int h) {
    int ki = blockIdx.x * blockDim.x + threadIdx.x;
    int qi = blockIdx.y * blockDim.y + threadIdx.y;
    if (qi >= T || ki >= T)
        return;
    const float scale = rsqrtf(static_cast<float>(hd));
    const float *q = qkv + static_cast<size_t>(qi) * (3 * I) + h * hd;
    const float *k = qkv + static_cast<size_t>(ki) * (3 * I) + I + h * hd;
    float acc = 0.f;
    for (int d = 0; d < hd; ++d)
        acc += q[d] * k[d];
    scores[static_cast<size_t>(qi) * T + ki] = acc * scale;
}

__global__ void k_attn_softmax(float *scores, int T) {
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= T)
        return;
    float *row = scores + static_cast<size_t>(qi) * T;
    float m = row[0];
    for (int i = 1; i < T; ++i) {
        if (row[i] > m)
            m = row[i];
    }
    float sum = 0.f;
    for (int i = 0; i < T; ++i) {
        row[i] = expf(row[i] - m);
        sum += row[i];
    }
    float inv = (sum == 0.f) ? 0.f : 1.f / sum;
    for (int i = 0; i < T; ++i)
        row[i] *= inv;
}

__global__ void k_attn_av(const float *qkv, const float *scores, float *ctx, int T, int I, int hd,
                          int h) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int qi = blockIdx.y * blockDim.y + threadIdx.y;
    if (qi >= T || d >= hd)
        return;
    const float *srow = scores + static_cast<size_t>(qi) * T;
    float acc = 0.f;
    for (int vi = 0; vi < T; ++vi) {
        const float *v = qkv + static_cast<size_t>(vi) * (3 * I) + 2 * I + h * hd;
        acc += srow[vi] * v[d];
    }
    ctx[static_cast<size_t>(qi) * I + h * hd + d] = acc;
}

// x += (has_mod ? mod[slot] : 1) * y
__global__ void k_residual_gate(float *x, const float *y, const float *mod, int n, int H,
                                int has_mod, int slot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int col = i % H;
    float g = has_mod ? mod[static_cast<size_t>(slot) * H + col] : 1.f;
    x[i] += g * y[i];
}

__global__ void k_swiglu_pack(const float *h1, float *gated, int T, int ffn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= T || i >= ffn)
        return;
    float g = h1[static_cast<size_t>(t) * (2 * ffn) + i];
    float u = h1[static_cast<size_t>(t) * (2 * ffn) + ffn + i];
    float sig = 1.f / (1.f + expf(-g));
    gated[static_cast<size_t>(t) * ffn + i] = g * sig * u;
}

struct GemmWs {
    float *x = nullptr;
    float *y = nullptr;
    uint16_t *w = nullptr;
    int cap_S = 0, cap_I = 0, cap_O = 0;
};
GemmWs g_gemm;

void gemm_ws_free() {
    dfree(g_gemm.x);
    dfree(g_gemm.y);
    dfree(g_gemm.w);
    g_gemm = GemmWs{};
}

bool gemm_ws_ensure(int S, int I, int O) {
    if (S <= g_gemm.cap_S && I <= g_gemm.cap_I && O <= g_gemm.cap_O && g_gemm.w)
        return true;
    const int nS = S > g_gemm.cap_S ? S : g_gemm.cap_S;
    const int nI = I > g_gemm.cap_I ? I : g_gemm.cap_I;
    const int nO = O > g_gemm.cap_O ? O : g_gemm.cap_O;
    GemmWs nxt{};
    nxt.x = dalloc<float>(static_cast<size_t>(nS) * nI);
    nxt.y = dalloc<float>(static_cast<size_t>(nS) * nO);
    nxt.w = dalloc<uint16_t>(static_cast<size_t>(nO) * nI);
    if (!nxt.x || !nxt.y || !nxt.w) {
        dfree(nxt.x);
        dfree(nxt.y);
        dfree(nxt.w);
        return false;
    }
    nxt.cap_S = nS;
    nxt.cap_I = nI;
    nxt.cap_O = nO;
    gemm_ws_free();
    g_gemm = nxt;
    return true;
}

int gemm_bf16_dev(const float *x, const uint16_t *w, float *y, int S, int I, int O) {
    if (!x || !w || !y || S <= 0 || I <= 0 || O <= 0)
        return 1;
    if (!gemm_ws_ensure(S, I, O))
        return 2;
    const size_t xb = sizeof(float) * static_cast<size_t>(S) * I;
    const size_t yb = sizeof(float) * static_cast<size_t>(S) * O;
    const size_t wb = sizeof(uint16_t) * static_cast<size_t>(O) * I;
    bool ok = ck(cudaMemcpy(g_gemm.x, x, xb, cudaMemcpyHostToDevice)) &&
              ck(cudaMemcpy(g_gemm.w, w, wb, cudaMemcpyHostToDevice));
    if (!ok)
        return 2;
    k_gemm_bf16_tiled<<<grid2(O, S), dim3(16, 16)>>>(g_gemm.x, g_gemm.w, g_gemm.y, S, I, O);
    if (!ck(cudaDeviceSynchronize()) || !ck(cudaMemcpy(y, g_gemm.y, yb, cudaMemcpyDeviceToHost)))
        return 2;
    return 0;
}

size_t gemm_ws_bytes() {
    if (!g_gemm.w)
        return 0;
    size_t n = sizeof(float) * static_cast<size_t>(g_gemm.cap_S) * g_gemm.cap_I;
    n += sizeof(float) * static_cast<size_t>(g_gemm.cap_S) * g_gemm.cap_O;
    n += sizeof(uint16_t) * static_cast<size_t>(g_gemm.cap_O) * g_gemm.cap_I;
    return n;
}

} // namespace

extern "C" int h3_cuda_probe(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0)
        return 1;
    return 0;
}

extern "C" void h3_cuda_ws_free(void) {
    ws_free();
    gemm_ws_free();
}

extern "C" size_t h3_cuda_workspace_bytes(void) { return ws_bytes() + gemm_ws_bytes(); }

extern "C" int h3_cuda_gemm_bf16_dev(const float *x, const uint16_t *w, float *y, int S, int I,
                                     int O) {
    return gemm_bf16_dev(x, w, y, S, I, O);
}

extern "C" int h3_cuda_dit_residual_dev(const uint8_t *blob, int64_t qkv_bytes, int64_t out_bytes,
                                        int64_t fc1_bytes, int64_t fc2_bytes, int hidden, int inner,
                                        int ffn, int head_dim, float *x, int tokens, float eps,
                                        const float *adaln_mod, const float *q_norm,
                                        const float *k_norm, const float *rope_cos,
                                        const float *rope_sin) {
    if (!blob || !x || hidden <= 0 || inner <= 0 || ffn <= 0 || tokens <= 0)
        return 1;
    if (qkv_bytes < 2 || out_bytes < 2 || fc1_bytes < 2 || fc2_bytes < 2)
        return 1;

    const int T = tokens;
    const int H = hidden;
    const int I = inner;
    int hd = head_dim > 0 ? head_dim : I;
    int heads = I / hd;
    if (heads < 1) {
        heads = 1;
        hd = I;
    }
    const int qkv_n = static_cast<int>(qkv_bytes / 2);
    const int out_n = static_cast<int>(out_bytes / 2);
    const int fc1_n = static_cast<int>(fc1_bytes / 2);
    const int fc2_n = static_cast<int>(fc2_bytes / 2);
    if (qkv_n <= 0 || out_n <= 0 || fc1_n <= 0 || fc2_n <= 0)
        return 1;
    const int64_t blob_n = qkv_bytes + out_bytes + fc1_bytes + fc2_bytes;
    const int has_mod = adaln_mod ? 1 : 0;
    const int do_q = q_norm ? 1 : 0;
    const int do_k = k_norm ? 1 : 0;
    const bool do_rope = rope_cos && rope_sin && hd >= 96;

    if (!ws_ensure(T, H, I, ffn, hd, blob_n))
        return 2;

    bool ok = ck(cudaMemcpy(g_ws.blob, blob, static_cast<size_t>(blob_n), cudaMemcpyHostToDevice)) &&
              ck(cudaMemcpy(g_ws.x, x, sizeof(float) * static_cast<size_t>(T) * H,
                            cudaMemcpyHostToDevice));
    if (ok && has_mod)
        ok = ck(cudaMemcpy(g_ws.mod, adaln_mod, sizeof(float) * static_cast<size_t>(6) * H,
                           cudaMemcpyHostToDevice));
    if (ok && do_q)
        ok = ck(cudaMemcpy(g_ws.qn, q_norm, sizeof(float) * static_cast<size_t>(hd),
                           cudaMemcpyHostToDevice));
    if (ok && do_k)
        ok = ck(cudaMemcpy(g_ws.kn, k_norm, sizeof(float) * static_cast<size_t>(hd),
                           cudaMemcpyHostToDevice));
    if (ok && do_rope) {
        const size_t rb = sizeof(float) * static_cast<size_t>(T) * 48;
        ok = ck(cudaMemcpy(g_ws.cos, rope_cos, rb, cudaMemcpyHostToDevice)) &&
             ck(cudaMemcpy(g_ws.sin, rope_sin, rb, cudaMemcpyHostToDevice));
    }

    if (ok) {
        const uint16_t *w_qkv = reinterpret_cast<const uint16_t *>(g_ws.blob);
        const uint16_t *w_out = reinterpret_cast<const uint16_t *>(g_ws.blob + qkv_bytes);
        const uint16_t *w_fc1 =
            reinterpret_cast<const uint16_t *>(g_ws.blob + qkv_bytes + out_bytes);
        const uint16_t *w_fc2 =
            reinterpret_cast<const uint16_t *>(g_ws.blob + qkv_bytes + out_bytes + fc1_bytes);

        k_adaln<<<grid1(T), 64>>>(g_ws.x, g_ws.mod, g_ws.xn, T, H, eps, has_mod, 0, 1);
        k_gemm_bf16_tiled<<<grid2(3 * I, T), dim3(16, 16)>>>(g_ws.xn, w_qkv, g_ws.qkv, T, H, 3 * I);
        if (do_q || do_k)
            k_qk_rmsnorm<<<grid2(heads, T), dim3(16, 16)>>>(g_ws.qkv, g_ws.qn, g_ws.kn, T, I, hd,
                                                            heads, do_q, do_k, eps);
        if (do_rope)
            k_rope_qk<<<grid2(heads, T), dim3(16, 16)>>>(g_ws.qkv, g_ws.cos, g_ws.sin, T, I, hd,
                                                         heads);

        ok = ck(cudaMemset(g_ws.ctx, 0, sizeof(float) * static_cast<size_t>(T) * I));
        for (int h = 0; ok && h < heads; ++h) {
            k_attn_scores<<<grid2(T, T), dim3(16, 16)>>>(g_ws.qkv, g_ws.scores, T, I, hd, h);
            k_attn_softmax<<<grid1(T), 64>>>(g_ws.scores, T);
            k_attn_av<<<grid2(hd, T), dim3(16, 16)>>>(g_ws.qkv, g_ws.scores, g_ws.ctx, T, I, hd, h);
            ok = ck(cudaGetLastError());
        }

        if (ok) {
            k_gemm_bf16_tiled<<<grid2(H, T), dim3(16, 16)>>>(g_ws.ctx, w_out, g_ws.attn, T, I, H);
            k_residual_gate<<<grid1(T * H), 64>>>(g_ws.x, g_ws.attn, g_ws.mod, T * H, H, has_mod, 2);

            k_adaln<<<grid1(T), 64>>>(g_ws.x, g_ws.mod, g_ws.xn, T, H, eps, has_mod, 3, 4);
            k_gemm_bf16_tiled<<<grid2(2 * ffn, T), dim3(16, 16)>>>(g_ws.xn, w_fc1, g_ws.h1, T, H,
                                                                  2 * ffn);
            k_swiglu_pack<<<grid2(ffn, T), dim3(16, 16)>>>(g_ws.h1, g_ws.gated, T, ffn);
            k_gemm_bf16_tiled<<<grid2(H, T), dim3(16, 16)>>>(g_ws.gated, w_fc2, g_ws.down, T, ffn,
                                                            H);
            k_residual_gate<<<grid1(T * H), 64>>>(g_ws.x, g_ws.down, g_ws.mod, T * H, H, has_mod, 5);

            ok = ck(cudaDeviceSynchronize()) &&
                 ck(cudaMemcpy(x, g_ws.x, sizeof(float) * static_cast<size_t>(T) * H,
                               cudaMemcpyDeviceToHost));
        }
    }

    return ok ? 0 : 2;
}
