#include "h3_cuda.hpp"

#include <cuda_runtime.h>

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

__global__ void k_bf16_to_f32(const uint16_t *src, float *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    dst[i] = __uint_as_float(static_cast<unsigned>(src[i]) << 16);
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

// y[S,O] = x[S,I] @ W[O,I]^T
__global__ void k_gemm_f32(const float *x, const float *w, float *y, int S, int I, int O) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= O || s >= S)
        return;
    const float *xs = x + static_cast<size_t>(s) * I;
    const float *wo = w + static_cast<size_t>(o) * I;
    float acc = 0.f;
    for (int i = 0; i < I; ++i)
        acc += xs[i] * wo[i];
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

} // namespace

extern "C" int h3_cuda_probe(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0)
        return 1;
    return 0;
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
    const size_t blob_n =
        static_cast<size_t>(qkv_bytes) + static_cast<size_t>(out_bytes) +
        static_cast<size_t>(fc1_bytes) + static_cast<size_t>(fc2_bytes);
    const int has_mod = adaln_mod ? 1 : 0;
    const int do_q = q_norm ? 1 : 0;
    const int do_k = k_norm ? 1 : 0;
    const bool do_rope = rope_cos && rope_sin && hd >= 96;

    uint8_t *dblob = dalloc<uint8_t>(blob_n);
    float *dx = dalloc<float>(static_cast<size_t>(T) * H);
    float *dxn = dalloc<float>(static_cast<size_t>(T) * H);
    float *wqkv = dalloc<float>(static_cast<size_t>(qkv_n));
    float *wout = dalloc<float>(static_cast<size_t>(out_n));
    float *wfc1 = dalloc<float>(static_cast<size_t>(fc1_n));
    float *wfc2 = dalloc<float>(static_cast<size_t>(fc2_n));
    float *dqkv = dalloc<float>(static_cast<size_t>(T) * 3 * I);
    float *dctx = dalloc<float>(static_cast<size_t>(T) * I);
    float *dattn = dalloc<float>(static_cast<size_t>(T) * H);
    float *dh1 = dalloc<float>(static_cast<size_t>(T) * 2 * ffn);
    float *dgated = dalloc<float>(static_cast<size_t>(T) * ffn);
    float *ddown = dalloc<float>(static_cast<size_t>(T) * H);
    float *dscores = dalloc<float>(static_cast<size_t>(T) * T);
    float *dmod = has_mod ? dalloc<float>(static_cast<size_t>(6) * H) : nullptr;
    float *dqn = do_q ? dalloc<float>(static_cast<size_t>(hd)) : nullptr;
    float *dkn = do_k ? dalloc<float>(static_cast<size_t>(hd)) : nullptr;
    float *dcos = do_rope ? dalloc<float>(static_cast<size_t>(T) * 48) : nullptr;
    float *dsin = do_rope ? dalloc<float>(static_cast<size_t>(T) * 48) : nullptr;

    bool ok = dblob && dx && dxn && wqkv && wout && wfc1 && wfc2 && dqkv && dctx && dattn && dh1 &&
              dgated && ddown && dscores;
    if (has_mod)
        ok = ok && dmod;
    if (do_q)
        ok = ok && dqn;
    if (do_k)
        ok = ok && dkn;
    if (do_rope)
        ok = ok && dcos && dsin;

    if (ok)
        ok = ck(cudaMemcpy(dblob, blob, blob_n, cudaMemcpyHostToDevice)) &&
             ck(cudaMemcpy(dx, x, sizeof(float) * static_cast<size_t>(T) * H,
                           cudaMemcpyHostToDevice));
    if (ok && has_mod)
        ok = ck(cudaMemcpy(dmod, adaln_mod, sizeof(float) * static_cast<size_t>(6) * H,
                           cudaMemcpyHostToDevice));
    if (ok && do_q)
        ok = ck(cudaMemcpy(dqn, q_norm, sizeof(float) * static_cast<size_t>(hd),
                           cudaMemcpyHostToDevice));
    if (ok && do_k)
        ok = ck(cudaMemcpy(dkn, k_norm, sizeof(float) * static_cast<size_t>(hd),
                           cudaMemcpyHostToDevice));
    if (ok && do_rope) {
        const size_t rb = sizeof(float) * static_cast<size_t>(T) * 48;
        ok = ck(cudaMemcpy(dcos, rope_cos, rb, cudaMemcpyHostToDevice)) &&
             ck(cudaMemcpy(dsin, rope_sin, rb, cudaMemcpyHostToDevice));
    }

    if (ok) {
        k_bf16_to_f32<<<grid1(qkv_n), 64>>>(reinterpret_cast<const uint16_t *>(dblob), wqkv, qkv_n);
        k_bf16_to_f32<<<grid1(out_n), 64>>>(
            reinterpret_cast<const uint16_t *>(dblob + qkv_bytes), wout, out_n);
        k_bf16_to_f32<<<grid1(fc1_n), 64>>>(
            reinterpret_cast<const uint16_t *>(dblob + qkv_bytes + out_bytes), wfc1, fc1_n);
        k_bf16_to_f32<<<grid1(fc2_n), 64>>>(
            reinterpret_cast<const uint16_t *>(dblob + qkv_bytes + out_bytes + fc1_bytes), wfc2,
            fc2_n);

        k_adaln<<<grid1(T), 64>>>(dx, dmod, dxn, T, H, eps, has_mod, 0, 1);
        k_gemm_f32<<<grid2(3 * I, T), dim3(16, 16)>>>(dxn, wqkv, dqkv, T, H, 3 * I);
        if (do_q || do_k)
            k_qk_rmsnorm<<<grid2(heads, T), dim3(16, 16)>>>(dqkv, dqn, dkn, T, I, hd, heads, do_q,
                                                            do_k, eps);
        if (do_rope)
            k_rope_qk<<<grid2(heads, T), dim3(16, 16)>>>(dqkv, dcos, dsin, T, I, hd, heads);

        ok = ck(cudaMemset(dctx, 0, sizeof(float) * static_cast<size_t>(T) * I));
        for (int h = 0; ok && h < heads; ++h) {
            k_attn_scores<<<grid2(T, T), dim3(16, 16)>>>(dqkv, dscores, T, I, hd, h);
            k_attn_softmax<<<grid1(T), 64>>>(dscores, T);
            k_attn_av<<<grid2(hd, T), dim3(16, 16)>>>(dqkv, dscores, dctx, T, I, hd, h);
            ok = ck(cudaGetLastError());
        }

        if (ok) {
            k_gemm_f32<<<grid2(H, T), dim3(16, 16)>>>(dctx, wout, dattn, T, I, H);
            k_residual_gate<<<grid1(T * H), 64>>>(dx, dattn, dmod, T * H, H, has_mod, 2);

            k_adaln<<<grid1(T), 64>>>(dx, dmod, dxn, T, H, eps, has_mod, 3, 4);
            k_gemm_f32<<<grid2(2 * ffn, T), dim3(16, 16)>>>(dxn, wfc1, dh1, T, H, 2 * ffn);
            k_swiglu_pack<<<grid2(ffn, T), dim3(16, 16)>>>(dh1, dgated, T, ffn);
            k_gemm_f32<<<grid2(H, T), dim3(16, 16)>>>(dgated, wfc2, ddown, T, ffn, H);
            k_residual_gate<<<grid1(T * H), 64>>>(dx, ddown, dmod, T * H, H, has_mod, 5);

            ok = ck(cudaDeviceSynchronize()) &&
                 ck(cudaMemcpy(x, dx, sizeof(float) * static_cast<size_t>(T) * H,
                               cudaMemcpyDeviceToHost));
        }
    }

    dfree(dblob);
    dfree(dx);
    dfree(dxn);
    dfree(wqkv);
    dfree(wout);
    dfree(wfc1);
    dfree(wfc2);
    dfree(dqkv);
    dfree(dctx);
    dfree(dattn);
    dfree(dh1);
    dfree(dgated);
    dfree(ddown);
    dfree(dscores);
    dfree(dmod);
    dfree(dqn);
    dfree(dkn);
    dfree(dcos);
    dfree(dsin);
    return ok ? 0 : 2;
}
