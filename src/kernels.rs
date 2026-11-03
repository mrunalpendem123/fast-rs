//! Metal kernels. Every kernel computes each row (word) independently of the others, in a fixed order, so a word's
//! numbers are the same whether it is decoded alone or checked inside a tree (exact guess-checking).
//! Each source is compiled per shape with `#define`s prepended (see Gpu::pipeline).

pub const HEADER: &str = "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n";

/// Embedding lookup from MLX 4-bit affine (group 64) rows, rounded to bf16 like MLX's dequantize. Defines: D.
pub const EMBED: &str = r#"
kernel void embed(const device int* ids [[buffer(0)]], const device uint* w [[buffer(1)]],
                  const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
                  device float* h [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, r = gid.y;
    if (c >= D) return;
    long t = ids[r];
    uint word = w[t * (D / 8) + c / 8];
    float q = float((word >> (4 * (c % 8))) & 0xFu);
    float v = float(s[t * (D / 64) + c / 64]) * q + float(b[t * (D / 64) + c / 64]);
    h[r * D + c] = float(bfloat(v));
}
"#;

/// Shared tail: from a row already written as half into xh (length KD), per 64-group sums for the v7 matmul.
/// xs = sum(x), xo = sum(x * off(k)) with off = 1024 for even k-steps, 64 for odd ones.
pub const GROUP_SUMS: &str = r#"
inline void group_sums(const device half* xr, device float* xo, device float* xs, uint tid, uint nthreads, int kd) {
    for (int g = int(tid); g < kd / 64; g += int(nthreads)) {
        float a = 0.0f, o = 0.0f;
        for (int ks = 0; ks < 8; ks++) {
            float t = 0.0f;
            for (int j = 0; j < 8; j++) t += float(xr[g * 64 + ks * 8 + j]);
            a += t;
            o += t * ((ks & 1) ? 64.0f : 1024.0f);
        }
        xs[g] = a;
        xo[g] = o;
    }
}
"#;

/// RMSNorm(h) * w -> half x + group sums. One threadgroup (256 threads) per row. Defines: D, EPS.
pub const NORM_PREP: &str = r#"
kernel void norm_prep(const device float* h [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                      device half* xh [[buffer(2)]], device float* xo [[buffer(3)]], device float* xs [[buffer(4)]],
                      uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const device float* hr = h + row * D;
    float ss = 0.0f;
    for (int i = int(tid); i < D; i += 256) ss += hr[i] * hr[i];
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; i++) tot += part[i];
    float r = rsqrt(tot / float(D) + EPS);
    device half* xr = xh + row * D;
    for (int i = int(tid); i < D; i += 256) xr[i] = half(hr[i] * r * float(w[i]));
    threadgroup_barrier(mem_flags::mem_device);
    group_sums(xr, xo + row * (D / 64), xs + row * (D / 64), tid, 256, D);
}
"#;

/// silu(gate) * up -> half x + group sums. gu row = [gate (I) | up (I)]. Defines: I.
pub const SWIGLU_PREP: &str = r#"
kernel void swiglu_prep(const device float* gu [[buffer(0)]], device half* xh [[buffer(1)]],
                        device float* xo [[buffer(2)]], device float* xs [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    const device float* g = gu + row * (2 * I);
    device half* xr = xh + row * I;
    for (int i = int(tid); i < I; i += 256) {
        float a = g[i];
        xr[i] = half(a / (1.0f + exp(-a)) * g[I + i]);
    }
    threadgroup_barrier(mem_flags::mem_device);
    group_sums(xr, xo + row * (I / 64), xs + row * (I / 64), tid, 256, I);
}
"#;

/// v7 check matmul (see fast/qmma7.py): out[M, N] (+)= x[M, K] . W^T. Defines: M, MT, K, N, NT, SK, ADD.
pub const QMM: &str = r#"
kernel void qmm(const device half* x [[buffer(0)]], const device float* xoff [[buffer(1)]], const device float* xs [[buffer(2)]],
                const device uint2* wq [[buffer(3)]], const device bfloat* sp [[buffer(4)]], const device bfloat* bp [[buffer(5)]],
                device float* out [[buffer(6)]],
                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                uint tg [[threadgroup_position_in_grid]]) {
    const int NGRP = K / 64;
    const int t0 = int(tg) * NT;
    const ushort qid = ushort(lane) / 4;
    const ushort fm = (qid & 4) + ((ushort(lane) / 2) % 4);
    const ushort fn = (qid & 2) * 2 + (ushort(lane) % 2) * 2;
    threadgroup float part[SK][8][NT * 8];
    // 8-row tiles one after another: each keeps the register footprint of a single tile; the threadgroup's weight
    // slice is re-read from cache for the next tile instead of spilling registers.
    for (int mt = 0; mt < MT; mt++) {
        simdgroup_float8x8 C[NT];
        _Pragma("unroll") for (int nt = 0; nt < NT; nt++) C[nt] = simdgroup_float8x8(0);
        const device half* xt = x + (mt * 8) * K;
        for (int g = int(sg); g < NGRP; g += SK) {
            uint2 wv[NT];
            _Pragma("unroll") for (int nt = 0; nt < NT; nt++) wv[nt] = wq[((t0 + nt) * NGRP + g) * 32 + lane];
            simdgroup_float8x8 T[NT];
            _Pragma("unroll") for (int nt = 0; nt < NT; nt++) T[nt] = simdgroup_float8x8(0);
            _Pragma("unroll") for (int ks = 0; ks < 8; ks++) {
                simdgroup_half8x8 A;
                simdgroup_load(A, xt + g * 64 + ks * 8, K);
                _Pragma("unroll") for (int nt = 0; nt < NT; nt++) {
                    uint w = (ks < 4) ? wv[nt].x : wv[nt].y;
                    const int j = ks & 3;
                    if (j >= 2) w = w >> 8;
                    uint bits = (j & 1) ? ((w & 0x00F000F0u) | 0x54005400u) : ((w & 0x000F000Fu) | 0x64006400u);
                    half2 hv = as_type<half2>(bits);
                    simdgroup_half8x8 B;
                    B.thread_elements()[0] = hv.x;
                    B.thread_elements()[1] = hv.y;
                    simdgroup_multiply_accumulate(T[nt], A, B, T[nt]);
                }
            }
            int m = mt * 8 + fm;
            float xo_ = xoff[m * NGRP + g], xsum = xs[m * NGRP + g];
            _Pragma("unroll") for (int nt = 0; nt < NT; nt++) {
                int o = ((t0 + nt) * NGRP + g) * 8 + fn;
                float s0 = float(sp[o]), s1 = float(sp[o + 1]), b0 = float(bp[o]), b1 = float(bp[o + 1]);
                C[nt].thread_elements()[0] += s0 * (T[nt].thread_elements()[0] - xo_) + b0 * xsum;
                C[nt].thread_elements()[1] += s1 * (T[nt].thread_elements()[1] - xo_) + b1 * xsum;
            }
        }
        _Pragma("unroll") for (int nt = 0; nt < NT; nt++) simdgroup_store(C[nt], &part[sg][0][nt * 8], NT * 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tid = sg * 32 + lane;
        for (int i = int(tid); i < 8 * NT * 8; i += SK * 32) {
            int r = i / (NT * 8), nn = i % (NT * 8);
            float v = 0.0f;
            for (int q = 0; q < SK; q++) v += part[q][r][nn];
            int m = mt * 8 + r;
            if (m < M) {
                if (ADD) out[m * N + t0 * 8 + nn] += v;
                else out[m * N + t0 * 8 + nn] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"#;

/// Per (row, head): RMSNorm of q / k heads, RoPE (non-traditional) at the row's position, q scaled; k and v written to
/// the memory at slot P + row. qkv row = [q (H*HD) | k (KVH*HD) | v (KVH*HD)]. One simdgroup per (row, head) over
/// H + KVH heads. q heads sit QS apart (2*HD when each carries its output gate); RoPE turns the first ROT dims.
/// Defines: H, KVH, HD, QS, ROT, S, EPS, THETA, SCALE.
pub const QK_ROPE: &str = r#"
struct Args { int n; int P; };
kernel void qk_rope(const device float* qkv [[buffer(0)]], const device bfloat* qn [[buffer(1)]], const device bfloat* kn [[buffer(2)]],
                    const device int* pos [[buffer(3)]], device float* qo [[buffer(4)]],
                    device half* kc [[buffer(5)]], device half* vc [[buffer(6)]], constant Args& a [[buffer(7)]],
                    uint lane [[thread_index_in_simdgroup]], uint gid [[threadgroup_position_in_grid]]) {
    const int NHD = H + KVH;
    int row = int(gid) / NHD, hh = int(gid) % NHD;
    bool isq = hh < H;
    const int W = H * QS + 2 * KVH * HD;
    const device float* src = qkv + row * W + (isq ? hh * QS : H * QS + (hh - H) * HD);
    const int E = HD / 32;                    // elements per lane: lane, lane+32, ...
    float v[E];
    float ss = 0.0f;
    for (int e = 0; e < E; e++) { v[e] = src[int(lane) + 32 * e]; ss += v[e] * v[e]; }
    ss = simd_sum(ss);
    float r = rsqrt(ss / float(HD) + EPS);
    const device bfloat* nw = isq ? qn : kn;
    for (int e = 0; e < E; e++) v[e] = v[e] * r * float(nw[int(lane) + 32 * e]);
    float p = float(pos[row]);
    float o[E];
    const int ER = ROT / 32;                    // elements per lane that RoPE turns
    for (int e = 0; e < ER / 2; e++) {          // pair (i, i + ROT/2), i = lane + 32e
        int i = int(lane) + 32 * e;
        float inv = pow(THETA, -2.0f * float(i) / float(ROT));
        float ang = p * inv, c = cos(ang), s = sin(ang);
        float x1 = v[e], x2 = v[e + ER / 2];
        o[e] = x1 * c - x2 * s;
        o[e + ER / 2] = x2 * c + x1 * s;
    }
    for (int e = ER; e < E; e++) o[e] = v[e];
    if (isq) {
        for (int e = 0; e < E; e++) qo[(row * H + hh) * HD + int(lane) + 32 * e] = o[e] * SCALE;
    } else if (row < a.n) {
        int kh = hh - H;
        int slot = a.P + row;
        for (int e = 0; e < E; e++) kc[(kh * S + slot) * HD + int(lane) + 32 * e] = half(o[e]);
        const device float* vs = qkv + row * W + H * QS + (KVH + kh) * HD;
        for (int e = 0; e < E; e++) vc[(kh * S + slot) * HD + int(lane) + 32 * e] = half(vs[int(lane) + 32 * e]);
    }
}
"#;

/// Tree attention, split pass: the row's logical key list (cache [lo, hi) then ancestors) cut into BS-key blocks.
/// rows[r] = (pos, hi, n_anc, 0); anc[r][MAXD]. Defines: H, KVH, D, S, W, MAXD, BS, NB.
pub const ATTN_SPLIT: &str = r#"
kernel void attn_split(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]], const device half* vc [[buffer(2)]],
                       const device int* rows [[buffer(3)]], const device int* anc [[buffer(4)]], device float* part [[buffer(5)]],
                       uint lane [[thread_index_in_simdgroup]], uint2 tgp [[threadgroup_position_in_grid]]) {
    int gid = int(tgp.x), blk = int(tgp.y);
    int row = gid / H, h = gid % H, kv = h / (H / KVH);
    int pos = rows[row * 4], hi = rows[row * 4 + 1], na = rows[row * 4 + 2];
    int lo = W > 0 ? max(0, pos + 1 - W) : 0;
    const int E = D / 32;
    int nc = hi - lo, L = nc + na;
    int t0 = blk * BS, t1 = min(L, t0 + BS);
    int pbase = ((row * H + h) * NB + blk) * (D + 2);
    if (t0 >= L) { if (lane == 0) { part[pbase + D] = -INFINITY; part[pbase + D + 1] = 0.0f; } return; }
    float qv[E], acc[E];
    for (int e = 0; e < E; e++) { qv[e] = q[(row * H + h) * D + int(lane) * E + e]; acc[e] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int t = t0; t < t1; t++) {
        int p = t < nc ? lo + t : anc[row * MAXD + (t - nc)];
        int base = (kv * S + p) * D + int(lane) * E;
        float d = 0.0f;
        for (int e = 0; e < E; e++) d += qv[e] * float(kc[base + e]);
        float s = simd_sum(d);
        float mn = max(m, s), c = exp(m - mn), pr = exp(s - mn);
        l = l * c + pr;
        for (int e = 0; e < E; e++) acc[e] = acc[e] * c + pr * float(vc[base + e]);
        m = mn;
    }
    for (int e = 0; e < E; e++) part[pbase + int(lane) * E + e] = acc[e];
    if (lane == 0) { part[pbase + D] = m; part[pbase + D + 1] = l; }
}
"#;

/// Merge the blocks in index order, write the head's output as half + group sums for the o-projection.
/// GATE: multiply by sigmoid(gate), the second half of the head's q-projection slot. Defines: H, D, NB, GATE, QS, QW.
/// Output row width H*D.
pub const ATTN_MERGE: &str = r#"
kernel void attn_merge(const device float* part [[buffer(0)]], device half* xh [[buffer(1)]],
                       device float* xo [[buffer(2)]], device float* xs [[buffer(3)]], const device float* qkv [[buffer(4)]],
                       uint lane [[thread_index_in_simdgroup]], uint gid [[threadgroup_position_in_grid]]) {
    const int E = D / 32;
    int row = int(gid) / H, h = int(gid) % H;
    float M_ = -INFINITY;
    for (int b = 0; b < NB; b++) M_ = max(M_, part[(int(gid) * NB + b) * (D + 2) + D]);
    float acc[E], l = 0.0f;
    for (int e = 0; e < E; e++) acc[e] = 0.0f;
    for (int b = 0; b < NB; b++) {
        int pb = (int(gid) * NB + b) * (D + 2);
        float mb = part[pb + D];
        if (mb == -INFINITY) continue;
        float c = exp(mb - M_);
        l += part[pb + D + 1] * c;
        for (int e = 0; e < E; e++) acc[e] += part[pb + int(lane) * E + e] * c;
    }
    device half* xr = xh + row * (H * D) + h * D;
    for (int e = 0; e < E; e++) {
        float o = acc[e] / l;
        if (GATE) { float g = qkv[row * QW + h * QS + D + int(lane) * E + e]; o = o / (1.0f + exp(-g)); }
        xr[int(lane) * E + e] = half(o);
    }
    simdgroup_barrier(mem_flags::mem_device);
    // this head covers D/64 groups of the o-projection input
    const int G = D / 64;
    if (int(lane) < G) {
        int g = int(lane);
        const device half* gr = xr + g * 64;
        float a = 0.0f, o = 0.0f;
        for (int ks = 0; ks < 8; ks++) {
            float t = 0.0f;
            for (int j = 0; j < 8; j++) t += float(gr[ks * 8 + j]);
            a += t;
            o += t * ((ks & 1) ? 64.0f : 1024.0f);
        }
        int gg = h * G + g;
        xs[row * (H * G) + gg] = a;
        xo[row * (H * G) + gg] = o;
    }
}
"#;

/// Greedy pick per row (ties -> lowest index). One threadgroup of 1024 per row. Defines: V.
pub const ARGMAX: &str = r#"
kernel void argmax(const device float* lg [[buffer(0)]], device int* out [[buffer(1)]],
                   uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    const device float* r = lg + long(row) * V;
    float best = -INFINITY; int idx = 0x7fffffff;
    for (int i = int(tid); i < V; i += 1024) { float x = r[i]; if (x > best) { best = x; idx = i; } }
    for (int off = 16; off > 0; off /= 2) {
        float ob = simd_shuffle_down(best, ushort(off)); int oi = simd_shuffle_down(idx, ushort(off));
        if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
    }
    if (lane == 0) { bv[sg] = best; bi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b = bv[0]; int ix = bi[0];
        for (int i = 1; i < 32; i++) if (bv[i] > b || (bv[i] == b && bi[i] < ix)) { b = bv[i]; ix = bi[i]; }
        out[row] = ix;
    }
}
"#;

/// Copy h [n, D] into column block `blk` of feats [n, NB*D] (the guesser reads the target's layer outputs). Defines: D, NBLK.
pub const COPY_COLS: &str = r#"
struct CArgs { int blk; };
kernel void copy_cols(const device float* h [[buffer(0)]], device float* f [[buffer(1)]], constant CArgs& a [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= D) return;
    f[gid.y * (NBLK * D) + a.blk * D + gid.x] = h[gid.y * D + gid.x];
}
"#;

/// float rows [n, KD] -> half + group sums (no norm). Defines: KD.
pub const PREP: &str = r#"
kernel void prep(const device float* x [[buffer(0)]], device half* xh [[buffer(1)]], device float* xo [[buffer(2)]],
                 device float* xs [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    device half* xr = xh + row * KD;
    for (int i = int(tid); i < KD; i += 256) xr[i] = half(x[row * KD + i]);
    threadgroup_barrier(mem_flags::mem_device);
    group_sums(xr, xo + row * (KD / 64), xs + row * (KD / 64), tid, 256, KD);
}
"#;

/// Top 3 (value, index) and log-sum-exp per row. 1024 threads per row. Defines: V.
pub const TOPK3: &str = r#"
kernel void topk3(const device float* lg [[buffer(0)]], device float* tv [[buffer(1)]], device int* ti [[buffer(2)]],
                  uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sv[256 * 3];
    threadgroup int si[256 * 3];
    threadgroup float sm[256], ssum[256];
    const device float* r = lg + long(row) * V;
    float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY; int i0 = 0, i1 = 0, i2 = 0;
    float m = -INFINITY, s = 0.0f;
    for (int i = int(tid); i < V; i += 256) {
        float x = r[i];
        if (x > m) { s = s * exp(m - x) + 1.0f; m = x; } else s += exp(x - m);
        if (x > v0) { v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = x; i0 = i; }
        else if (x > v1) { v2 = v1; i2 = i1; v1 = x; i1 = i; }
        else if (x > v2) { v2 = x; i2 = i; }
    }
    sv[tid * 3] = v0; sv[tid * 3 + 1] = v1; sv[tid * 3 + 2] = v2;
    si[tid * 3] = i0; si[tid * 3 + 1] = i1; si[tid * 3 + 2] = i2;
    sm[tid] = m; ssum[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float M = -INFINITY;
        for (int t = 0; t < 256; t++) M = max(M, sm[t]);
        float S = 0.0f;
        for (int t = 0; t < 256; t++) if (sm[t] > -INFINITY) S += ssum[t] * exp(sm[t] - M);
        float lse = M + log(S);
        float b0 = -INFINITY, b1 = -INFINITY, b2 = -INFINITY; int j0 = 0, j1 = 0, j2 = 0;
        for (int t = 0; t < 768; t++) {
            float x = sv[t]; int ix = si[t];
            if (x > b0) { b2 = b1; j2 = j1; b1 = b0; j1 = j0; b0 = x; j0 = ix; }
            else if (x > b1) { b2 = b1; j2 = j1; b1 = x; j1 = ix; }
            else if (x > b2) { b2 = x; j2 = ix; }
        }
        tv[row * 3] = b0 - lse; tv[row * 3 + 1] = b1 - lse; tv[row * 3 + 2] = b2 - lse;
        ti[row * 3] = j0; ti[row * 3 + 1] = j1; ti[row * 3 + 2] = j2;
    }
}
"#;

/// Gated DeltaNet (Qwen3.5 linear layers), step 1 per (row, 128-channel block): causal depthwise conv (kernel 4) over
/// the row's own history (earlier rows of this pass via prev[row][3] >= 0, else the committed tail: -1 newest .. -3
/// oldest), silu, then q/k heads L2-normalised (q also scaled by 1/DK). keep: also save the row's raw and final
/// values and its b/a inputs, so an accepted guess can be committed later. Defines: C, KD, DK, VD, HV, NLIN, KS.
pub const LIN_CONV: &str = r#"
struct LArgs { int keep; int n; };
kernel void lin_conv(const device float* pre [[buffer(0)]], const device float* tail [[buffer(1)]],
                     const device bfloat* w [[buffer(2)]], const device int* prev [[buffer(3)]],
                     device float* post [[buffer(4)]], device float* keep [[buffer(5)]], constant LArgs& a [[buffer(6)]],
                     uint lane [[thread_index_in_simdgroup]], uint gid [[threadgroup_position_in_grid]]) {
    const int NB = C / 128;
    int row = int(gid) / NB, blk = int(gid) % NB;
    float v[4], x0[4];
    float ss = 0.0f;
    for (int e = 0; e < 4; e++) {
        int c = blk * 128 + int(lane) + 32 * e;
        x0[e] = pre[row * NLIN + c];
        float acc = x0[e] * float(w[c * 4 + 3]);
        for (int k = 0; k < 3; k++) {
            int pr = prev[row * 3 + k];
            float xp = pr >= 0 ? pre[pr * NLIN + c] : tail[(3 + pr) * C + c];
            acc += xp * float(w[c * 4 + 2 - k]);
        }
        v[e] = acc / (1.0f + exp(-acc));
        ss += v[e] * v[e];
    }
    if (blk < 2 * KD / 128) {
        ss = simd_sum(ss);
        float r = rsqrt(ss / float(DK) + 1e-6f / float(DK));
        float sc = blk < KD / 128 ? 1.0f / float(DK) : rsqrt(float(DK));
        for (int e = 0; e < 4; e++) v[e] = v[e] * r * sc;
    }
    for (int e = 0; e < 4; e++) {
        int c = blk * 128 + int(lane) + 32 * e;
        post[row * C + c] = v[e];
        if (a.keep && row < a.n) { keep[row * KS + c] = x0[e]; keep[row * KS + C + c] = v[e]; }
    }
    if (a.keep && row < a.n && blk == 0)
        for (int i = int(lane); i < 2 * HV; i += 32) keep[row * KS + 2 * C + i] = pre[row * NLIN + C + VD + i];
}
"#;

/// Committed conv tail (last 3 raw rows) <- old tail followed by rows ridx[0..n] of src. Defines: C.
pub const LIN_TAIL: &str = r#"
struct TArgs { int n; int stride; };
kernel void lin_tail(const device float* src [[buffer(0)]], const device int* ridx [[buffer(1)]], device float* tail [[buffer(2)]],
                     constant TArgs& a [[buffer(3)]], uint c [[thread_position_in_grid]]) {
    if (int(c) >= C) return;
    float t0 = tail[c], t1 = tail[C + c], t2 = tail[2 * C + c];
    for (int i = 0; i < a.n; i++) { t0 = t1; t1 = t2; t2 = src[ridx[i] * a.stride + c]; }
    tail[c] = t0; tail[C + c] = t1; tail[2 * C + c] = t2;
}
"#;

/// Delta rule, one simdgroup per (value head, value row of the state); each lane owns 4 key columns. Rows are taken in
/// order j = 0..n (row ridx[j]); row j starts from the state after row par[j] (-1: the committed state), so a tree's
/// branches each get their own state (parents come first) and a chain is the same arithmetic. write: store the last
/// row's state as the new committed state. One kernel for prompt chunks, single words, trees and commits: a word's
/// numbers never depend on which. Defines: HV, HK, DK, DV, KD.
pub const LIN_DELTA: &str = r#"
struct DArgs { int n; int write; int pstride; int poff; int abstride; int boff; };
kernel void lin_delta(const device float* post [[buffer(0)]], const device float* ab [[buffer(1)]],
                      const device float* alog [[buffer(2)]], const device bfloat* dtb [[buffer(3)]],
                      device float* state [[buffer(4)]], const device int* par [[buffer(5)]], const device int* ridx [[buffer(6)]],
                      device float* y [[buffer(7)]], constant DArgs& a [[buffer(8)]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint2 tg [[threadgroup_position_in_grid]]) {
    int hv = int(tg.x), dv = int(tg.y) * 4 + int(sg), hk = hv / (HV / HK);
    device float* sp = state + (hv * DV + dv) * DK + int(lane) * 4;
    float s0[4], cur[4], st[16][4];
    for (int i = 0; i < 4; i++) { s0[i] = sp[i]; cur[i] = s0[i]; }
    float A = exp(alog[hv]), bias = float(dtb[hv]);
    for (int j = 0; j < a.n; j++) {
        int r = ridx[j], pj = par[j];
        float s[4];
        for (int i = 0; i < 4; i++) s[i] = pj < 0 ? s0[i] : (pj == j - 1 ? cur[i] : st[pj][i]);
        const device float* pr = post + r * a.pstride + a.poff;
        const device float* q = pr + hk * DK + int(lane) * 4;
        const device float* k = pr + KD + hk * DK + int(lane) * 4;
        float v = pr[2 * KD + hv * DV + dv];
        float bb = ab[r * a.abstride + a.boff + hv], aa = ab[r * a.abstride + a.boff + HV + hv] + bias;
        float soft = aa > 20.0f ? aa : log(1.0f + exp(aa));
        float g = exp(-A * soft), beta = 1.0f / (1.0f + exp(-bb));
        float kv = 0.0f;
        for (int i = 0; i < 4; i++) { s[i] = s[i] * g; kv += s[i] * k[i]; }
        kv = simd_sum(kv);
        float delta = (v - kv) * beta;
        float o = 0.0f;
        for (int i = 0; i < 4; i++) { s[i] = s[i] + k[i] * delta; o += s[i] * q[i]; }
        o = simd_sum(o);
        if (lane == 0) y[r * (HV * DV) + hv * DV + dv] = o;
        for (int i = 0; i < 4; i++) cur[i] = s[i];
        if (j < 16) for (int i = 0; i < 4; i++) st[j][i] = s[i];
    }
    if (a.write && a.n > 0) for (int i = 0; i < 4; i++) sp[i] = cur[i];
}
"#;

/// Per (row, value head): RMSNorm(y) * w * silu(z) -> half + group sums for the out-projection. Defines: HV, DV, C, NLIN, EPS.
pub const LIN_NORM: &str = r#"
kernel void lin_norm(const device float* y [[buffer(0)]], const device float* zb [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                     device half* xh [[buffer(3)]], device float* xo [[buffer(4)]], device float* xs [[buffer(5)]],
                     uint lane [[thread_index_in_simdgroup]], uint gid [[threadgroup_position_in_grid]]) {
    int row = int(gid) / HV, hv = int(gid) % HV;
    const device float* yr = y + row * (HV * DV) + hv * DV;
    float v[DV / 32];
    float ss = 0.0f;
    for (int e = 0; e < DV / 32; e++) { v[e] = yr[int(lane) + 32 * e]; ss += v[e] * v[e]; }
    ss = simd_sum(ss);
    float r = rsqrt(ss / float(DV) + EPS);
    device half* xr = xh + row * (HV * DV) + hv * DV;
    for (int e = 0; e < DV / 32; e++) {
        int d = int(lane) + 32 * e;
        float z = zb[row * NLIN + C + hv * DV + d];
        xr[d] = half(v[e] * r * float(w[d]) * (z / (1.0f + exp(-z))));
    }
    simdgroup_barrier(mem_flags::mem_device);
    const int G = DV / 64;
    if (int(lane) < G) {
        int g = int(lane);
        const device half* gr = xr + g * 64;
        float s_ = 0.0f, o = 0.0f;
        for (int ks = 0; ks < 8; ks++) {
            float t = 0.0f;
            for (int j = 0; j < 8; j++) t += float(gr[ks * 8 + j]);
            s_ += t;
            o += t * ((ks & 1) ? 64.0f : 1024.0f);
        }
        xs[row * (HV * G) + hv * G + g] = s_;
        xo[row * (HV * G) + hv * G + g] = o;
    }
}
"#;
