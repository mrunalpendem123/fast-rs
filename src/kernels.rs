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

/// Prompt-reading variant of QMM: each threadgroup takes RB 8-row tiles (grid y) and decodes every weight fragment
/// once for all of them, instead of once per tile. Per row the sums run in the same order as QMM's.
/// Defines: M, K, N, NT, SK, RB, ADD.
pub const QMM_BIG: &str = r#"
kernel void qmm_big(const device half* x [[buffer(0)]], const device float* xoff [[buffer(1)]], const device float* xs [[buffer(2)]],
                const device uint2* wq [[buffer(3)]], const device bfloat* sp [[buffer(4)]], const device bfloat* bp [[buffer(5)]],
                device float* out [[buffer(6)]],
                uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                uint2 tg [[threadgroup_position_in_grid]]) {
    const int NGRP = K / 64;
    const int t0 = int(tg.x) * NT;
    const int r0 = int(tg.y) * RB * 8;
    const ushort qid = ushort(lane) / 4;
    const ushort fm = (qid & 4) + ((ushort(lane) / 2) % 4);
    const ushort fn = (qid & 2) * 2 + (ushort(lane) % 2) * 2;
    threadgroup float part[SK][RB * 8][NT * 8];
    threadgroup half Bs[SK][64][NT * 8];
    simdgroup_float8x8 C[RB][NT];
    _Pragma("unroll") for (int rb = 0; rb < RB; rb++) _Pragma("unroll") for (int nt = 0; nt < NT; nt++) C[rb][nt] = simdgroup_float8x8(0);
    for (int g = int(sg); g < NGRP; g += SK) {
        // decode this group's weights once into threadgroup memory (this simdgroup's own slice), shared by all RB tiles
        _Pragma("unroll") for (int nt = 0; nt < NT; nt++) {
            uint2 wv = wq[((t0 + nt) * NGRP + g) * 32 + lane];
            _Pragma("unroll") for (int ks = 0; ks < 8; ks++) {
                uint w = (ks < 4) ? wv.x : wv.y;
                const int j = ks & 3;
                if (j >= 2) w = w >> 8;
                uint bits = (j & 1) ? ((w & 0x00F000F0u) | 0x54005400u) : ((w & 0x000F000Fu) | 0x64006400u);
                half2 hv = as_type<half2>(bits);
                Bs[sg][ks * 8 + fm][nt * 8 + fn] = hv.x;
                Bs[sg][ks * 8 + fm][nt * 8 + fn + 1] = hv.y;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float s0[NT], s1[NT], b0[NT], b1[NT];
        _Pragma("unroll") for (int nt = 0; nt < NT; nt++) {
            int o = ((t0 + nt) * NGRP + g) * 8 + fn;
            s0[nt] = float(sp[o]); s1[nt] = float(sp[o + 1]); b0[nt] = float(bp[o]); b1[nt] = float(bp[o + 1]);
        }
        _Pragma("unroll") for (int rb = 0; rb < RB; rb++) {
            const device half* xt = x + (r0 + rb * 8) * K;
            simdgroup_float8x8 T[NT];
            _Pragma("unroll") for (int nt = 0; nt < NT; nt++) T[nt] = simdgroup_float8x8(0);
            _Pragma("unroll") for (int ks = 0; ks < 8; ks++) {
                simdgroup_half8x8 A;
                simdgroup_load(A, xt + g * 64 + ks * 8, K);
                _Pragma("unroll") for (int nt = 0; nt < NT; nt++) {
                    simdgroup_half8x8 Bf;
                    simdgroup_load(Bf, &Bs[sg][ks * 8][nt * 8], NT * 8);
                    simdgroup_multiply_accumulate(T[nt], A, Bf, T[nt]);
                }
            }
            int m = r0 + rb * 8 + fm;
            float xo_ = xoff[m * NGRP + g], xsum = xs[m * NGRP + g];
            _Pragma("unroll") for (int nt = 0; nt < NT; nt++) {
                C[rb][nt].thread_elements()[0] += s0[nt] * (T[nt].thread_elements()[0] - xo_) + b0[nt] * xsum;
                C[rb][nt].thread_elements()[1] += s1[nt] * (T[nt].thread_elements()[1] - xo_) + b1[nt] * xsum;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    _Pragma("unroll") for (int rb = 0; rb < RB; rb++) _Pragma("unroll") for (int nt = 0; nt < NT; nt++)
        simdgroup_store(C[rb][nt], &part[sg][rb * 8][nt * 8], NT * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(sg * 32 + lane); i < RB * 8 * NT * 8; i += SK * 32) {
        int r = i / (NT * 8), nn = i % (NT * 8);
        float v = 0.0f;
        for (int q = 0; q < SK; q++) v += part[q][r][nn];
        int m = r0 + r;
        if (m < M) {
            if (ADD) out[m * N + t0 * 8 + nn] += v;
            else out[m * N + t0 * 8 + nn] = v;
        }
    }
}
"#;

/// Per (row, head): RMSNorm of q / k heads, RoPE (non-traditional) at the row's position, q scaled; k and v written to
/// the memory at slot P + row. qkv row = [q (H*HD) | k (KVH*HD) | v (KVH*HD)]. One simdgroup per (row, head) over
/// H + KVH heads. Defines: H, KVH, HD, S, EPS, THETA, SCALE.
pub const QK_ROPE: &str = r#"
struct Args { int n; int P; };
kernel void qk_rope(const device float* qkv [[buffer(0)]], const device bfloat* qn [[buffer(1)]], const device bfloat* kn [[buffer(2)]],
                    const device int* pos [[buffer(3)]], device float* qo [[buffer(4)]],
                    device half* kc [[buffer(5)]], device half* vc [[buffer(6)]], constant Args& a [[buffer(7)]],
                    uint lane [[thread_index_in_simdgroup]], uint gid [[threadgroup_position_in_grid]]) {
    const int NHD = H + KVH;
    int row = int(gid) / NHD, hh = int(gid) % NHD;
    bool isq = hh < H;
    const int W = (H + 2 * KVH) * HD;
    const device float* src = qkv + row * W + hh * HD;
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
    for (int e = 0; e < E / 2; e++) {           // pair (i, i + HD/2), i = lane + 32e
        int i = int(lane) + 32 * e;
        float inv = pow(THETA, -2.0f * float(i) / float(HD));
        float ang = p * inv, c = cos(ang), s = sin(ang);
        float x1 = v[e], x2 = v[e + E / 2];
        o[e] = x1 * c - x2 * s;
        o[e + E / 2] = x2 * c + x1 * s;
    }
    if (isq) {
        for (int e = 0; e < E; e++) qo[(row * H + hh) * HD + int(lane) + 32 * e] = o[e] * SCALE;
    } else if (row < a.n) {
        int kh = hh - H;
        int slot = a.P + row;
        for (int e = 0; e < E; e++) kc[(kh * S + slot) * HD + int(lane) + 32 * e] = half(o[e]);
        const device float* vs = qkv + row * W + (H + KVH + kh) * HD;
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
/// Defines: H, D, NB.  Output row width H*D.
pub const ATTN_MERGE: &str = r#"
kernel void attn_merge(const device float* part [[buffer(0)]], device half* xh [[buffer(1)]],
                       device float* xo [[buffer(2)]], device float* xs [[buffer(3)]],
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
    for (int e = 0; e < E; e++) xr[int(lane) * E + e] = half(acc[e] / l);
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
