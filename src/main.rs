//! fast-rs: native Metal engine for models that fit in memory (Qwen3 dense first). No MLX, no Python.
//! One command buffer per pass; every kernel is row-independent, so guess-checking gives exactly plain decoding's words.
//! Guesser: z-lab DFlash (a block of 16 positions in one pass, reading the target's layer outputs) + best-first tree.
mod gemma;
mod kernels;
mod server;
mod st;

use gemma::{Asst, G4Cfg, G4M, gemma_layer, read_g4};

use anyhow::{Context, Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::collections::{BinaryHeap, HashMap};
use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;
use std::time::Instant;

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;

const MAXD: usize = 16; // deepest tree path / guesser block
const BS: usize = 128; // attention key block
const MAXR: usize = 128; // rows per pass (prompt chunk)
const LIN_R: usize = 4;
const DCAP: usize = 256; // DSpark: candidate tokens kept per drafted position (the 64 best and their bucket's ties) // state rows per lane in the delta-rule kernel for chains (trees: 1, so branch states fit registers)

thread_local! {
    /// seconds spent compiling source, building pipelines, and how many kernels were built (for load-time reports)
    pub static COMPILE_T: std::cell::Cell<(f64, f64, usize)> = const { std::cell::Cell::new((0.0, 0.0, 0)) };
    static PIPES: std::cell::RefCell<HashMap<String, Pipe>> = std::cell::RefCell::new(HashMap::new());
}

struct Gpu {
    dev: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: HashMap<String, Pipe>,
}

impl Gpu {
    fn new() -> Result<Self> {
        let dev = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = dev.newCommandQueue().context("queue")?;
        Ok(Gpu { dev, queue, pipes: HashMap::new() })
    }

    /// Compile `src` with `#define`s (cached by name + defines) and return its pipeline.
    fn pipe(&mut self, name: &str, src: &str, defs: &[(&str, String)]) -> Result<Pipe> {
        let key = format!("{name}|{}", defs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(","));
        if let Some(p) = self.pipes.get(&key) {
            return Ok(p.clone());
        }
        // kernels compiled for an earlier model in this process are reused (the server switches models)
        if let Some(p) = PIPES.with(|c| c.borrow().get(&key).cloned()) {
            self.pipes.insert(key, p.clone());
            return Ok(p);
        }
        let mut s = String::from(kernels::HEADER);
        for (k, v) in defs {
            s += &format!("#define {k} {v}\n");
        }
        s += kernels::GROUP_SUMS;
        s += src;
        let t0 = Instant::now();
        let opts = MTLCompileOptions::new();
        let lib = self.dev.newLibraryWithSource_options_error(&NSString::from_str(&s), Some(&opts))
            .map_err(|e| anyhow!("compile {key}: {}", e.localizedDescription()))?;
        let f = lib.newFunctionWithName(&NSString::from_str(name)).context("function")?;
        let t1 = Instant::now();
        let p = self.dev.newComputePipelineStateWithFunction_error(&f).map_err(|e| anyhow!("pipeline {key}: {}", e.localizedDescription()))?;
        COMPILE_T.with(|c| { let (a, b, n) = c.get(); c.set((a + (t1 - t0).as_secs_f64(), b + t1.elapsed().as_secs_f64(), n + 1)); });
        if std::env::var("DEBUG_PIPES").is_ok() {
            eprintln!("{key}: max threads {} tg mem {}", p.maxTotalThreadsPerThreadgroup(), p.staticThreadgroupMemoryLength());
        }
        PIPES.with(|c| c.borrow_mut().insert(key.clone(), p.clone()));
        self.pipes.insert(key, p.clone());
        Ok(p)
    }

    fn buf(&self, bytes: usize) -> Buf {
        self.dev.newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared).expect("buffer")
    }

    fn buf_from<T>(&self, data: &[T]) -> Buf {
        let b = self.buf(std::mem::size_of_val(data));
        write(&b, data);
        b
    }
}

fn write<T>(b: &Buf, data: &[T]) {
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, b.contents().as_ptr() as *mut u8, std::mem::size_of_val(data)) };
}

fn read<T: Copy>(b: &Buf, n: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const T, n).to_vec() }
}

fn bind(e: &Enc, bufs: &[&Buf]) {
    for (i, b) in bufs.iter().enumerate() {
        unsafe { e.setBuffer_offset_atIndex(Some(b), 0, i) };
    }
}

fn sz(w: usize) -> MTLSize {
    MTLSize { width: w, height: 1, depth: 1 }
}

/// A 4-bit weight matrix in the v7 lane-native layout.
struct QW {
    wq: Buf,
    wh: Option<Buf>, // 6-bit models: the top 2 bits of each weight, one u32 per (tile, group, lane)
    sp: Buf,
    bp: Buf,
    n: usize,
    k: usize,
}

const FM: [usize; 32] = { let mut a = [0; 32]; let mut l = 0; while l < 32 { let q = l / 4; a[l] = (q & 4) + ((l / 2) % 4); l += 1; } a };
const FN: [usize; 32] = { let mut a = [0; 32]; let mut l = 0; while l < 32 { let q = l / 4; a[l] = (q & 2) * 2 + (l % 2) * 2; l += 1; } a };

/// One matrix in MLX 4- or 6-bit layout (w [N, K*bits/32] u32 little-endian bitstream, s/b [N, K/64] bf16 bits),
/// borrowed or owned.
struct Q4<'a> {
    bits: usize,
    w: std::borrow::Cow<'a, [u32]>,
    s: std::borrow::Cow<'a, [u16]>,
    b: std::borrow::Cow<'a, [u16]>,
    rows: usize,
}

/// Repacked weights cached on disk, so a model loads by mapping one file instead of rebuilding every matrix.
/// The first load of a model writes the file (in the order load() repacks); later loads hand the GPU no-copy views
/// of it. File-backed pages are clean: under memory pressure macOS drops and re-reads them instead of swapping.
/// Same bytes as a fresh repack, so outputs are unchanged.
const WCACHE_VER: &str = "v7-lane-native-1";
const PAGE: usize = 16384;

enum WCache {
    Read { map: std::sync::Arc<memmap2::MmapMut>, off: usize },
    Write { f: std::io::BufWriter<std::fs::File>, off: usize, tmp: std::path::PathBuf, path: std::path::PathBuf },
}

thread_local! {
    static WCACHE: std::cell::RefCell<Option<WCache>> = const { std::cell::RefCell::new(None) };
}

fn wcache_reading() -> bool {
    WCACHE.with(|c| matches!(&*c.borrow(), Some(WCache::Read { .. })))
}

/// Cache file for (model, guesser): keyed by their files' names, sizes and dates and the layout version.
fn wcache_path(dir: &Path, guesser: Option<&Path>) -> Option<std::path::PathBuf> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    WCACHE_VER.hash(&mut h);
    for d in std::iter::once(dir).chain(guesser) {
        let mut files: Vec<_> = std::fs::read_dir(d).ok()?.filter_map(|e| e.ok()).map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false)).collect();
        files.sort();
        for f in files {
            let m = std::fs::metadata(&f).ok()?;
            (f.to_string_lossy().to_string(), m.len(), m.modified().ok()).hash(&mut h);
        }
    }
    let home = std::env::var("HOME").ok()?;
    Some(std::path::PathBuf::from(home).join(format!(".cache/fast-rs/weights-{:016x}.bin", h.finish())))
}

fn wcache_begin(dir: &Path, guesser: Option<&Path>) -> Option<std::sync::Arc<memmap2::MmapMut>> {
    if std::env::var("FAST_NO_WCACHE").is_ok() {
        return None;
    }
    let path = wcache_path(dir, guesser)?;
    if let Ok(f) = std::fs::File::open(&path) {
        // a private copy-on-write mapping: writable as Metal wants, still backed by the file
        if let Ok(map) = unsafe { memmap2::MmapOptions::new().map_copy(&f) } {
            let map = std::sync::Arc::new(map);
            WCACHE.with(|c| *c.borrow_mut() = Some(WCache::Read { map: map.clone(), off: 0 }));
            return Some(map);
        }
    }
    let _ = std::fs::create_dir_all(path.parent()?);
    let tmp = path.with_extension("tmp");
    let f = std::fs::File::create(&tmp).ok()?;
    WCACHE.with(|c| *c.borrow_mut() = Some(WCache::Write { f: std::io::BufWriter::with_capacity(1 << 22, f), off: 0, tmp, path }));
    None
}

fn wcache_end() -> Result<()> {
    match WCACHE.with(|c| c.borrow_mut().take()) {
        Some(WCache::Write { f, tmp, path, .. }) => {
            let f = f.into_inner().map_err(|e| anyhow!("weight cache: {e}"))?;
            f.sync_all()?;
            std::fs::rename(&tmp, &path)?;
        }
        Some(WCache::Read { map, off }) => anyhow::ensure!(off == map.len(), "weight cache size mismatch: rebuild by deleting it"),
        None => {}
    }
    Ok(())
}

/// Drop a half-written cache file (a load that failed part way).
fn wcache_abort() {
    if let Some(WCache::Write { tmp, .. }) = WCACHE.with(|c| c.borrow_mut().take()) {
        let _ = std::fs::remove_file(tmp);
    }
}

fn padded(n: usize) -> usize {
    n.div_ceil(PAGE) * PAGE
}

/// Several matrices stacked by rows -> v7 layout. Row tiles are independent, so they are split across all cores.
fn repack(gpu: &Gpu, parts: &[Q4], k: usize) -> QW {
    let g_n = k / 64;
    let n: usize = parts.iter().map(|p| p.rows).sum();
    let bits = parts[0].bits;
    assert!(parts.iter().all(|p| p.bits == bits));
    let mut sizes = vec![n / 8 * g_n * 32 * 2 * 4, n / 8 * g_n * 8 * 2, n / 8 * g_n * 8 * 2];
    if bits == 6 { sizes.push(n / 8 * g_n * 32 * 4); }
    let cached = WCACHE.with(|c| {
        let mut c = c.borrow_mut();
        let Some(WCache::Read { map, off }) = &mut *c else { return None };
        let mut bufs = Vec::new();
        for &sz in &sizes {
            if *off + padded(sz) > map.len() {
                return None;
            }
            let ptr = NonNull::new(unsafe { map.as_ptr().add(*off) } as *mut c_void).unwrap();
            bufs.push(unsafe { gpu.dev.newBufferWithBytesNoCopy_length_options_deallocator(ptr, padded(sz), MTLResourceOptions::StorageModeShared, None) }?);
            *off += padded(sz);
        }
        Some(bufs)
    });
    if let Some(mut b) = cached {
        let wh = if bits == 6 { b.pop() } else { None };
        let (bp, sp, wq) = (b.pop().unwrap(), b.pop().unwrap(), b.pop().unwrap());
        return QW { wq, wh, sp, bp, n, k };
    }
    let mut wq = vec![0u32; n / 8 * g_n * 32 * 2];
    let mut sp = vec![0u16; n / 8 * g_n * 8];
    let mut bp = vec![0u16; n / 8 * g_n * 8];
    let mut wh = vec![0u32; if bits == 6 { n / 8 * g_n * 32 } else { 0 }];
    // tile t -> (part, tile within part)
    let mut owner = Vec::with_capacity(n / 8);
    for (pi, part) in parts.iter().enumerate() {
        for tl in 0..part.rows / 8 { owner.push((pi, tl)); }
    }
    let rw = k * bits / 32; // u32 words per row
    // weight (row r, column c) from the little-endian bitstream
    let qat = |w: &[u32], r: usize, c: usize| -> u32 {
        let bit = r * rw * 32 + c * bits;
        let (i, sh) = (bit / 32, bit % 32);
        let v = if sh + bits > 32 { (w[i] >> sh) | (w[i + 1] << (32 - sh)) } else { w[i] >> sh };
        v & ((1 << bits) - 1)
    };
    let tile = |t: usize, wq: &mut [u32], wh: &mut [u32], sp: &mut [u16], bp: &mut [u16]| {
        let (pi, tl) = owner[t];
        let (w, s, b) = (&parts[pi].w, &parts[pi].s, &parts[pi].b);
        for g in 0..g_n {
            for lane in 0..32 {
                let (fm, fnn) = (FM[lane], FN[lane]);
                let mut hi = 0u32;
                for word in 0..2 {
                    let mut v = 0u32;
                    for half_ in 0..2 {
                        let r = tl * 8 + fnn + half_;
                        for i in 0..4 {
                            let ks = word * 4 + i;
                            let q = qat(w, r, g * 64 + ks * 8 + fm);
                            v |= (q & 0xF) << (4 * i + 16 * half_);
                            hi |= (q >> 4) << (2 * (ks * 2 + half_));
                        }
                    }
                    wq[(g * 32 + lane) * 2 + word] = v;
                }
                if bits == 6 { wh[g * 32 + lane] = hi; }
            }
            for r in 0..8 {
                sp[g * 8 + r] = s[(tl * 8 + r) * g_n + g];
                bp[g * 8 + r] = b[(tl * 8 + r) * g_n + g];
            }
        }
    };
    let tiles = n / 8;
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8).min(tiles.max(1));
    let per = tiles.div_ceil(threads);
    std::thread::scope(|sc| {
        let hn = if bits == 6 { g_n * 32 } else { 0 };
        let (mut wr, mut hr, mut sr, mut br) = (&mut wq[..], &mut wh[..], &mut sp[..], &mut bp[..]);
        let mut t0 = 0;
        while t0 < tiles {
            let cnt = per.min(tiles - t0);
            let (w1, w2) = std::mem::take(&mut wr).split_at_mut(cnt * g_n * 64);
            let (h1, h2) = std::mem::take(&mut hr).split_at_mut(cnt * hn);
            let (s1, s2) = std::mem::take(&mut sr).split_at_mut(cnt * g_n * 8);
            let (b1, b2) = std::mem::take(&mut br).split_at_mut(cnt * g_n * 8);
            (wr, hr, sr, br) = (w2, h2, s2, b2);
            let tile = &tile;
            sc.spawn(move || {
                for i in 0..cnt {
                    tile(t0 + i, &mut w1[i * g_n * 64..(i + 1) * g_n * 64], &mut h1[i * hn..(i + 1) * hn],
                         &mut s1[i * g_n * 8..(i + 1) * g_n * 8], &mut b1[i * g_n * 8..(i + 1) * g_n * 8]);
                }
            });
            t0 += cnt;
        }
    });
    WCACHE.with(|c| {
        let mut c = c.borrow_mut();
        let failed = if let Some(WCache::Write { f, off, .. }) = &mut *c {
            use std::io::Write;
            let mut put = |p: *const u8, n: usize| -> std::io::Result<()> {
                f.write_all(unsafe { std::slice::from_raw_parts(p, n) })?;
                f.write_all(&vec![0u8; padded(n) - n])?;
                *off += padded(n);
                Ok(())
            };
            put(wq.as_ptr() as *const u8, sizes[0]).and_then(|_| put(sp.as_ptr() as *const u8, sizes[1]))
                .and_then(|_| put(bp.as_ptr() as *const u8, sizes[2]))
                .and_then(|_| if bits == 6 { put(wh.as_ptr() as *const u8, sizes[3]) } else { Ok(()) }).is_err()
        } else { false };
        if failed {
            if let Some(WCache::Write { tmp, .. }) = c.take() { let _ = std::fs::remove_file(tmp); } // e.g. disk full: just don't cache
        }
    });
    let wh = if bits == 6 { Some(gpu.buf_from(&wh)) } else { None };
    QW { wq: gpu.buf_from(&wq), wh, sp: gpu.buf_from(&sp), bp: gpu.buf_from(&bp), n, k }
}

fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

fn f32_to_bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

/// bf16 matrix [rows, k] -> MLX-style affine 4-bit, group 64 (for the guesser, whose exactness does not matter).
fn quant4(src: &[u16], rows: usize, k: usize) -> Q4<'static> {
    let g_n = k / 64;
    let mut w = vec![0u32; rows * k / 8];
    let mut s = vec![0u16; rows * g_n];
    let mut b = vec![0u16; rows * g_n];
    for r in 0..rows {
        for g in 0..g_n {
            let vals: Vec<f32> = (0..64).map(|j| bf16_to_f32(src[r * k + g * 64 + j])).collect();
            let (mn, mx) = vals.iter().fold((f32::MAX, f32::MIN), |(a, c), &v| (a.min(v), c.max(v)));
            let sc = ((mx - mn) / 15.0).max(1e-8);
            let (scb, mnb) = (f32_to_bf16(sc), f32_to_bf16(mn));
            let (scr, mnr) = (bf16_to_f32(scb), bf16_to_f32(mnb));
            s[r * g_n + g] = scb;
            b[r * g_n + g] = mnb;
            for (j, &v) in vals.iter().enumerate() {
                let q = ((v - mnr) / scr).round().clamp(0.0, 15.0) as u32;
                let kk = g * 64 + j;
                w[r * (k / 8) + kk / 8] |= q << (4 * (kk % 8));
            }
        }
    }
    Q4 { bits: 4, w: w.into(), s: s.into(), b: b.into(), rows }
}

struct Layer {
    qkv: QW,
    kv: Option<QW>, // guesser only: k and v for the target-feature context
    o: QW,
    gu: QW,
    down: QW,
    in_n: Buf,
    post_n: Buf,
    qn: Buf,
    kn: Buf,
    kc: Buf,
    vc: Buf,
    lin: Option<Lin>, // Gated DeltaNet layer (Qwen3.5): qkv = [in_proj_qkv | z | b | a], o = out_proj
    at: At,           // this layer's attention shape
    g4: Option<G4L>,  // Gemma 4: sandwich norms, per-layer input, layer scalar
    sc: Option<SConv>, // LFM2 short-convolution layer: qkv = in_proj [B | C | x], o = out_proj
}

/// An LFM2 short-convolution layer's taps and its per-position cache of B*x (bf16 [slots][D]).
struct SConv {
    w: Buf,
    cache: Buf,
}

/// One attention layer's shape. Gemma 4 mixes sliding layers (head 256) with full ones (head 512) and lets later
/// layers read an earlier layer's memory (src) instead of writing their own (kvh_w = 0).
#[derive(Clone)]
struct At {
    h: usize,      // query heads
    hd: usize,     // head size
    kvh_w: usize,  // key/value heads this layer writes (0: it reads layer src's)
    kvh: usize,    // key/value heads it reads
    src: usize,    // layer whose memory it reads
    w: usize,      // sliding window (0: all)
    theta: f32,
    rot: usize,    // RoPE: dims turned, pair distance, exponent denominator (see QK_ROPE)
    ph: usize,
    ed: usize,
    scale: f32,
    vnorm: bool,
    qs: usize,     // q head stride in the projection (2*hd with an output gate)
    gate: bool,
    trad: bool,    // interleaved RoPE pairs (2j, 2j+1) instead of (i, i + hd/2)
}

impl At {
    fn qwen(c: &Cfg, i: usize) -> At {
        At { h: c.h, hd: c.hd, kvh_w: c.kvh, kvh: c.kvh, src: i, w: 0, theta: c.theta, rot: c.rot, ph: c.rot / 2, ed: c.rot,
             scale: (c.hd as f32).powf(-0.5), vnorm: false, qs: c.qs(), gate: c.gate, trad: c.trad }
    }
    fn qkvw(&self) -> usize { self.h * self.qs + 2 * self.kvh_w * self.hd }
}

/// Gemma 4 layer extras. post_n of the Layer is its pre-feedforward norm.
struct G4L {
    pa_n: Buf,          // after attention
    po_n: Buf,          // after the MLP
    ple: Option<(QW, QW, Buf)>, // per-layer input: gate (d -> ple), projection (ple -> d), norm after it
    scalar: f32,
}

/// A linear-attention layer's own weights and memory: a running state per value head instead of a key/value list.
struct Lin {
    conv: Buf,  // bf16 [C][4]
    alog: Buf,  // f32 [HV]
    dtb: Buf,   // bf16 [HV]
    norm: Buf,  // bf16 [DV]
    state: Buf, // f32 [HV][DV][DK], committed
    tail: Buf,  // f32 [3][C], last 3 raw conv inputs, committed
    keep: Buf,  // f32 [16][KS]: the last tree pass's rows (raw, final, b, a), for committing accepted ones
}

#[derive(Clone, Copy)]
struct LinCfg {
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
}

impl LinCfg {
    fn c(&self) -> usize { 2 * self.hk * self.dk + self.hv * self.dv }      // conv channels: q, k, v
    fn nlin(&self) -> usize { self.c() + self.hv * self.dv + 2 * self.hv }  // projection width: + z, b, a
    fn ks(&self) -> usize { 2 * self.c() + 2 * self.hv }                  // kept row: raw, final, b, a
}

#[derive(Clone)]
struct Cfg {
    d: usize,
    h: usize,
    kvh: usize,
    hd: usize,
    inter: usize,
    vocab: usize,
    eps: f32,
    theta: f32,
    rot: usize,            // dims RoPE turns (head_dim for Qwen3, a quarter for Qwen3.5)
    gate: bool,            // attention output gate in the q projection (Qwen3.5)
    lin: Option<LinCfg>,
    linear: Vec<bool>,     // per layer: linear attention?
    gemma: Option<G4Cfg>,  // Gemma 4
    conv: Vec<bool>,       // per layer: LFM2 short convolution?
    lfm: bool,             // LFM2 weight names (operator_norm, feed_forward.w1/w3/w2, out_proj, q/k_layernorm)
    trad: bool,            // interleaved RoPE (LFM2.5 DSpark drafters: rope_is_neox_style = false)
}

impl Cfg {
    fn qs(&self) -> usize { if self.gate { 2 * self.hd } else { self.hd } }
}

/// A stack of decoder layers with its own memory (target model, or the guesser).
struct Stack {
    cfg: Cfg,
    layers: Vec<Layer>,
    norm: Buf,
}

/// Bytes of tensors from mmapped safetensors files.
struct Weights {
    idx: HashMap<String, st::TensorInfo>,
    maps: HashMap<std::path::PathBuf, memmap2::Mmap>,
    bits: usize, // of the already-quantized matrices (config.json quantization.bits)
}

impl Weights {
    fn open(dir: &Path) -> Result<Self> {
        let idx = st::index(dir)?;
        let mut maps = HashMap::new();
        for t in idx.values() {
            if !maps.contains_key(&t.file) {
                let f = std::fs::File::open(&t.file)?;
                maps.insert(t.file.clone(), unsafe { memmap2::Mmap::map(&f)? });
            }
        }
        let bits = std::fs::read_to_string(dir.join("config.json")).ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|c| c["quantization"]["bits"].as_u64()).unwrap_or(4) as usize;
        Ok(Weights { idx, maps, bits })
    }
    fn raw(&self, name: &str) -> Result<&[u8]> {
        let t = self.idx.get(name).with_context(|| format!("missing {name}"))?;
        Ok(&self.maps[&t.file][t.offset as usize..t.offset as usize + t.len])
    }
    fn u32s(&self, name: &str) -> Result<&[u32]> {
        let b = self.raw(name)?;
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u32, b.len() / 4) })
    }
    fn f32s(&self, name: &str) -> Result<&[f32]> {
        let b = self.raw(name)?;
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) })
    }
    fn u16s(&self, name: &str) -> Result<&[u16]> {
        let b = self.raw(name)?;
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) })
    }
    fn shape(&self, name: &str) -> Vec<usize> {
        self.idx[name].shape.iter().map(|&x| x as usize).collect()
    }
    /// A linear layer as Q4: already 4-bit (MLX), or bf16 quantized here.
    fn q4(&self, name: &str) -> Result<Q4<'_>> {
        if wcache_reading() {
            // only the row count is needed: repack() takes the matrix from the cache file
            let bits = if self.idx.contains_key(&format!("{name}.scales")) { self.bits } else { 4 };
            return Ok(Q4 { bits, w: Vec::new().into(), s: Vec::new().into(), b: Vec::new().into(), rows: self.shape(&format!("{name}.weight"))[0] });
        }
        if self.idx.contains_key(&format!("{name}.scales")) {
            Ok(Q4 { bits: self.bits, w: self.u32s(&format!("{name}.weight"))?.into(), s: self.u16s(&format!("{name}.scales"))?.into(),
                    b: self.u16s(&format!("{name}.biases"))?.into(), rows: self.shape(&format!("{name}.weight"))[0] })
        } else {
            let sh = self.shape(&format!("{name}.weight"));
            Ok(quant4(self.u16s(&format!("{name}.weight"))?, sh[0], sh[1]))
        }
    }
}

fn lin_defs(c: &Cfg) -> Vec<(&'static str, String)> {
    let l = c.lin.unwrap();
    [("C", l.c()), ("KD", l.hk * l.dk), ("DK", l.dk), ("DV", l.dv), ("VD", l.hv * l.dv), ("HV", l.hv), ("HK", l.hk), ("NLIN", l.nlin()), ("KS", l.ks())]
        .into_iter().map(|(k, v)| (k, v.to_string())).chain([("EPS", format!("{:e}f", c.eps))]).collect()
}

/// Delta-rule pipeline defines: R state rows per lane (same arithmetic per state element for any R).
fn delta_defs(c: &Cfg, r: usize) -> Vec<(&'static str, String)> {
    let mut d = lin_defs(c);
    d.push(("R", r.to_string()));
    d
}

fn set_args(e: &Enc, a: &[i32], idx: usize) {
    unsafe { e.setBytes_length_atIndex(NonNull::new(a.as_ptr() as *mut c_void).unwrap(), a.len() * 4, idx) };
}

fn zero(b: &Buf) {
    unsafe { std::ptr::write_bytes(b.contents().as_ptr() as *mut u8, 0, b.length()) };
}

fn merge_defs(a: &At, nb: usize) -> Vec<(&'static str, String)> {
    vec![("H", a.h.to_string()), ("D", a.hd.to_string()), ("NB", nb.to_string()), ("GATE", (a.gate as i32).to_string()),
         ("QS", a.qs.to_string()), ("QW", a.qkvw().to_string())]
}

fn split_defs(a: &At, maxlen: usize, nb: usize) -> Vec<(&'static str, String)> {
    vec![("H", a.h.to_string()), ("KVH", a.kvh.to_string()), ("D", a.hd.to_string()), ("S", maxlen.to_string()), ("W", a.w.to_string()),
         ("MAXD", MAXD.to_string()), ("BS", BS.to_string()), ("NB", nb.to_string())]
}

fn read_cfg(c: &serde_json::Value) -> Cfg {
    let c = if c["text_config"].is_object() { &c["text_config"] } else { c };
    let f = |k: &str| c[k].as_f64().unwrap_or(0.0);
    let h = f("num_attention_heads") as usize;
    let d = f("hidden_size") as usize;
    let rp = &c["rope_parameters"];
    let hd = c["head_dim"].as_u64().map(|x| x as usize).unwrap_or(d / h);
    let n = f("num_hidden_layers") as usize;
    let linear: Vec<bool> = (0..n).map(|i| c["layer_types"][i].as_str() == Some("linear_attention")).collect();
    let conv: Vec<bool> = (0..n).map(|i| c["layer_types"][i].as_str() == Some("conv")).collect();
    Cfg {
        d,
        h,
        kvh: f("num_key_value_heads") as usize,
        hd,
        inter: f("intermediate_size") as usize,
        vocab: f("vocab_size") as usize,
        eps: c["rms_norm_eps"].as_f64().or(c["norm_eps"].as_f64()).unwrap_or(1e-6) as f32,
        theta: c["rope_theta"].as_f64().or(rp["rope_theta"].as_f64()).unwrap_or(1e6) as f32,
        rot: (hd as f64 * rp["partial_rotary_factor"].as_f64().or(c["partial_rotary_factor"].as_f64()).unwrap_or(1.0)) as usize,
        gate: c["attn_output_gate"].as_bool().unwrap_or(false),
        lin: linear.iter().any(|&x| x).then(|| LinCfg { hk: f("linear_num_key_heads") as usize, hv: f("linear_num_value_heads") as usize,
            dk: f("linear_key_head_dim") as usize, dv: f("linear_value_head_dim") as usize }),
        linear,
        gemma: read_g4(c),
        lfm: c["model_type"].as_str() == Some("lfm2"),
        trad: c["rope_is_neox_style"].as_bool() == Some(false) && std::env::var("FORCE_NEOX").is_err(),
        conv,
    }
}

fn load_stack(gpu: &Gpu, w: &Weights, cfg: &Cfg, n_layers: usize, pre: &str, norm: &str, maxlen: usize, guesser: bool) -> Result<Stack> {
    let mut layers = Vec::new();
    for i in 0..n_layers {
        let p = format!("{pre}layers.{i}.");
        let a = format!("{p}self_attn.");
        let m = format!("{p}mlp.");
        let kvbytes = cfg.kvh * maxlen * cfg.hd * 2;
        if cfg.linear.get(i) == Some(&true) {
            let lc = cfg.lin.unwrap();
            let la = format!("{p}linear_attn.");
            layers.push(Layer {
                qkv: repack(gpu, &[w.q4(&format!("{la}in_proj_qkv"))?, w.q4(&format!("{la}in_proj_z"))?, w.q4(&format!("{la}in_proj_b"))?,
                                   w.q4(&format!("{la}in_proj_a"))?], cfg.d),
                kv: None,
                o: repack(gpu, &[w.q4(&format!("{la}out_proj"))?], lc.hv * lc.dv),
                gu: repack(gpu, &[w.q4(&format!("{m}gate_proj"))?, w.q4(&format!("{m}up_proj"))?], cfg.d),
                down: repack(gpu, &[w.q4(&format!("{m}down_proj"))?], cfg.inter),
                in_n: gpu.buf_from(w.u16s(&format!("{p}input_layernorm.weight"))?),
                post_n: gpu.buf_from(w.u16s(&format!("{p}post_attention_layernorm.weight"))?),
                qn: gpu.buf(16),
                kn: gpu.buf(16),
                kc: gpu.buf(16),
                vc: gpu.buf(16),
                lin: Some(Lin {
                    conv: gpu.buf_from(w.u16s(&format!("{la}conv1d.weight"))?),
                    alog: gpu.buf_from(w.f32s(&format!("{la}A_log"))?),
                    dtb: gpu.buf_from(w.u16s(&format!("{la}dt_bias"))?),
                    norm: gpu.buf_from(w.u16s(&format!("{la}norm.weight"))?),
                    state: gpu.buf(lc.hv * lc.dv * lc.dk * 4),
                    tail: gpu.buf(3 * lc.c() * 4),
                    keep: gpu.buf(MAXD * lc.ks() * 4),
                }),
                at: At::qwen(cfg, i),
                g4: None,
                sc: None,
            });
            continue;
        }
        if cfg.gemma.is_some() {
            layers.push(gemma_layer(gpu, w, cfg, i, &p, maxlen)?);
            continue;
        }
        if cfg.lfm {
            let f = format!("{p}feed_forward.");
            let conv = cfg.conv.get(i) == Some(&true);
            let mut at = At::qwen(cfg, i);
            let (qkv, o, qn, kn, kc, vc, sc) = if conv {
                at.kvh_w = 0; // no keys or values: commit and attention skip it
                let cv = format!("{p}conv.");
                (repack(gpu, &[w.q4(&format!("{cv}in_proj"))?], cfg.d), repack(gpu, &[w.q4(&format!("{cv}out_proj"))?], cfg.d),
                 gpu.buf(16), gpu.buf(16), gpu.buf(16), gpu.buf(16),
                 Some(SConv { w: gpu.buf_from(w.u16s(&format!("{cv}conv.weight"))?), cache: gpu.buf(maxlen * cfg.d * 2) }))
            } else {
                (repack(gpu, &[w.q4(&format!("{a}q_proj"))?, w.q4(&format!("{a}k_proj"))?, w.q4(&format!("{a}v_proj"))?], cfg.d),
                 repack(gpu, &[w.q4(&format!("{a}out_proj"))?], cfg.h * cfg.hd),
                 gpu.buf_from(w.u16s(&format!("{a}q_layernorm.weight"))?), gpu.buf_from(w.u16s(&format!("{a}k_layernorm.weight"))?),
                 gpu.buf(kvbytes), gpu.buf(kvbytes), None)
            };
            layers.push(Layer {
                qkv, kv: None, o,
                gu: repack(gpu, &[w.q4(&format!("{f}w1"))?, w.q4(&format!("{f}w3"))?], cfg.d),
                down: repack(gpu, &[w.q4(&format!("{f}w2"))?], cfg.inter),
                in_n: gpu.buf_from(w.u16s(&format!("{p}operator_norm.weight"))?),
                post_n: gpu.buf_from(w.u16s(&format!("{p}ffn_norm.weight"))?),
                qn, kn, kc, vc, lin: None, at, g4: None, sc,
            });
            continue;
        }
        layers.push(Layer {
            qkv: repack(gpu, &[w.q4(&format!("{a}q_proj"))?, w.q4(&format!("{a}k_proj"))?, w.q4(&format!("{a}v_proj"))?], cfg.d),
            kv: if guesser { Some(repack(gpu, &[w.q4(&format!("{a}k_proj"))?, w.q4(&format!("{a}v_proj"))?], cfg.d)) } else { None },
            o: repack(gpu, &[w.q4(&format!("{a}o_proj"))?], cfg.h * cfg.hd),
            gu: repack(gpu, &[w.q4(&format!("{m}gate_proj"))?, w.q4(&format!("{m}up_proj"))?], cfg.d),
            down: repack(gpu, &[w.q4(&format!("{m}down_proj"))?], cfg.inter),
            in_n: gpu.buf_from(w.u16s(&format!("{p}input_layernorm.weight"))?),
            post_n: gpu.buf_from(w.u16s(&format!("{p}post_attention_layernorm.weight"))?),
            qn: gpu.buf_from(w.u16s(&format!("{a}q_norm.weight"))?),
            kn: gpu.buf_from(w.u16s(&format!("{a}k_norm.weight"))?),
            kc: gpu.buf(kvbytes),
            vc: gpu.buf(kvbytes),
            lin: None,
            at: At::qwen(cfg, i),
            g4: None,
            sc: None,
        });
    }
    Ok(Stack { cfg: cfg.clone(), layers, norm: gpu.buf_from(w.u16s(norm)?) })
}

struct Guesser {
    stack: Stack,
    fc: QW,
    hidden_norm: Buf,
    target_layers: Vec<usize>,
    mask: i32,
    block: usize,
    run_block: usize,        // block drafted per pass (≤ block): fewer positions cost less and a tree of 8 rarely uses more
    c: usize,                // context words already in the guesser's memory
    pending: Vec<f32>,       // target features of committed words not yet in its memory [rows, F]
    markov: Option<Weights>, // DSpark: low-rank bigram bias w1[prev] . w2[next] added to each drafted position
    from_anchor: bool,       // DSpark: block row 0 (the anchor) predicts the first new word; DFlash: row 1 does
}

struct Model {
    gpu: Gpu,
    t: Stack,
    g: Option<Guesser>,
    maxlen: usize,
    emb: Weights, // the word table stays in the file; each pass copies only its rows (saves ~0.35 GB of GPU memory on 8B)
    emb_w: Buf,
    emb_s: Buf,
    emb_b: Buf,
    emb_ids: Buf,
    epre: String,   // name of the word table in the checkpoint
    lin_prev: Buf,  // [rows][3]: each row's previous 3 conv inputs (row of this pass, or committed tail -1..-3)
    lin_par: Buf,   // [rows]: row whose state each row continues (-1: committed)
    lin_acc: Buf,   // accepted tree rows, for commit
    lin_chain: Buf, // -1, 0, 1, ...: a chain's parents
    lin_gb: Buf,    // [rows][2*HV]: per row and value head, decay g and write strength beta
    lin_tree: bool, // the next pass is a guess tree: linear layers keep their committed state until commit()
    head: QW,
    ids: Buf,
    pos: Buf,
    rows: Buf,
    anc: Buf,
    h: Buf,
    xh: Buf,
    xo: Buf,
    xs: Buf,
    qkv: Buf,
    q: Buf,
    part: Buf,
    gu: Buf,
    feats: Buf,
    logits: Buf,
    pick: Buf,
    tv: Buf,
    ti: Buf,
    cv: Buf,                      // DSpark: per draft row, candidate logits / token ids / count (TOPKC)
    ci: Buf,
    cn: Buf,
    dcand: Vec<Vec<(f32, i32)>>,  // the last draft's candidates per position (DSpark Markov chain)
    p: usize,
    hist: Vec<i32>,
    pub draft_s: f64,
    commit_s: f64,
    gpu_draft: f64, // GPU-clock busy time per generate(): draft and verify command buffers
    gpu_verify: f64,
    tune: HashMap<String, (usize, usize)>,
    want_depth: usize,   // deepest tree level the next guess needs (budget - 1)
    pub reuse: bool,     // keep the memory of words a new prompt shares with the previous conversation
    pub turn_tok: Option<i32>, // chat turn marker (<|im_start|>): hybrid models snapshot their state after the last one
    snap: Option<(usize, Vec<Vec<u8>>)>, // (position, linear layers' state + tail) at the last turn boundary
    pub kept: usize,     // words of the last prompt whose memory was reused
    g4m: Option<G4M>,    // Gemma 4 per-layer inputs
    asst: Option<Asst>,  // Gemma 4 assistant (guesser)
    hrows: Vec<f32>,     // last pass's final hidden rows (before the norm), for the assistant
    last_hn: Vec<f32>,   // normed hidden of the last committed word: the assistant's starting point
    tmp: Buf,            // [rows][d] scratch (Gemma: block outputs before their norm)
    tmp2: Buf,           // [rows][ple]
    rscale: Buf,         // [rows]: power-of-two input scale of the last prep (rows past half's range)
    rs_next: bool,       // the next matmul multiplies its rows back by rscale
    pub prof: [f64; 4],  // assistant: GPU wait s, CPU head s, rounds
    wmap: Option<std::sync::Arc<memmap2::MmapMut>>, // weight cache mapping; declared last so the GPU buffers go first
}

impl Model {
    pub(crate) fn gpu_bytes(&self) -> usize {
        self.gpu.dev.currentAllocatedSize()
    }

    fn load(dir: &Path, guesser: Option<&Path>, maxlen: usize) -> Result<Self> {
        let wmap = wcache_begin(dir, guesser);
        let r = Self::load_inner(dir, guesser, maxlen);
        match r {
            Ok(mut m) => {
                if let Err(e) = wcache_end() { eprintln!("{e}"); }
                m.wmap = wmap;
                Ok(m)
            }
            Err(e) => {
                wcache_abort();
                Err(e)
            }
        }
    }

    fn load_inner(dir: &Path, guesser: Option<&Path>, maxlen: usize) -> Result<Self> {
        let gpu = Gpu::new()?;
        let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        if !matches!(c["quantization"]["bits"].as_i64(), Some(4 | 6)) || c["quantization"]["group_size"].as_i64() != Some(64) {
            bail!("only MLX 4- or 6-bit group-64 models for now");
        }
        let cfg = read_cfg(&c);
        let tc = if c["text_config"].is_object() { &c["text_config"] } else { &c };
        let w = Weights::open(dir)?;
        // Qwen3.5 checkpoints carry the text model under language_model.
        let lm = if w.idx.contains_key("language_model.model.embed_tokens.weight") { "language_model." } else { "" };
        let epre = format!("{lm}model.embed_tokens");
        let t0 = Instant::now();
        let fnorm = if cfg.lfm { format!("{lm}model.embedding_norm.weight") } else { format!("{lm}model.norm.weight") };
        let t = load_stack(&gpu, &w, &cfg, cfg.linear.len(), &format!("{lm}model."), &fnorm, maxlen, false)?;
        let tie = tc["tie_word_embeddings"].as_bool().or(c["tie_word_embeddings"].as_bool()).unwrap_or(false);
        let head = repack(&gpu, &[if tie { w.q4(&epre)? } else { w.q4(&format!("{lm}lm_head"))? }], cfg.d);
        let g4m = if cfg.gemma.is_some() { Some(G4M::load(&gpu, &w, lm, &cfg, MAXR)?) } else { None };
        let gconf: Option<serde_json::Value> = match guesser {
            Some(gd) => Some(serde_json::from_str(&std::fs::read_to_string(gd.join("config.json"))?)?),
            None => None,
        };
        let is_asst = gconf.as_ref().map(|c| c["model_type"].as_str() == Some("gemma4_assistant")).unwrap_or(false);
        let asst = if is_asst { Some(Asst::load(&gpu, guesser.unwrap(), &t)?) } else { None };
        let g = match guesser.filter(|_| !is_asst) {
            Some(gd) => {
                let gc = gconf.clone().unwrap();
                let gcfg = read_cfg(&gc);
                let gw = Weights::open(gd)?;
                let tl: Vec<usize> = gc["dflash_config"]["target_layer_ids"].as_array().context("target_layer_ids")?
                    .iter().map(|x| x.as_u64().unwrap() as usize).collect();
                let stack = load_stack(&gpu, &gw, &gcfg, gc["num_hidden_layers"].as_u64().unwrap() as usize, "", "norm.weight", maxlen, true)?;
                let has_markov = gw.idx.contains_key("markov_head.markov_w1.weight");
                let block = gc["block_size"].as_u64().or(gc["dflash_config"]["block_size"].as_u64()).context("block_size")? as usize;
                // Qwen3.5-4B + z-lab DFlash: block 8 keeps acceptance and drafts in ~2/3 the time (Spec-Bench dev 60:
                // 1.18-1.33x, outputs identical); Qwen3-4B loses acceptance at 8, so it keeps the trained block.
                let run_block = if c["model_type"].as_str() == Some("qwen3_5") && !has_markov { block.min(8) } else { block };
                Some(Guesser {
                    fc: repack(&gpu, &[gw.q4("fc")?], tl.len() * gcfg.d),
                    hidden_norm: gpu.buf_from(gw.u16s("hidden_norm.weight")?),
                    mask: gc["dflash_config"]["mask_token_id"].as_i64().unwrap() as i32,
                    block,
                    run_block,
                    target_layers: tl,
                    stack,
                    c: 0,
                    pending: Vec::new(),
                    markov: if has_markov { Some(gw) } else { None },
                    from_anchor: gc["sample_from_anchor"].as_bool().unwrap_or(gc["architectures"][0].as_str().unwrap_or("").contains("DSpark")),
                })
            }
            None => None,
        };
        eprintln!("weights repacked in {:.1}s", t0.elapsed().as_secs_f64());
        let nbmax = (maxlen + MAXD) / BS + 1;
        let r = MAXR;
        let nf = g.as_ref().map(|g| g.target_layers.len()).unwrap_or(1);
        let lc = cfg.lin.unwrap_or(LinCfg { hk: 0, hv: 0, dk: 0, dv: 0 });
        // scratch sizes: the widest of every layer kind (target, guesser stack, assistant)
        let mut all: Vec<&Layer> = t.layers.iter().filter(|l| l.lin.is_none()).collect();
        if let Some(g) = &g { all.extend(g.stack.layers.iter()); }
        if let Some(a) = &asst { all.extend(a.layers.iter()); }
        let qkvw = all.iter().map(|l| l.qkv.n).max().unwrap_or(0).max(lc.c());
        let qw = all.iter().map(|l| l.at.h * l.at.hd).max().unwrap_or(0).max(lc.hv * lc.dv);
        let ple = cfg.gemma.as_ref().map(|g| g.ple).unwrap_or(0);
        let xw = cfg.d.max(cfg.inter).max(qw).max(nf * cfg.d).max(all.iter().map(|l| l.down.k.max(l.o.k)).max().unwrap_or(0))
            .max(2 * cfg.d).max(ple.max(64));
        let guw = all.iter().map(|l| l.gu.n).max().unwrap_or(0).max(lc.nlin());
        let partw = all.iter().map(|l| l.at.h * (l.at.hd + 2)).max().unwrap_or(0);
        let (tmp, tmp2, rsb) = (gpu.buf(r * cfg.d * 4), gpu.buf(r * ple.max(64) * 4), gpu.buf(r * 4));
        Ok(Model {
            emb_w: gpu.buf(r * cfg.d * w.bits / 8),
            emb_s: gpu.buf(r * cfg.d / 64 * 2),
            emb_b: gpu.buf(r * cfg.d / 64 * 2),
            emb_ids: gpu.buf_from(&(0..r as i32).collect::<Vec<_>>()),
            emb: w,
            epre,
            lin_prev: gpu.buf(r * 3 * 4),
            lin_par: gpu.buf(r * 4),
            lin_acc: gpu.buf(MAXD * 4),
            lin_chain: gpu.buf_from(&(-1..r as i32 - 1).collect::<Vec<_>>()),
            lin_gb: gpu.buf(r * 2 * lc.hv.max(1) * 4),
            lin_tree: false,
            head,
            ids: gpu.buf(r * 4),
            pos: gpu.buf(r * 4),
            rows: gpu.buf(r * 16),
            anc: gpu.buf(r * MAXD * 4),
            h: gpu.buf(r * cfg.d * 4),
            xh: gpu.buf(r * xw * 2),
            xo: gpu.buf(r * xw / 64 * 4),
            xs: gpu.buf(r * xw / 64 * 4),
            qkv: gpu.buf(r * qkvw * 4),
            q: gpu.buf(r * qw * 4),
            part: gpu.buf(r * nbmax * partw * 4),
            gu: gpu.buf(r * guw * 4),
            feats: gpu.buf(r * nf * cfg.d * 4),
            logits: gpu.buf(16 * cfg.vocab * 4),
            pick: gpu.buf(16 * 4),
            tv: gpu.buf(16 * 3 * 4),
            ti: gpu.buf(16 * 3 * 4),
            cv: gpu.buf(16 * DCAP * 4),
            ci: gpu.buf(16 * DCAP * 4),
            cn: gpu.buf(16 * 4),
            dcand: Vec::new(),
            t,
            g,
            maxlen,
            gpu,
            p: 0,
            hist: Vec::new(),
            draft_s: 0.0,
            commit_s: 0.0,
            gpu_draft: 0.0,
            gpu_verify: 0.0,
            tune: HashMap::new(),
            want_depth: 15,
            reuse: false,
            turn_tok: None,
            snap: None,
            kept: 0,
            g4m,
            asst,
            hrows: Vec::new(),
            last_hn: Vec::new(),
            tmp,
            tmp2,
            rscale: rsb,
            rs_next: false,
            prof: [0.0; 4],
            wmap: None,
        })
    }

    /// Time every (NT, SK) layout for every matrix shape at 8 and 16 rows on this GPU; cache the winners on disk.
    fn autotune(&mut self) -> Result<()> {
        let dev = self.gpu.dev.name().to_string().replace(' ', "_");
        let path = std::path::PathBuf::from(std::env::var("HOME")?).join(format!(".cache/fast-rs/tune-{dev}.json"));
        if let Ok(s) = std::fs::read_to_string(&path) {
            let v: HashMap<String, (usize, usize)> = serde_json::from_str(&s)?;
            self.tune.extend(v);
        }
        let mut mats: Vec<*const QW> = Vec::new();
        let l0 = &self.t.layers[0];
        mats.extend([&l0.qkv as *const QW, &l0.o, &l0.gu, &l0.down, &self.head]);
        if let Some(fa) = self.t.layers.iter().find(|l| l.lin.is_none() != l0.lin.is_none()) {
            mats.extend([&fa.qkv as *const QW, &fa.o]); // hybrid models: the other layer kind's shapes too
        }
        if let Some(g) = &self.g {
            let gl = &g.stack.layers[0];
            mats.extend([&gl.qkv as *const QW, &gl.o, &gl.gu, &gl.down, gl.kv.as_ref().unwrap(), &g.fc]);
        }
        let mut changed = false;
        let t0 = Instant::now();
        for &wp in &mats {
            let w = unsafe { &*wp };
            for mt in [1usize, 2] {
                let key = format!("{}x{}x{mt}", w.n, w.k);
                if self.tune.contains_key(&key) {
                    continue;
                }
                let (xh, xo, xs, out) = (self.xh.clone(), self.xo.clone(), self.xs.clone(), self.logits.clone());
                let mut cands = Vec::new();
                for nt in [1usize, 2, 4] {
                    for sk in [1usize, 2, 4, 8] {
                        if w.n % (8 * nt) == 0 && sk <= w.k / 64 && sk * 8 * nt * 8 * 4 <= 32768 {
                            cands.push((f64::MAX, nt, sk));
                        }
                    }
                }
                // interleaved rounds, best time per candidate (the machine is noisy)
                for _ in 0..5 {
                    for c in cands.iter_mut() {
                        let cb = self.gpu.queue.commandBuffer().context("cb")?;
                        let e = cb.computeCommandEncoder().context("enc")?;
                        for _ in 0..10 {
                            self.enc_qmm_with(&e, w, [&xh, &xo, &xs], &out, mt * 8, false, c.1, c.2)?;
                        }
                        e.endEncoding();
                        cb.commit();
                        cb.waitUntilCompleted();
                        c.0 = c.0.min(cb.GPUEndTime() - cb.GPUStartTime());
                    }
                }
                // keep the default layout unless a candidate beats it by >= 10% in the same rounds
                self.tune.remove(&key);
                let (dn, ds) = self.cfg_for(w.n, w.k, mt);
                let base = cands.iter().find(|c| c.1 == dn && c.2 == ds).map(|c| c.0).unwrap_or(f64::MAX);
                let mut best = cands.iter().cloned().fold((f64::MAX, 2, 1), |a, c| if c.0 < a.0 { c } else { a });
                if best.0 > 0.9 * base {
                    best = (base, dn, ds);
                }
                self.tune.insert(key, (best.1, best.2));
                changed = true;
            }
        }
        if changed {
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(&path, serde_json::to_string(&self.tune)?)?;
            eprintln!("tuned {} shapes in {:.1}s -> {}", self.tune.len(), t0.elapsed().as_secs_f64(), path.display());
        }
        Ok(())
    }

    /// Compile every kernel shape a run can hit (all row counts, all attention block counts) so no compile lands in
    /// a timed region.
    fn warm(&mut self) -> Result<()> {
        let t0 = Instant::now();
        let mut mats: Vec<*const QW> = Vec::new();
        let l0 = &self.t.layers[0];
        mats.extend([&l0.qkv as *const QW, &l0.o, &l0.gu, &l0.down, &self.head]);
        if let Some(fa) = self.t.layers.iter().find(|l| l.lin.is_none() != l0.lin.is_none()) {
            mats.extend([&fa.qkv as *const QW, &fa.o]); // hybrid models: the other layer kind's shapes too
        }
        if let Some(g) = &self.g {
            let gl = &g.stack.layers[0];
            mats.extend([&gl.qkv as *const QW, &gl.o, &gl.gu, &gl.down, gl.kv.as_ref().unwrap(), &g.fc]);
        }
        for &wp in &mats {
            let w = unsafe { &*wp };
            for rows in (8..=MAXR).step_by(8) {
                for add in [false, true] {
                    let (nt, sk) = self.cfg_for(w.n, w.k, rows / 8);
                    self.gpu.pipe("qmm", kernels::QMM, &[("M", rows.to_string()), ("MT", (rows / 8).to_string()), ("K", w.k.to_string()),
                        ("N", w.n.to_string()), ("NT", nt.to_string()), ("SK", sk.to_string()), ("ADD", (add as i32).to_string()), ("RS", "0".into()),
                        ("BITS", if w.wh.is_some() { "6" } else { "4" }.into())])?;
                }
            }
        }
        let ats: Vec<At> = self.t.layers.iter().filter(|l| l.lin.is_none()).map(|l| l.at.clone()).collect();
        // short conversations are ready at once; longer ones compile their attention variant on first use
        for nb in 1..=(self.maxlen + MAXD).div_ceil(BS).min(2048 / BS + 1) {
            for a in &ats {
                self.gpu.pipe("attn_split", kernels::ATTN_SPLIT, &split_defs(a, self.maxlen, nb))?;
                self.gpu.pipe("attn_merge", kernels::ATTN_MERGE, &merge_defs(a, nb))?;
            }
        }
        eprintln!("compiled {} kernel variants in {:.1}s", self.gpu.pipes.len(), t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Words at the start of `prompt` whose memory is already in place from the previous call (at least one word is
    /// left to read, for its next-word pick).
    fn shared_prefix(&self, prompt: &[i32]) -> usize {
        let n = self.hist.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        n.min(prompt.len().saturating_sub(1))
    }

    /// Keep the first `keep` words of memory (target and guesser), forget the rest.
    fn rewind(&mut self, keep: usize) {
        if keep == 0 {
            return self.reset();
        }
        self.p = keep;
        self.hist.truncate(keep);
        let f = self.g.as_ref().map(|g| g.target_layers.len() * self.t.cfg.d).unwrap_or(0);
        if let Some(g) = &mut self.g {
            if g.c > keep {
                g.c = keep;
                g.pending.clear();
            } else {
                g.pending.truncate((keep - g.c) * f);
            }
        }
    }

    fn reset(&mut self) {
        self.p = 0;
        self.hist.clear();
        self.snap = None;
        for l in &self.t.layers {
            if let Some(lin) = &l.lin {
                zero(&lin.state);
                zero(&lin.tail);
            }
        }
        if let Some(g) = &mut self.g {
            g.c = 0;
            g.pending.clear();
        }
    }

    fn cfg_for(&self, n: usize, k: usize, mt: usize) -> (usize, usize) {
        // one layout per matrix for every row count, so a word's numbers never depend on how many rows shared its
        // pass (prompt chunk, guess tree or single word): what makes reusing an earlier turn's memory exact
        let _ = mt;
        if let Some(&c) = self.tune.get(&format!("{n}x{k}x1")) {
            return c;
        }
        let nt = if n % 32 == 0 { 2 } else { 1 };
        let tiles = n / (8 * nt);
        let mut sk = if tiles >= 512 { 1 } else if tiles >= 128 { 2 } else { 4 };
        sk = sk.min(k / 64);
        while sk > 1 && sk * 8 * nt * 8 * 4 > 32768 {
            sk /= 2;
        }
        (nt, sk)
    }

    fn enc_qmm(&mut self, e: &Enc, w: &QW, x: [&Buf; 3], out: &Buf, rows: usize, add: bool) -> Result<()> {
        let mt = rows / 8;
        let (nt, sk) = self.cfg_for(w.n, w.k, mt);
        self.enc_qmm_with(e, w, x, out, rows, add, nt, sk)
    }

    fn enc_qmm_with(&mut self, e: &Enc, w: &QW, x: [&Buf; 3], out: &Buf, rows: usize, add: bool, nt: usize, sk: usize) -> Result<()> {
        self.enc_qmm_at(e, w, x, 0, out, rows, add, nt, sk)
    }

    /// As enc_qmm_with, reading x from row `r0` on.
    fn enc_qmm_at(&mut self, e: &Enc, w: &QW, x: [&Buf; 3], r0: usize, out: &Buf, rows: usize, add: bool, nt: usize, sk: usize) -> Result<()> {
        if rows > 128 {
            bail!("qmm rows > 128");
        }
        let mt = rows / 8;
        let rs = std::mem::take(&mut self.rs_next);
        let p = self.gpu.pipe("qmm", kernels::QMM, &[("M", rows.to_string()), ("MT", mt.to_string()), ("K", w.k.to_string()),
            ("N", w.n.to_string()), ("NT", nt.to_string()), ("SK", sk.to_string()), ("ADD", (add as i32).to_string()), ("RS", (rs as i32).to_string()),
            ("BITS", if w.wh.is_some() { "6" } else { "4" }.into())])?;
        e.setComputePipelineState(&p);
        bind(e, &[x[0], x[1], x[2], &w.wq, &w.sp, &w.bp, out, &self.rscale, w.wh.as_ref().unwrap_or(&w.wq)]);
        if r0 > 0 {
            unsafe {
                e.setBuffer_offset_atIndex(Some(x[0]), r0 * w.k * 2, 0);
                e.setBuffer_offset_atIndex(Some(x[1]), r0 * (w.k / 64) * 4, 1);
                e.setBuffer_offset_atIndex(Some(x[2]), r0 * (w.k / 64) * 4, 2);
            }
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(w.n / (8 * nt)), sz(32 * sk));
        Ok(())
    }

    fn enc_norm(&mut self, e: &Enc, d: usize, eps: f32, h: &Buf, w: &Buf, n: usize) -> Result<()> {
        let p = self.gpu.pipe("norm_prep", kernels::NORM_PREP, &[("D", d.to_string()), ("EPS", format!("{eps:e}f"))])?;
        e.setComputePipelineState(&p);
        let (xh, xo, xs) = (self.xh.clone(), self.xo.clone(), self.xs.clone());
        bind(e, &[h, w, &xh, &xo, &xs]);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n), sz(256));
        Ok(())
    }

    fn enc_rope(&mut self, e: &Enc, c: &Cfg, heads: usize, src: &Buf, layer: &Layer, n_real: usize, p0: usize, n: usize, pos: &Buf) -> Result<()> {
        let a = &layer.at;
        let p = self.gpu.pipe("qk_rope", kernels::QK_ROPE, &[("H", heads.to_string()), ("KVH", a.kvh_w.to_string()), ("HD", a.hd.to_string()),
            ("QS", a.qs.to_string()), ("ROT", a.rot.to_string()), ("PH", a.ph.to_string()), ("ED", a.ed.to_string()), ("VNORM", (a.vnorm as i32).to_string()), ("TRAD", (a.trad as i32).to_string()),
            ("S", self.maxlen.to_string()), ("EPS", format!("{:e}f", c.eps)), ("THETA", format!("{:e}f", a.theta)),
            ("SCALE", format!("{:e}f", a.scale))])?;
        e.setComputePipelineState(&p);
        let q = self.q.clone();
        bind(e, &[src, &layer.qn, &layer.kn, pos, &q, &layer.kc, &layer.vc]);
        let args: [i32; 2] = [n_real as i32, p0 as i32];
        unsafe { e.setBytes_length_atIndex(NonNull::new(args.as_ptr() as *mut c_void).unwrap(), 8, 7) };
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n * (heads + a.kvh_w)), sz(32));
        Ok(())
    }

    fn enc_embed(&mut self, e: &Enc, d: usize, n: usize) -> Result<()> {
        let escale = self.t.cfg.gemma.as_ref().map(|_| (d as f32).sqrt()).unwrap_or(1.0);
        let p = self.gpu.pipe("embed", kernels::EMBED, &[("D", d.to_string()), ("ESCALE", format!("{escale:e}f")), ("BITS", self.emb.bits.to_string())])?;
        e.setComputePipelineState(&p);
        let ids: Vec<i32> = read(&self.ids, n);
        let ep = &self.epre;
        let (tw, ts, tb) = (self.emb.u32s(&format!("{ep}.weight"))?, self.emb.u16s(&format!("{ep}.scales"))?,
                            self.emb.u16s(&format!("{ep}.biases"))?);
        let (rw, rs) = (d * self.emb.bits / 32, d / 64);
        let gather = |t: &[u16]| -> Vec<u16> { ids.iter().flat_map(|&i| t[i as usize * rs..(i as usize + 1) * rs].iter().copied()).collect() };
        write(&self.emb_w, &ids.iter().flat_map(|&i| tw[i as usize * rw..(i as usize + 1) * rw].iter().copied()).collect::<Vec<u32>>());
        write(&self.emb_s, &gather(ts));
        write(&self.emb_b, &gather(tb));
        let b = [self.emb_ids.clone(), self.emb_w.clone(), self.emb_s.clone(), self.emb_b.clone(), self.h.clone()];
        bind(e, &[&b[0], &b[1], &b[2], &b[3], &b[4]]);
        e.dispatchThreads_threadsPerThreadgroup(MTLSize { width: d, height: n, depth: 1 }, sz(256));
        Ok(())
    }

    /// Layers of `s` over the n rows in self.h (rows/anc/pos already written); new K/V at slots p0..; ends with the final
    /// norm in xh/xo/xs. capture: copy layer outputs listed in `cap` into self.feats.
    fn enc_stack(&mut self, e: &Enc, s: &Stack, n: usize, n_real: usize, p0: usize, span: usize, cap: &[usize]) -> Result<()> {
        let c = s.cfg.clone();
        let nb = (span.min(self.maxlen) + MAXD).div_ceil(BS);
        let p_swi = self.gpu.pipe("swiglu_prep", kernels::SWIGLU_PREP, &[("I", c.inter.to_string()), ("GELU", "0".into())])?;
        let b = [self.h.clone(), self.xh.clone(), self.xo.clone(), self.xs.clone(), self.qkv.clone(), self.gu.clone()];
        let [h, xh, xo, xs, qkv, gu] = &b;
        for (li, l) in s.layers.iter().enumerate() {
            self.enc_norm(e, c.d, c.eps, h, &l.in_n, n)?;
            if l.g4.is_some() {
                let src = &s.layers[l.at.src];
                self.enc_gemma_layer(e, &c, l, (&src.kc, &src.vc), li, n, n_real, p0, nb)?;
                if let Some(j) = cap.iter().position(|&x| x == li) {
                    self.enc_copy(e, c.d, j, cap.len(), n)?;
                }
                continue;
            }
            if let Some(lin) = &l.lin {
                self.enc_lin(e, &c, l, lin, n, n_real)?;
            } else if let Some(sc) = &l.sc {
                self.enc_sconv(e, c.d, l, sc, n, n_real, p0)?;
            } else {
                self.enc_qmm(e, &l.qkv, [xh, xo, xs], qkv, n, false)?;
                let pos = self.pos.clone();
                self.enc_rope(e, &c, l.at.h, qkv, l, n_real, p0, n, &pos)?;
                self.enc_attn(e, &l.at, (&l.kc, &l.vc), n, nb)?;
                self.enc_qmm(e, &l.o, [xh, xo, xs], h, n, true)?;
            }
            self.enc_norm(e, c.d, c.eps, h, &l.post_n, n)?;
            self.enc_qmm(e, &l.gu, [xh, xo, xs], gu, n, false)?;
            e.setComputePipelineState(&p_swi);
            bind(e, &[gu, xh, xo, xs, &self.rscale]);
            e.dispatchThreadgroups_threadsPerThreadgroup(sz(n), sz(256));
            self.enc_qmm(e, &l.down, [xh, xo, xs], h, n, true)?;
            if let Some(j) = cap.iter().position(|&x| x == li) {
                self.enc_copy(e, c.d, j, cap.len(), n)?;
            }
        }
        self.enc_norm(e, c.d, c.eps, h, &s.norm, n)
    }

    /// Attention of the n rows' q (self.q) over memory `kv`, merged into xh/xo/xs for the o-projection.
    fn enc_attn(&mut self, e: &Enc, a: &At, kv: (&Buf, &Buf), n: usize, nb: usize) -> Result<()> {
        let p_split = self.gpu.pipe("attn_split", kernels::ATTN_SPLIT, &split_defs(a, self.maxlen, nb))?;
        let p_merge = self.gpu.pipe("attn_merge", kernels::ATTN_MERGE, &merge_defs(a, nb))?;
        let b = [self.q.clone(), self.rows.clone(), self.anc.clone(), self.part.clone(), self.xh.clone(), self.xo.clone(), self.xs.clone(), self.qkv.clone()];
        let [q, rows, anc, part, xh, xo, xs, qkv] = &b;
        e.setComputePipelineState(&p_split);
        bind(e, &[q, kv.0, kv.1, rows, anc, part]);
        e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: n * a.h, height: nb, depth: 1 }, sz(32));
        e.setComputePipelineState(&p_merge);
        bind(e, &[part, xh, xo, xs, qkv]);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n * a.h), sz(32));
        Ok(())
    }

    /// h -> column block j of self.feats (layer outputs a DFlash guesser reads).
    fn enc_copy(&mut self, e: &Enc, d: usize, j: usize, nblk: usize, n: usize) -> Result<()> {
        let p = self.gpu.pipe("copy_cols", kernels::COPY_COLS, &[("D", d.to_string()), ("NBLK", nblk.max(1).to_string())])?;
        e.setComputePipelineState(&p);
        bind(e, &[&self.h, &self.feats]);
        set_args(e, &[j as i32], 2);
        e.dispatchThreads_threadsPerThreadgroup(MTLSize { width: d, height: n, depth: 1 }, sz(256));
        Ok(())
    }

    /// Linear layer (Gated DeltaNet) over the n rows prepared in xh: projections, conv, delta rule, gated norm,
    /// out-projection added to h. In a tree pass the committed state stays as it is (commit() replays accepted rows).
    fn enc_lin(&mut self, e: &Enc, c: &Cfg, l: &Layer, lin: &Lin, n: usize, n_real: usize) -> Result<()> {
        let lc = c.lin.unwrap();
        let (cc, nlin, ks) = (lc.c(), lc.nlin(), lc.ks());
        let b = [self.xh.clone(), self.xo.clone(), self.xs.clone(), self.gu.clone(), self.qkv.clone(), self.q.clone(), self.h.clone(),
                 self.lin_prev.clone(), self.lin_par.clone(), self.emb_ids.clone()];
        let [xh, xo, xs, gu, post, y, h, prev, par, ident] = &b;
        let defs = lin_defs(c);
        let tree = self.lin_tree;
        self.enc_qmm(e, &l.qkv, [xh, xo, xs], gu, n, false)?;
        let p = self.gpu.pipe("lin_conv", kernels::LIN_CONV, &defs)?;
        e.setComputePipelineState(&p);
        bind(e, &[gu, &lin.tail, &lin.conv, prev, post, &lin.keep, &self.lin_gb /* 6: args */, &lin.alog, &lin.dtb, &self.lin_gb]);
        set_args(e, &[tree as i32, n_real as i32], 6);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n * cc / 128), sz(32));
        if !tree {
            let p = self.gpu.pipe("lin_tail", kernels::LIN_TAIL, &defs)?;
            e.setComputePipelineState(&p);
            bind(e, &[gu, ident, &lin.tail]);
            set_args(e, &[n_real as i32, nlin as i32], 3);
            e.dispatchThreads_threadsPerThreadgroup(sz(cc), sz(256));
        }
        let r = if tree { 1 } else { LIN_R };
        let p = self.gpu.pipe("lin_delta", kernels::LIN_DELTA, &delta_defs(c, r))?;
        e.setComputePipelineState(&p);
        bind(e, &[post, &self.lin_gb, post, post, &lin.state, par, ident, y]);
        set_args(e, &[n_real as i32, !tree as i32, cc as i32, 0, 2 * lc.hv as i32, 0, tree as i32], 8);
        e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: lc.hv, height: lc.dv / 4 / r, depth: 1 }, sz(128));
        let p = self.gpu.pipe("lin_norm", kernels::LIN_NORM, &defs)?;
        e.setComputePipelineState(&p);
        bind(e, &[y, gu, &lin.norm, xh, xo, xs]);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n * lc.hv), sz(32));
        let _ = ks;
        self.enc_qmm(e, &l.o, [xh, xo, xs], h, n, true)
    }

    /// LFM2 short-convolution layer over the n rows prepared in xh: in_proj, conv over each row's own predecessors,
    /// out_proj added to h.
    fn enc_sconv(&mut self, e: &Enc, d: usize, l: &Layer, sc: &SConv, n: usize, n_real: usize, p0: usize) -> Result<()> {
        let b = [self.xh.clone(), self.xo.clone(), self.xs.clone(), self.qkv.clone(), self.tmp.clone(), self.lin_prev.clone()];
        let [xh, xo, xs, bcx, y, prv] = &b;
        self.enc_qmm(e, &l.qkv, [xh, xo, xs], bcx, n, false)?;
        let defs = [("D", d.to_string())];
        let pw = self.gpu.pipe("sconv_write", kernels::SCONV, &defs)?;
        e.setComputePipelineState(&pw);
        bind(e, &[bcx, &sc.cache]);
        set_args(e, &[n_real as i32, p0 as i32], 2);
        e.dispatchThreads_threadsPerThreadgroup(MTLSize { width: d, height: n, depth: 1 }, sz(256));
        let pc = self.gpu.pipe("sconv", kernels::SCONV, &defs)?;
        e.setComputePipelineState(&pc);
        bind(e, &[bcx, &sc.cache, &sc.w, prv, y]);
        set_args(e, &[n_real as i32, p0 as i32], 5);
        e.dispatchThreads_threadsPerThreadgroup(MTLSize { width: d, height: n, depth: 1 }, sz(256));
        self.enc_prep(e, y, d, n)?;
        self.enc_qmm(e, &l.o, [xh, xo, xs], &self.h.clone(), n, true)
    }

    /// Each row's two predecessor slots for the short-convolution layers (chain: the slots before it; tree: its path,
    /// continuing into the committed words before the root).
    fn write_sconv(&self, rows: &[[i32; 4]], anc: &[[i32; MAXD]], n_real: usize, n: usize) {
        if !self.t.cfg.conv.iter().any(|&x| x) {
            return;
        }
        let p = self.p as i32;
        let mut prv = vec![-1i32; n * 2];
        for r in 0..n_real {
            let up: Vec<i32> = if self.lin_tree {
                let na = rows[r][2] as usize; // path root..r, slots p + node
                (0..na.saturating_sub(1)).rev().map(|k| anc[r][k]).chain([p - 1, p - 2]).take(2).collect()
            } else {
                let s = p + r as i32;
                vec![s - 1, s - 2]
            };
            for k in 0..2 {
                prv[r * 2 + k] = if up[k] >= 0 { up[k] } else { -1 };
            }
        }
        write(&self.lin_prev, &prv);
    }

    /// Each row's conv history and state parent for the linear layers (chain: the rows before it; tree: its path).
    fn write_lin(&self, rows: &[[i32; 4]], anc: &[[i32; MAXD]], n_real: usize, n: usize) {
        if self.t.cfg.lin.is_none() {
            return;
        }
        let p = self.p as i32;
        let (mut prev, mut par) = (vec![0i32; n * 3], vec![-1i32; n]);
        for r in 0..n {
            let rr = if r < n_real { r } else { 0 };
            let up: Vec<i32> = if self.lin_tree {
                let na = rows[rr][2] as usize;
                (0..na.saturating_sub(1)).rev().map(|k| anc[rr][k] - p).collect()
            } else {
                (0..rr as i32).rev().take(3).collect()
            };
            par[r] = up.first().copied().unwrap_or(-1);
            for k in 0..3 {
                prev[r * 3 + k] = if k < up.len() { up[k] } else { -((k - up.len()) as i32 + 1) };
            }
        }
        write(&self.lin_prev, &prev);
        write(&self.lin_par, &par);
    }

    fn write_rows(&self, ids: &[i32], pos: &[i32], rows: &[[i32; 4]], anc: &[[i32; MAXD]]) -> usize {
        let n = ids.len().div_ceil(8) * 8;
        let pad = |v: &[i32]| -> Vec<i32> { let mut o = v.to_vec(); while o.len() < n { o.push(v[0]); } o };
        write(&self.ids, &pad(ids));
        write(&self.pos, &pad(pos));
        let mut rr = rows.to_vec();
        let mut aa = anc.to_vec();
        while rr.len() < n {
            rr.push(rows[0]);
            aa.push(anc[0]);
        }
        write(&self.rows, &rr);
        write(&self.anc, &aa);
        n
    }

    /// Target pass over a tree or prompt chunk. all_rows: greedy pick for every row (tree), else only for the last row.
    fn forward(&mut self, ids: &[i32], pos: &[i32], rows: &[[i32; 4]], anc: &[[i32; MAXD]], all_rows: bool) -> Result<Vec<i32>> {
        let n_real = ids.len();
        let n = self.write_rows(ids, pos, rows, anc);
        self.write_lin(rows, anc, n_real, n);
        self.write_sconv(rows, anc, n_real, n);
        let cap: Vec<usize> = self.g.as_ref().map(|g| g.target_layers.clone()).unwrap_or_default();
        let t = unsafe { &*(&self.t as *const Stack) };
        let (d, vocab) = (t.cfg.d, t.cfg.vocab);
        let cb = self.gpu.queue.commandBuffer().context("cb")?;
        let e = cb.computeCommandEncoder().context("enc")?;
        self.enc_embed(&e, d, n)?;
        if self.g4m.is_some() {
            self.enc_ple(&e, n)?;
        }
        self.enc_stack(&e, t, n, n_real, self.p, self.p + n_real, &cap)?;
        let head = unsafe { &*(&self.head as *const QW) };
        let (xh, xo, xs, logits, pick) = (self.xh.clone(), self.xo.clone(), self.xs.clone(), self.logits.clone(), self.pick.clone());
        let p_arg = self.gpu.pipe("argmax", kernels::ARGMAX, &[("V", vocab.to_string())])?;
        if all_rows {
            self.enc_qmm(&e, head, [&xh, &xo, &xs], &logits, n, false)?;
            e.setComputePipelineState(&p_arg);
            bind(&e, &[&logits, &pick]);
            e.dispatchThreadgroups_threadsPerThreadgroup(sz(n), sz(1024));
            e.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            self.gpu_verify += cb.GPUEndTime() - cb.GPUStartTime();
            if self.asst.is_some() {
                self.hrows = read(&self.h, n_real * d);
            }
            return Ok(read(&self.pick, n_real));
        }
        e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if self.asst.is_some() {
            self.hrows = read(&self.h, n_real * d);
        }
        // the last row's prepared input -> rows 0..8 of small head buffers
        let g = d / 64;
        let r = n_real - 1;
        let hx = self.gpu.buf(8 * d * 2);
        let ho = self.gpu.buf(8 * g * 4);
        let hs = self.gpu.buf(8 * g * 4);
        unsafe {
            for j in 0..8 {
                std::ptr::copy_nonoverlapping((xh.contents().as_ptr() as *const u16).add(r * d), (hx.contents().as_ptr() as *mut u16).add(j * d), d);
                std::ptr::copy_nonoverlapping((xo.contents().as_ptr() as *const f32).add(r * g), (ho.contents().as_ptr() as *mut f32).add(j * g), g);
                std::ptr::copy_nonoverlapping((xs.contents().as_ptr() as *const f32).add(r * g), (hs.contents().as_ptr() as *mut f32).add(j * g), g);
            }
        }
        let cb = self.gpu.queue.commandBuffer().context("cb")?;
        let e = cb.computeCommandEncoder().context("enc")?;
        self.enc_qmm(&e, head, [&hx, &ho, &hs], &logits, 8, false)?;
        e.setComputePipelineState(&p_arg);
        bind(&e, &[&logits, &pick]);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(8), sz(1024));
        e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        Ok(read(&self.pick, 1))
    }

    /// Target features (captured layer outputs) of rows `idx` of the last target pass -> guesser's pending context.
    fn keep_features(&mut self, idx: &[usize]) {
        if let Some(g) = &mut self.g {
            let f = g.target_layers.len() * self.t.cfg.d;
            let all: Vec<f32> = read(&self.feats, (idx.iter().max().unwrap() + 1) * f);
            for &r in idx {
                g.pending.extend_from_slice(&all[r * f..(r + 1) * f]);
            }
        }
    }

    /// Guesser: add pending context to its memory, then fill its block after `root`; returns per-position top-3
    /// (log-prob, token) for positions 1..block.
    fn guess(&mut self, root: i32) -> Result<Vec<[(f32, i32); 3]>> {
        let want = self.want_depth;
        let g = unsafe { &mut *(self.g.as_mut().unwrap() as *mut Guesser) };
        let gc = g.stack.cfg.clone();
        let f = g.target_layers.len() * gc.d;
        let cb = self.gpu.queue.commandBuffer().context("cb")?;
        let e = cb.computeCommandEncoder().context("enc")?;
        let (h, xh, xo, xs, qkv) = (self.h.clone(), self.xh.clone(), self.xo.clone(), self.xs.clone(), self.qkv.clone());
        let pend = std::mem::take(&mut g.pending);
        let total = pend.len() / f;
        // context: features -> fc -> hidden_norm -> each layer's k/v (with k_norm + RoPE at the words' positions)
        let mut done = 0;
        while done < total {
            let a = (total - done).min(MAXR);
            let n = a.div_ceil(8) * 8;
            let mut chunk = pend[done * f..(done + a) * f].to_vec();
            chunk.resize(n * f, 0.0);
            let fb = self.gpu.buf_from(&chunk);
            let pos: Vec<i32> = (0..n).map(|i| (g.c + i) as i32).collect();
            let posb = self.gpu.buf_from(&pos);  // own buffer: the block below rewrites self.pos before the GPU runs
            let p_prep = self.gpu.pipe("prep", kernels::PREP, &[("KD", f.to_string()), ("RSC", "0".into())])?;
            e.setComputePipelineState(&p_prep);
            bind(&e, &[&fb, &xh, &xo, &xs, &self.rscale]);
            e.dispatchThreadgroups_threadsPerThreadgroup(sz(n), sz(256));
            self.enc_qmm(&e, &g.fc, [&xh, &xo, &xs], &h, n, false)?;
            self.enc_norm(&e, gc.d, gc.eps, &h, &g.hidden_norm, n)?;
            for l in &g.stack.layers {
                self.enc_qmm(&e, l.kv.as_ref().unwrap(), [&xh, &xo, &xs], &qkv, n, false)?;
                self.enc_rope(&e, &gc, 0, &qkv, l, a, g.c, n, &posb)?;
            }
            // the encoder runs these in order; the next chunk's writes to pos/xh happen after this cb is committed
            if done + a < total {
                e.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                g.c += a;
                done += a;
                return self.guess_rest(root, pend[done * f..].to_vec());
            }
            g.c += a;
            done += a;
        }
        // block: [root, mask x (B-1)] at positions c.., every block row sees the whole context and the whole block
        let b = std::env::var("DFLASH_BLOCK").ok().and_then(|v| v.parse().ok()).unwrap_or(g.run_block).min(g.block);
        let ids: Vec<i32> = std::iter::once(root).chain(std::iter::repeat(g.mask).take(b - 1)).collect();
        let pos: Vec<i32> = (0..b).map(|j| (g.c + j) as i32).collect();
        let rows: Vec<[i32; 4]> = pos.iter().map(|&p| [p, g.c as i32, b as i32, 0]).collect();
        let mut a = [0i32; MAXD];
        for j in 0..b {
            a[j] = (g.c + j) as i32;
        }
        let anc = vec![a; b];
        let n = self.write_rows(&ids, &pos, &rows, &anc);
        self.enc_embed(&e, gc.d, n)?;
        self.enc_stack(&e, &g.stack, n, b, g.c, g.c + b, &[])?;
        let head = unsafe { &*(&self.head as *const QW) };
        let logits = self.logits.clone();
        // the tree cannot go deeper than hr; x rows start at block row 1 (DFlash) or at the anchor row 0 (DSpark)
        let first = if g.from_anchor { 0 } else { 1 };
        let hr = (b - first).min(want.max(1));
        let hn = hr.div_ceil(8) * 8;
        let (nt, sk) = self.cfg_for(head.n, head.k, hn / 8);
        self.enc_qmm_at(&e, head, [&xh, &xo, &xs], first, &logits, hn, false, nt, sk)?;
        let markov = g.markov.is_some();
        if markov {
            let p_c = self.gpu.pipe("topkc", kernels::TOPKC, &[("V", gc.vocab.to_string()), ("NEED", "64".into()), ("CAP", DCAP.to_string())])?;
            e.setComputePipelineState(&p_c);
            bind(&e, &[&logits, &self.cv, &self.ci, &self.cn]);
            e.dispatchThreadgroups_threadsPerThreadgroup(sz(hr), sz(1024));
        } else {
            let p_top = self.gpu.pipe("topk3", kernels::TOPK3, &[("V", gc.vocab.to_string())])?;
            e.setComputePipelineState(&p_top);
            let (tv, ti) = (self.tv.clone(), self.ti.clone());
            bind(&e, &[&logits, &tv, &ti]);
            e.dispatchThreadgroups_threadsPerThreadgroup(sz(hr), sz(256));
        }
        e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        self.gpu_draft += cb.GPUEndTime() - cb.GPUStartTime();
        if markov {
            let (cnt, v, t): (Vec<i32>, Vec<f32>, Vec<i32>) = (read(&self.cn, hr), read(&self.cv, hr * DCAP), read(&self.ci, hr * DCAP));
            self.dcand = (0..hr).map(|r| (0..cnt[r] as usize).map(|j| (v[r * DCAP + j], t[r * DCAP + j])).collect()).collect();
            // a top-3 view, only needed by the tree builder (DSPARK_TREE); the chain reads dcand directly
            if std::env::var("DSPARK_TREE").is_err() {
                return Ok(vec![[(0.0, 0); 3]; hr]);
            }
            return Ok(self.dcand.iter().map(|c| {
                let mut c = c.clone();
                c.sort_by(|a, b| b.0.total_cmp(&a.0));
                let at = |i: usize| c.get(i).copied().unwrap_or((f32::MIN, 0));
                [at(0), at(1), at(2)]
            }).collect());
        }
        let v: Vec<f32> = read(&self.tv, hr * 3);
        let t: Vec<i32> = read(&self.ti, hr * 3);
        Ok((0..hr).map(|r| [(v[r * 3], t[r * 3]), (v[r * 3 + 1], t[r * 3 + 1]), (v[r * 3 + 2], t[r * 3 + 2])]).collect())
    }

    fn guess_rest(&mut self, root: i32, rest: Vec<f32>) -> Result<Vec<[(f32, i32); 3]>> {
        self.g.as_mut().unwrap().pending = rest;
        self.guess(root)
    }

    /// DDTree: best-first by path log-prob over per-position top-3, up to `budget` nodes (root included).
    fn build_tree(&mut self, root: i32, budget: usize) -> Result<(Vec<i32>, Vec<i32>)> {
        self.want_depth = budget - 1;
        let top = self.guess(root)?;
        if std::env::var("DEBUG_GUESS").is_ok() {
            eprintln!("guess after {root} (context {}): {:?}", self.g.as_ref().unwrap().c, top.iter().map(|t| t[0].1).collect::<Vec<_>>());
            eprintln!("top values: {:?}", top.iter().take(3).map(|t| t[0].0).collect::<Vec<_>>());
            let d = self.t.cfg.d;
            let hv: Vec<f32> = read(&self.h, 16 * d);
            let lg: Vec<f32> = read(&self.logits, 3 * 1000);
            eprintln!("h row0[..4] {:?} nan rows {:?}", &hv[..4], (0..16).filter(|r| hv[r * d..(r + 1) * d].iter().any(|x| !x.is_finite())).collect::<Vec<_>>());
            eprintln!("logits row1[..4] {:?}", &lg[1000..1004]);
            let f: Vec<f32> = read(&self.feats, 2 * 5 * d);
            eprintln!("feats row0 max {:?}", f.iter().take(5 * d).fold(0f32, |a, &x| a.max(x.abs())));
            std::process::exit(0);
        }
        let _ = 0;
        #[derive(PartialEq)]
        struct C(f32, usize, usize, i32); // score, seq, parent node, token
        impl Eq for C {}
        impl PartialOrd for C { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
        impl Ord for C { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.0.total_cmp(&o.0).then(o.1.cmp(&self.1)) } }
        let (mut toks, mut parent, mut dep, mut score) = (vec![root], vec![-1i32], vec![0usize], vec![0f32]);
        let mut heap = BinaryHeap::new();
        let mut seq = 0;
        // DSpark: each drafted position's logits get a bias from the token drafted before it (the parent in the tree)
        let markov = self.g.as_ref().and_then(|g| g.markov.as_ref()).map(|w| {
            let r = w.shape("markov_head.markov_w1.weight")[1];
            (w.u16s("markov_head.markov_w1.weight").unwrap(), w.u16s("markov_head.markov_w2.weight").unwrap(), r)
        });
        let bias = |u: i32, v: i32| -> f32 {
            let Some((w1, w2, r)) = markov else { return 0.0 };
            if std::env::var("NO_MARKOV").is_ok() { return 0.0; }
            let (a, b) = (&w1[u as usize * r..(u as usize + 1) * r], &w2[v as usize * r..(v as usize + 1) * r]);
            a.iter().zip(b).map(|(&x, &y)| bf16_to_f32(x) * bf16_to_f32(y)).sum()
        };
        if markov.is_some() && !self.dcand.is_empty() && std::env::var("DSPARK_CHAIN").is_err() {
            // DSpark tree: best-first by path probability, each child scored as softmax over the candidates of
            // (base logit + Markov bias of its own parent's token): confident positions stay a chain, unsure ones branch
            let (w1, w2, r) = markov.unwrap();
            let kids = |u: i32, d: usize| -> Vec<(f32, i32)> {
                let a: Vec<f32> = w1[u as usize * r..(u as usize + 1) * r].iter().map(|&x| bf16_to_f32(x)).collect();
                let mut sc: Vec<(f32, i32)> = self.dcand[d].iter().map(|&(v, t)| {
                    let row = &w2[t as usize * r..(t as usize + 1) * r];
                    let mut acc = 0f32;
                    for k in 0..r { acc += a[k] * bf16_to_f32(row[k]); }
                    (v + acc, t)
                }).collect();
                let m = sc.iter().fold(f32::MIN, |x, y| x.max(y.0));
                let lse = m + sc.iter().map(|x| (x.0 - m).exp()).sum::<f32>().ln();
                sc.sort_by(|x, y| y.0.total_cmp(&x.0));
                sc.truncate(4);
                sc.into_iter().map(|(x, t)| (x - lse, t)).collect()
            };
            let hr = self.dcand.len();
            let (mut toks, mut parent, mut dep) = (vec![root], vec![-1i32], vec![0usize]);
            let mut heap = BinaryHeap::new();
            let mut seq = 0;
            for (lp, t) in kids(root, 0) { heap.push(C(lp, seq, 0, t)); seq += 1; }
            while let Some(C(sc, _, p, t)) = heap.pop() {
                if toks.len() >= budget { break; }
                toks.push(t); parent.push(p as i32); dep.push(dep[p] + 1);
                let (c, d) = (toks.len() - 1, dep[toks.len() - 1]);
                if d < hr && d < MAXD - 1 {
                    for (lp, tt) in kids(t, d) { heap.push(C(sc + lp, seq, c, tt)); seq += 1; }
                }
            }
            let mut order: Vec<usize> = (0..toks.len()).collect();
            order.sort_by_key(|&i| (dep[i], i));
            let mut new = vec![0usize; toks.len()];
            for (j, &o) in order.iter().enumerate() { new[o] = j; }
            return Ok((order.iter().map(|&o| toks[o]).collect(), order.iter().map(|&o| if parent[o] >= 0 { new[parent[o] as usize] as i32 } else { -1 }).collect()));
        }
        if markov.is_some() && std::env::var("DSPARK_TREE").is_err() {
            // DSpark is trained to draft a chain: at each position, argmax of base logit + Markov bias of the token before
            let (w1, w2, r) = markov.unwrap();
            let (mut toks, mut prev) = (vec![root], root);
            let mut a = vec![0f32; r];
            for d in 0..self.dcand.len().min(budget - 1) {
                for (k, x) in w1[prev as usize * r..(prev as usize + 1) * r].iter().enumerate() { a[k] = bf16_to_f32(*x); }
                let (mut bs, mut best) = (f32::MIN, -1);
                for &(v, t) in &self.dcand[d] {
                    let row = &w2[t as usize * r..(t as usize + 1) * r];
                    let mut acc = 0f32;
                    for k in 0..r { acc += a[k] * bf16_to_f32(row[k]); }
                    if v + acc > bs { bs = v + acc; best = t; }
                }
                if best < 0 { break; }
                toks.push(best);
                prev = best;
            }
            let parent: Vec<i32> = (0..toks.len() as i32).map(|i| i - 1).collect();
            return Ok((toks, parent));
        }
        for &(v, t) in &top[0] {
            heap.push(C(v + bias(root, t), seq, 0, t));
            seq += 1;
        }
        while let Some(C(s, _, p, t)) = heap.pop() {
            if toks.len() >= budget {
                break;
            }
            toks.push(t);
            parent.push(p as i32);
            dep.push(dep[p] + 1);
            score.push(s);
            let (c, d) = (toks.len() - 1, dep[toks.len() - 1]);
            if d < top.len() && d < MAXD - 1 {
                for &(v, tt) in &top[d] {
                    heap.push(C(s + v + bias(t, tt), seq, c, tt));
                    seq += 1;
                }
            }
        }
        // breadth-first order (parents before children; slots = P + index)
        let mut order: Vec<usize> = (0..toks.len()).collect();
        order.sort_by_key(|&i| (dep[i], i));
        let mut new = vec![0usize; toks.len()];
        for (j, &o) in order.iter().enumerate() {
            new[o] = j;
        }
        Ok((order.iter().map(|&o| toks[o]).collect(), order.iter().map(|&o| if parent[o] >= 0 { new[parent[o] as usize] as i32 } else { -1 }).collect()))
    }

    fn feed(&mut self, ids: &[i32]) -> Result<i32> {
        let mut last = 0;
        for c in ids.chunks(MAXR) {
            let n = c.len();
            let pos: Vec<i32> = (0..n).map(|i| (self.p + i) as i32).collect();
            let rows: Vec<[i32; 4]> = pos.iter().map(|&p| [p, p + 1, 0, 0]).collect();
            last = self.forward(c, &pos, &rows, &vec![[0i32; MAXD]; n], false)?[0];
            self.keep_features(&(0..n).collect::<Vec<_>>());
            if self.asst.is_some() {
                self.last_hn = self.hidden_row(n - 1);
            }
            self.p += n;
            self.hist.extend_from_slice(c);
        }
        Ok(last)
    }

    /// Copy the linear layers' committed state and conv tail (hybrid models), to continue a later prompt from here.
    fn take_snap(&mut self) {
        let mut v = Vec::new();
        for l in &self.t.layers {
            if let Some(lin) = &l.lin {
                for b in [&lin.state, &lin.tail] {
                    v.push(unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const u8, b.length()) }.to_vec());
                }
            }
        }
        self.snap = Some((self.p, v));
    }

    fn restore_snap(&mut self) {
        let Some((_, v)) = &self.snap else { return };
        let mut it = v.iter();
        for l in &self.t.layers {
            if let Some(lin) = &l.lin {
                for b in [&lin.state, &lin.tail] {
                    write(b, it.next().unwrap());
                }
            }
        }
    }

    fn plain(&mut self, prompt: &[i32], max_new: usize, eos: &[i32]) -> Result<(Vec<i32>, f64)> {
        self.reset();
        let mut y = self.feed(prompt)?;
        let mut out = vec![y];
        let t = Instant::now();
        while out.len() < max_new && !eos.contains(&y) {
            let p = self.p as i32;
            y = self.forward(&[y], &[p], &[[p, p + 1, 0, 0]], &[[0; MAXD]], true)?[0];
            self.p += 1;
            self.hist.push(out[out.len() - 1]);
            out.push(y);
        }
        Ok((out, t.elapsed().as_secs_f64()))
    }

    /// Move accepted tree rows (slots P + node) to P..P+a-1 in every target layer's memory (CPU, shared buffers).
    fn commit(&mut self, acc: &[usize]) -> Result<()> {
        for l in self.t.layers.iter().filter(|l| l.lin.is_none() && l.at.kvh_w > 0) {
            let row_bytes = l.at.hd * 2;
            for b in [&l.kc, &l.vc] {
                let base = b.contents().as_ptr() as *mut u8;
                for kh in 0..l.at.kvh_w {
                    for (i, &node) in acc.iter().enumerate() {
                        if node != i {
                            unsafe {
                                std::ptr::copy(base.add((kh * self.maxlen + self.p + node) * row_bytes),
                                               base.add((kh * self.maxlen + self.p + i) * row_bytes), row_bytes);
                            }
                        }
                    }
                }
            }
        }
        for l in &self.t.layers {
            let Some(sc) = &l.sc else { continue };
            let row_bytes = self.t.cfg.d * 2;
            let base = sc.cache.contents().as_ptr() as *mut u8;
            for (i, &node) in acc.iter().enumerate() {
                if node != i {
                    unsafe { std::ptr::copy(base.add((self.p + node) * row_bytes), base.add((self.p + i) * row_bytes), row_bytes) };
                }
            }
        }
        // linear layers: replay the accepted rows from the committed state (same kernel, same arithmetic as in the tree)
        if let Some(lc) = self.t.cfg.lin {
            let c = self.t.cfg.clone();
            let (cc, ks) = (lc.c() as i32, lc.ks() as i32);
            write(&self.lin_acc, &acc.iter().map(|&a| a as i32).collect::<Vec<_>>());
            let defs = lin_defs(&c);
            let pd = self.gpu.pipe("lin_delta", kernels::LIN_DELTA, &delta_defs(&c, LIN_R))?;
            let pt = self.gpu.pipe("lin_tail", kernels::LIN_TAIL, &defs)?;
            let cb = self.gpu.queue.commandBuffer().context("cb")?;
            let e = cb.computeCommandEncoder().context("enc")?;
            for l in &self.t.layers {
                let Some(lin) = &l.lin else { continue };
                e.setComputePipelineState(&pd);
                bind(&e, &[&lin.keep, &lin.keep, &lin.keep, &lin.keep, &lin.state, &self.lin_chain, &self.lin_acc, &self.q]);
                set_args(&e, &[acc.len() as i32, 1, ks, cc, ks, 2 * cc, 0], 8);
                e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: lc.hv, height: lc.dv / 4 / LIN_R, depth: 1 }, sz(128));
                e.setComputePipelineState(&pt);
                bind(&e, &[&lin.keep, &self.lin_acc, &lin.tail]);
                set_args(&e, &[acc.len() as i32, ks], 3);
                e.dispatchThreads_threadsPerThreadgroup(sz(lc.c()), sz(256));
            }
            e.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
        }
        self.p += acc.len();
        Ok(())
    }

    fn tree(&mut self, prompt: &[i32], max_new: usize, eos: &[i32], budget: usize) -> Result<(Vec<i32>, f64, usize)> {
        let mut out = Vec::new();
        let (t, passes) = self.generate(prompt, max_new, eos, budget, &mut |w| { out.extend_from_slice(w); true })?;
        Ok((out, t, passes))
    }

    /// Exact tree decoding; `on_words` gets each pass's new words (return false to stop). Returns (seconds, passes).
    pub fn generate(&mut self, prompt: &[i32], max_new: usize, eos: &[i32], budget: usize,
                    on_words: &mut dyn FnMut(&[i32]) -> bool) -> Result<(f64, usize)> {
        // tree rows' ancestor lists and the per-row buffers hold MAXD entries: a bigger tree overflows them and
        // silently breaks exactness for this and later calls
        anyhow::ensure!(budget >= 1 && budget <= MAXD, "tree budget {budget} not supported (1..={MAXD})");
        let mut keep = if self.reuse { self.shared_prefix(prompt) } else { 0 };
        let hybrid = self.t.cfg.lin.is_some();
        if hybrid && keep != self.hist.len() {
            // a running state can only be continued, or restored from the snapshot at the last turn boundary (where the
            // next prompt's history departs from this one: the template rewrites the answer's think block)
            match self.snap.as_ref().map(|s| s.0) {
                Some(sp) if sp > 0 && keep >= sp => {
                    keep = sp;
                    self.restore_snap();
                }
                _ => keep = 0,
            }
        }
        self.rewind(keep);
        self.kept = keep;
        self.draft_s = 0.0;
        self.commit_s = 0.0;
        self.gpu_draft = 0.0;
        self.gpu_verify = 0.0;
        // hybrid + reuse: read the prompt up to just after its last "<|im_start|>assistant\n", snapshot, read the rest
        let cut = match (hybrid && self.reuse, self.turn_tok) {
            (true, Some(tt)) => prompt.iter().rposition(|&x| x == tt).map(|i| i + 3).filter(|&b| b > keep && b < prompt.len()),
            _ => None,
        };
        let mut y = match cut {
            Some(b) => {
                self.feed(&prompt[keep..b])?;
                self.take_snap();
                self.feed(&prompt[b..])?
            }
            None => self.feed(&prompt[keep..])?,
        };
        let mut n_out = 1;
        let mut passes = 0;
        let t = Instant::now();
        if !on_words(&[y]) {
            return Ok((0.0, 0));
        }
        while n_out < max_new && !eos.contains(&y) {
            let td = Instant::now();
            let (toks, parent) = if self.asst.is_some() {
                self.build_tree_asst(y, budget)?
            } else if self.g.is_some() {
                self.build_tree(y, budget)?
            } else {
                (vec![y], vec![-1])
            };
            self.draft_s += td.elapsed().as_secs_f64();
            let n = toks.len();
            let mut depth = vec![0usize; n];
            for j in 1..n {
                depth[j] = depth[parent[j] as usize] + 1;
            }
            let p = self.p as i32;
            let pos: Vec<i32> = depth.iter().map(|&dd| p + dd as i32).collect();
            let (mut rows, mut anc) = (Vec::new(), Vec::new());
            for j in 0..n {
                let mut path = vec![];
                let mut c = j as i32;
                while c >= 0 {
                    path.push(c);
                    c = parent[c as usize];
                }
                path.reverse();
                let mut a = [0i32; MAXD];
                for (k, &x) in path.iter().enumerate() {
                    a[k] = p + x;
                }
                rows.push([pos[j], p, path.len() as i32, 0]);
                anc.push(a);
            }
            self.lin_tree = true;
            let best = self.forward(&toks, &pos, &rows, &anc, true);
            self.lin_tree = false;
            let best = best?;
            passes += 1;
            let (mut cur, mut acc, mut emitted) = (0usize, vec![0usize], vec![]);
            loop {
                let nxt = best[cur];
                emitted.push(nxt);
                match (0..n).find(|&c| parent[c] == cur as i32 && toks[c] == nxt) {
                    Some(k) if !eos.contains(&nxt) => {
                        cur = k;
                        acc.push(k);
                    }
                    _ => break,
                }
            }
            for &a in &acc {
                self.hist.push(toks[a]);
            }
            self.keep_features(&acc);
            if self.asst.is_some() {
                self.last_hn = self.hidden_row(*acc.last().unwrap());
            }
            let tc = Instant::now();
            self.commit(&acc)?;
            self.commit_s += tc.elapsed().as_secs_f64();
            emitted.truncate(max_new - n_out);
            n_out += emitted.len();
            y = *emitted.last().unwrap();
            if !on_words(&emitted) {
                break;
            }
        }
        Ok((t.elapsed().as_secs_f64(), passes))
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(|s| s.as_str()) == Some("specbench") {
        // fast-rs specbench MODEL GUESSER question.jsonl OUT.jsonl [MAX_NEW=1024] [BUDGET=8]
        return specbench(&a[2..]);
    }
    if a.get(1).map(|s| s.as_str()) == Some("prefixtest") {
        // fast-rs prefixtest MODEL GUESSER question.jsonl [N] [MAX_NEW]
        return prefixtest(&a[2..]);
    }
    if a.get(1).map(|s| s.as_str()) == Some("bench") {
        // fast-rs bench MODEL GUESSER|- PROMPTS.jsonl OUT.jsonl [RUNS] [BUDGET] [MAX_NEW]
        return bench(&a[2..]);
    }
    if a.get(1).map(|s| s.as_str()) == Some("serve") {
        // fast-rs serve MODEL [GUESSER] [PORT]
        let model = a.get(2).context("model dir")?.clone();
        let guesser = a.get(3).filter(|s| s.as_str() != "-").cloned();
        let port: u16 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(8800);
        return server::run(model, guesser, port);
    }
    let dir = Path::new(a.get(1).map(|s| s.as_str()).unwrap_or("/Users/mrunalpendem/models/qwen3-4b-4bit"));
    let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(a.get(2).context("prompts json")?)?)?;
    let max_new: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let guesser = a.get(4).map(Path::new);
    let budgets: Vec<usize> = a.get(5).map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![8, 16]);
    let mut m = Model::load(dir, guesser, 4096)?;
    eprintln!("GPU memory {:.2} GB of {:.2} GB comfortable", m.gpu.dev.currentAllocatedSize() as f64 / 1e9,
              m.gpu.dev.recommendedMaxWorkingSetSize() as f64 / 1e9);
    m.autotune()?;
    if std::env::var("BENCH_LIN").is_ok() {
        // GPU time of one linear layer's kernels at 1 / 8 / 16 rows, tree and chain (20 repeats in one encoder)
        let li = m.t.layers.iter().position(|l| l.lin.is_some()).context("no linear layer")?;
        let l = unsafe { &*(&m.t.layers[li] as *const Layer) };
        let c = m.t.cfg.clone();
        for rows in [1usize, 8, 16] {
            for tree in [false, true] {
                m.lin_tree = tree;
                let n = rows.div_ceil(8) * 8;
                let rr: Vec<[i32; 4]> = (0..rows).map(|j| [j as i32, 0, j as i32 + 1, 0]).collect();
                let aa: Vec<[i32; MAXD]> = (0..rows).map(|j| { let mut a = [0i32; MAXD]; for k in 0..=j { a[k] = k as i32; } a }).collect();
                m.write_lin(&rr, &aa, rows, n);
                let mut best = f64::MAX;
                for _ in 0..10 {
                    let cb = m.gpu.queue.commandBuffer().unwrap();
                    let e = cb.computeCommandEncoder().unwrap();
                    for _ in 0..20 {
                        m.enc_lin(&e, &c, l, l.lin.as_ref().unwrap(), n, rows)?;
                    }
                    e.endEncoding();
                    cb.commit();
                    cb.waitUntilCompleted();
                    best = best.min((cb.GPUEndTime() - cb.GPUStartTime()) / 20.0);
                }
                println!("linear layer, {rows:2} rows, {}: {:6.1} us", if tree { "tree " } else { "chain" }, best * 1e6);
            }
        }
        m.lin_tree = false;
        return Ok(());
    }
    if std::env::var("BENCH_PASS").is_ok() {
        // matmul share of one 8-row verify pass: every target matrix (and the head) timed alone on the GPU clock.
        // Bring the GPU up to full clock first (short bursts run at a lower one).
        {
            let l0 = unsafe { &*(&m.t.layers[0] as *const Layer) };
            let (xh, xo, xs, out) = (m.xh.clone(), m.xo.clone(), m.xs.clone(), m.logits.clone());
            let t0 = Instant::now();
            while t0.elapsed().as_secs_f64() < 3.0 {
                let cb = m.gpu.queue.commandBuffer().unwrap();
                let e = cb.computeCommandEncoder().unwrap();
                for _ in 0..100 { m.enc_qmm(&e, &l0.gu, [&xh, &xo, &xs], &out, 8, false)?; }
                e.endEncoding(); cb.commit(); cb.waitUntilCompleted();
            }
        }
        let mut cache: HashMap<(usize, usize), f64> = HashMap::new();
        let mut time = |m: &mut Model, w: &QW| -> Result<f64> {
            if let Some(&t) = cache.get(&(w.n, w.k)) { return Ok(t); }
            let (xh, xo, xs, out) = (m.xh.clone(), m.xo.clone(), m.xs.clone(), m.logits.clone());
            let mut best = f64::MAX;
            for _ in 0..6 {
                let cb = m.gpu.queue.commandBuffer().unwrap();
                let e = cb.computeCommandEncoder().unwrap();
                for _ in 0..60 { m.enc_qmm(&e, w, [&xh, &xo, &xs], &out, 8, false)?; }
                e.endEncoding(); cb.commit(); cb.waitUntilCompleted();
                best = best.min((cb.GPUEndTime() - cb.GPUStartTime()) / 60.0);
            }
            cache.insert((w.n, w.k), best);
            Ok(best)
        };
        let (mut tot, mut bytes) = (0.0, 0usize);
        let mut by: HashMap<&str, f64> = HashMap::new();
        let n_layers = m.t.layers.len();
        for i in 0..n_layers {
            let l = unsafe { &*(&m.t.layers[i] as *const Layer) };
            let kind = if l.sc.is_some() { "conv" } else if l.lin.is_some() { "linear" } else { "attn" };
            let mut ws: Vec<(&str, &QW)> = vec![("in/qkv", &l.qkv), ("out", &l.o), ("gate+up", &l.gu), ("down", &l.down)];
            if let Some(g) = &l.g4 { if let Some((a, b, _)) = &g.ple { ws.push(("ple", a)); ws.push(("ple", b)); } }
            for (nm, w) in ws {
                let t = time(&mut m, w)?;
                tot += t; bytes += w.n * w.k / 2 + w.n * w.k / 64 * 4;
                *by.entry(if nm == "in/qkv" || nm == "out" { kind } else { "mlp" }).or_default() += t;
            }
        }
        let head = unsafe { &*(&m.head as *const QW) };
        let th = time(&mut m, head)?;
        let mut parts: Vec<_> = by.into_iter().collect();
        parts.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("8-row matmuls of one verify pass: {:.2} ms layers ({:.0} MB, {:.0} GB/s) + head {:.2} ms = {:.2} ms",
                 tot * 1e3, bytes as f64 / 1e6, bytes as f64 / tot / 1e9, th * 1e3, (tot + th) * 1e3);
        println!("  by part: {}", parts.iter().map(|(k, v)| format!("{k} {:.2} ms", v * 1e3)).collect::<Vec<_>>().join(", "));
        return Ok(());
    }
    if std::env::var("BENCH_QMM").is_ok() {
        // GPU time of each target matrix at 8 and 16 rows (20 repeats in one encoder)
        let l0 = unsafe { &*(&m.t.layers[0] as *const Layer) };
        let head = unsafe { &*(&m.head as *const QW) };
        for (name, w) in [("qkv", &l0.qkv), ("o", &l0.o), ("gate+up", &l0.gu), ("down", &l0.down), ("head", head)] {
            let mut line = format!("{name:8} N={:6} K={:5}", w.n, w.k);
            for rows in [8usize, 16] {
                let (xh, xo, xs, out) = (m.xh.clone(), m.xo.clone(), m.xs.clone(), m.logits.clone());
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let cb = m.gpu.queue.commandBuffer().unwrap();
                    let e = cb.computeCommandEncoder().unwrap();
                    for _ in 0..20 {
                        m.enc_qmm(&e, w, [&xh, &xo, &xs], &out, rows, false)?;
                    }
                    e.endEncoding();
                    cb.commit();
                    cb.waitUntilCompleted();
                    best = best.min((cb.GPUEndTime() - cb.GPUStartTime()) / 20.0);
                }
                let gbs = (w.n * w.k / 2 + w.n * w.k / 64 * 4) as f64 / best / 1e9;
                line += &format!(" | {rows} rows {:7.1} us {:5.1} GB/s", best * 1e6, gbs);
            }
            println!("{line}");
        }
        return Ok(());
    }
    let eos: Vec<i32> = prompts["eos"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
    let mut dump = serde_json::Map::new(); // DUMP=path: each prompt's plain words, for comparing against a reference
    for p in prompts["prompts"].as_array().unwrap() {
        let name = p["name"].as_str().unwrap();
        let ids: Vec<i32> = p["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        let reference: Vec<i32> = p["python_plain"].as_array().map(|v| v.iter().map(|x| x.as_i64().unwrap() as i32).collect()).unwrap_or_default();
        m.plain(&ids, 8, &eos)?; // warm up (compile)
        let (plain, tp) = m.plain(&ids, max_new, &eos)?;
        let same_py = plain.iter().zip(&reference).take_while(|(a, b)| a == b).count();
        dump.insert(name.to_string(), serde_json::json!(plain));
        let mut line = format!("{name:10} plain {:5.1} w/s (= Python {same_py}/{})", plain.len() as f64 / tp, reference.len().min(plain.len()));
        if m.g.is_some() || m.asst.is_some() {
            for &b in &budgets {
                m.tree(&ids, 16, &eos, b)?; // warm up this tree size
                m.prof = [0.0; 4];
                let (tree, tt, passes) = m.tree(&ids, max_new, &eos, b)?;
                if m.asst.is_some() {
                    eprintln!("  assistant per pass: {:.1} rounds, GPU {:.1} ms, CPU head {:.1} ms", m.prof[2] / passes as f64,
                              m.prof[0] / passes as f64 * 1e3, m.prof[1] / passes as f64 * 1e3);
                }
                let pp = passes as f64 / 1e3;
                line += &format!(" | tree{b}: {:5.1} w/s {:.2} w/chk (guess {:.1} + check {:.1} ms, commit {:.1}; GPU busy {:.1}+{:.1} of {:.1} ms) {}", tree.len() as f64 / tt,
                    tree.len() as f64 / passes as f64, m.draft_s / pp, (tt - m.draft_s) / pp,
                    m.commit_s / pp, m.gpu_draft / pp, m.gpu_verify / pp, tt / pp,
                    if tree == plain { "OK" } else { "DIFF" });
            }
        }
        println!("{line}");
    }
    if let Ok(path) = std::env::var("DUMP") {
        std::fs::write(path, serde_json::Value::Object(dump).to_string())?;
    }
    Ok(())
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() { 0.0 } else if v.len() % 2 == 1 { v[v.len() / 2] } else { (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0 }
}

/// Formal benchmark: (1) llama-bench-style pp512 / tg128 (plain decoding, random tokens), (2) every prompt of a
/// tokenized set: plain once (the reference), then RUNS tree runs; speed = generated words / generation time
/// (prompt reading excluded, as mlx-lm and Spec-Bench report it); identical-output check on every run.
fn bench(a: &[String]) -> Result<()> {
    let model = Path::new(&a[0]);
    let guesser = Some(&a[1]).filter(|s| s.as_str() != "-").map(Path::new);
    let lines = std::fs::read_to_string(&a[2])?;
    let mut out = std::fs::File::create(&a[3])?;
    let runs: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);
    let budget: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
    let max_new: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(128);
    use std::io::Write;
    let mut m = Model::load(model, guesser, 4096)?;
    m.autotune()?;
    let swap = || std::process::Command::new("sysctl").arg("-n").arg("vm.swapusage").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    eprintln!("swap at start: {}", swap());
    // (1) pp512 / tg128: random token ids (as llama-bench), plain decoding, median of RUNS
    {
        let mut rng = 12345u64;
        let mut rnd = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng % 150000) as i32 + 100 };
        let prompt: Vec<i32> = (0..512).map(|_| rnd()).collect();
        let (mut pp, mut tg) = (Vec::new(), Vec::new());
        m.plain(&prompt[..64], 8, &[])?;
        for _ in 0..runs {
            m.reset();
            let t = Instant::now();
            m.feed(&prompt)?;
            pp.push(512.0 / t.elapsed().as_secs_f64());
            let (o, s) = m.plain(&prompt, 128, &[])?;
            tg.push(o.len() as f64 / s);
        }
        let (ppm, tgm) = (median(&mut pp.clone()), median(&mut tg.clone()));
        println!("pp512 {ppm:.1} t/s (runs {pp:.1?}) | tg128 {tgm:.1} t/s (runs {tg:.1?})");
        writeln!(out, "{}", serde_json::json!({"test": "pp512_tg128", "pp512": pp, "tg128": tg}))?;
    }
    // (2) prompt set
    let mut by_cat: std::collections::BTreeMap<String, (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize)> = Default::default();
    let (mut tot_w, mut tot_t, mut tot_wp, mut tot_tp) = (0usize, 0f64, 0usize, 0f64);
    let mut first = true;
    for line in lines.lines().filter(|l| !l.trim().is_empty()) {
        let p: serde_json::Value = serde_json::from_str(line)?;
        let cat = p["category"].as_str().unwrap_or("all").to_string();
        let ids: Vec<i32> = p["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        let eos: Vec<i32> = p["eos"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        if first {
            m.tree(&ids, 16, &eos, budget)?; // warm-up
            first = false;
        }
        let (plain, tp) = m.plain(&ids, max_new, &eos)?;
        let mut speeds = Vec::new();
        let mut wpc = Vec::new();
        let mut same = 0;
        for _ in 0..runs {
            let (tree, tt, passes) = m.tree(&ids, max_new, &eos, budget)?;
            speeds.push(tree.len() as f64 / tt);
            wpc.push(tree.len() as f64 / passes.max(1) as f64);
            same += (tree == plain) as usize;
            tot_w += tree.len();
            tot_t += tt;
        }
        tot_wp += plain.len();
        tot_tp += tp;
        let e = by_cat.entry(cat.clone()).or_default();
        e.0.push(median(&mut speeds.clone()));
        e.1.push(wpc[0]);
        e.2.push(plain.len() as f64 / tp);
        e.3 += same;
        e.4 += runs;
        writeln!(out, "{}", serde_json::json!({"category": cat, "id": p["id"], "prompt_words": ids.len(), "words": plain.len(),
            "plain_wps": plain.len() as f64 / tp, "tree_wps_runs": speeds, "words_per_check": wpc[0], "identical_runs": same, "runs": runs}))?;
        out.flush()?;
    }
    println!("\n{:16} {:>7} {:>9} {:>8} {:>12} {:>10}", "category", "prompts", "plain w/s", "tree w/s", "words/check", "identical");
    for (cat, (sp, wc, pl, same, n)) in by_cat.iter_mut() {
        let k = sp.len();
        println!("{cat:16} {k:>7} {:>9.1} {:>8.1} {:>12.2} {:>6}/{n}", median(pl), median(sp), median(wc), same);
    }
    println!("overall (total words / total time): plain {:.1} w/s, tree {:.1} w/s = {:.2}x | swap at end: {}",
             tot_wp as f64 / tot_tp, tot_w as f64 / tot_t, (tot_w as f64 / tot_t) / (tot_wp as f64 / tot_tp), swap());
    Ok(())
}

/// Spec-Bench, its protocol: every question of question.jsonl (all turns; turn k sees the model's earlier answers),
/// greedy, up to MAX_NEW new tokens per turn, batch 1. Per question: speed = new tokens / wall time over its turns
/// (wall time of the whole call, prompt reading included), for plain autoregressive decoding (the baseline) and for
/// our method. Speedup = mean method speed / mean baseline speed. Mean accepted tokens = tokens / verification steps.
/// Output equality is checked on every turn (Spec-Bench's equal.py).
fn specbench(a: &[String]) -> Result<()> {
    use std::io::Write;
    let model = std::path::PathBuf::from(&a[0]);
    let guesser = Some(&a[1]).filter(|s| s.as_str() != "-").map(std::path::PathBuf::from);
    let qs = std::fs::read_to_string(&a[2])?;
    let mut out = std::fs::File::create(&a[3])?;
    let max_new: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let budget: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
    let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let mut env = minijinja::Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template_owned("chat", server::template(&model)?)?;
    let mut eos: Vec<i32> = Vec::new();
    for t in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tok.token_to_id(t) {
            eos.push(id as i32);
        }
    }
    let maxlen = 6144;
    let mut m = Model::load(&model, guesser.as_deref(), maxlen)?;
    m.autotune()?;
    m.warm()?;
    let swap = || std::process::Command::new("sysctl").arg("-n").arg("vm.swapusage").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let power = || std::process::Command::new("pmset").args(["-g", "batt"]).output().map(|o| String::from_utf8_lossy(&o.stdout).lines().next().unwrap_or("").to_string()).unwrap_or_default();
    eprintln!("start: swap {} | {}", swap(), power());
    const MT: [&str; 8] = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"];
    // warm-up generation (not recorded)
    let w = tok.encode(server::render(&env, &[serde_json::json!({"role": "user", "content": "Hello"})])?, false).map_err(|e| anyhow!("{e}"))?;
    let wids: Vec<i32> = w.get_ids().iter().map(|&x| x as i32).collect();
    m.plain(&wids, 32, &eos)?;
    m.tree(&wids, 32, &eos, budget)?;
    let mut cats: std::collections::BTreeMap<String, Vec<(f64, f64, f64, bool)>> = Default::default();
    let (mut skipped, mut done) = (0, 0);
    let t_all = Instant::now();
    for line in qs.lines().filter(|l| !l.trim().is_empty()) {
        let q: serde_json::Value = serde_json::from_str(line)?;
        let cat = q["category"].as_str().unwrap_or("").to_string();
        let cat = if MT.contains(&cat.as_str()) { "mt_bench".to_string() } else { cat };
        let mut msgs: Vec<serde_json::Value> = Vec::new();
        let (mut tok_b, mut wall_b, mut tok_m, mut wall_m, mut steps) = (0usize, 0f64, 0usize, 0f64, 0usize);
        let mut identical = true;
        let mut ok = true;
        let mut turns_rec = Vec::new();
        for turn in q["turns"].as_array().unwrap() {
            msgs.push(serde_json::json!({"role": "user", "content": turn}));
            let ids: Vec<i32> = tok.encode(server::render(&env, &msgs)?, false).map_err(|e| anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            if ids.len() + max_new + 2 * MAXD > maxlen {
                ok = false;
                break;
            }
            let t = Instant::now();
            let (base, _) = m.plain(&ids, max_new, &eos)?;
            let wb = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (meth, _, passes) = m.tree(&ids, max_new, &eos, budget)?;
            let wm = t.elapsed().as_secs_f64();
            identical &= base == meth;
            tok_b += base.len();
            wall_b += wb;
            tok_m += meth.len();
            wall_m += wm;
            steps += passes + 1; // + the prompt pass that produced the first token
            turns_rec.push(serde_json::json!({"prompt_tokens": ids.len(), "new_tokens": base.len(), "base_wall": wb, "method_wall": wm,
                                              "steps": passes + 1, "identical": base == meth}));
            let ans: Vec<u32> = base.iter().filter(|x| !eos.contains(x)).map(|&x| x as u32).collect();
            msgs.push(serde_json::json!({"role": "assistant", "content": tok.decode(&ans, true).unwrap_or_default()}));
        }
        if !ok {
            skipped += 1;
            continue;
        }
        done += 1;
        let (sb, sm, tau) = (tok_b as f64 / wall_b, tok_m as f64 / wall_m, tok_m as f64 / steps as f64);
        cats.entry(cat.clone()).or_default().push((sb, sm, tau, identical));
        writeln!(out, "{}", serde_json::json!({"question_id": q["question_id"], "category": cat, "baseline_speed": sb, "method_speed": sm,
            "mean_accepted": tau, "identical": identical, "turns": turns_rec}))?;
        out.flush()?;
        if done % 20 == 0 {
            eprintln!("{done} questions, {:.0} min, swap {}", t_all.elapsed().as_secs_f64() / 60.0, swap());
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!("Spec-Bench ({} + {} tree {budget}, greedy, max {max_new} new tokens/turn, wall time incl. prompt reading)", model.file_name().unwrap_or_default().to_string_lossy(), guesser.as_deref().and_then(|g| g.file_name()).map(|g| g.to_string_lossy().to_string()).unwrap_or("no guesser".into()));
    println!("{:16} {:>9} {:>12} {:>11} {:>8} {:>13} {:>10}", "category", "questions", "baseline t/s", "method t/s", "speedup", "mean accepted", "identical");
    let mut all = Vec::new();
    for (cat, v) in &cats {
        let (b, me, tau): (Vec<f64>, Vec<f64>, Vec<f64>) = (v.iter().map(|x| x.0).collect(), v.iter().map(|x| x.1).collect(), v.iter().map(|x| x.2).collect());
        println!("{cat:16} {:>9} {:>12.1} {:>11.1} {:>7.2}x {:>13.2} {:>6}/{}", v.len(), mean(&b), mean(&me), mean(&me) / mean(&b), mean(&tau),
                 v.iter().filter(|x| x.3).count(), v.len());
        all.extend(v.iter().cloned());
    }
    let (b, me, tau): (Vec<f64>, Vec<f64>, Vec<f64>) = (all.iter().map(|x| x.0).collect(), all.iter().map(|x| x.1).collect(), all.iter().map(|x| x.2).collect());
    println!("{:16} {:>9} {:>12.1} {:>11.1} {:>7.2}x {:>13.2} {:>6}/{}", "overall", all.len(), mean(&b), mean(&me), mean(&me) / mean(&b), mean(&tau),
             all.iter().filter(|x| x.3).count(), all.len());
    println!("skipped (too long for {maxlen} context): {skipped} | total {:.0} min | end: swap {} | {}", t_all.elapsed().as_secs_f64() / 60.0, swap(), power());
    Ok(())
}

/// Second turns of two-turn questions: plain decoding from scratch (the reference) vs guess-tree decoding that keeps
/// the first turn's memory. Checks the outputs are identical and reports how much reading was skipped.
fn prefixtest(a: &[String]) -> Result<()> {
    let model = std::path::PathBuf::from(&a[0]);
    let guesser = Some(&a[1]).filter(|s| s.as_str() != "-").map(std::path::PathBuf::from);
    let qs = std::fs::read_to_string(&a[2])?;
    let n_q: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let max_new: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(512);
    let tok = tokenizers::Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let mut env = minijinja::Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template_owned("chat", server::template(&model)?)?;
    let eos: Vec<i32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.token_to_id(t)).map(|x| x as i32).collect();
    let mut m = Model::load(&model, guesser.as_deref(), 6144)?;
    m.autotune()?;
    m.warm()?;
    let enc = |msgs: &[serde_json::Value]| -> Result<Vec<i32>> {
        Ok(tok.encode(server::render(&env, msgs)?, false).map_err(|e| anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect())
    };
    // a conversation: the question's turns plus two follow-ups. Pass 1 talks with memory reuse on; pass 2 replays
    // every prompt of pass 1 from scratch (guess tree, then plain decoding as the reference).
    let extra = ["Thanks. Now summarize everything above in two sentences.", "Give one practical example of that."];
    let (mut same, mut turns_n, mut t_fresh, mut t_reuse, mut read_all, mut read_kept, mut done) = (0, 0, 0f64, 0f64, 0usize, 0usize, 0);
    for line in qs.lines().filter(|l| !l.trim().is_empty()) {
        let q: serde_json::Value = serde_json::from_str(line)?;
        let mut turns: Vec<String> = q["turns"].as_array().unwrap().iter().map(|x| x.as_str().unwrap_or("").to_string()).collect();
        turns.extend(extra.iter().map(|s| s.to_string()));
        let mut msgs: Vec<serde_json::Value> = Vec::new();
        let mut rec = Vec::new();
        m.reset();
        m.reuse = true;
        m.turn_tok = tok.token_to_id("<|im_start|>").map(|x| x as i32);
        for (k, turn) in turns.iter().enumerate() {
            msgs.push(serde_json::json!({"role": "user", "content": turn}));
            let ids = enc(&msgs)?;
            if ids.len() + max_new + 64 > 6144 {
                break;
            }
            let t = Instant::now();
            let mut out = Vec::new();
            m.generate(&ids, max_new, &eos, 8, &mut |w| { out.extend_from_slice(w); true })?;
            let kept = m.kept;
            let tr = t.elapsed().as_secs_f64();
            let ans: Vec<u32> = out.iter().filter(|x| !eos.contains(x)).map(|&x| x as u32).collect();
            msgs.push(serde_json::json!({"role": "assistant", "content": tok.decode(&ans, true).unwrap_or_default()}));
            rec.push((k, ids, kept, out, tr));
        }
        m.reuse = false;
        for (k, ids, kept, out, tr) in rec {
            if k == 0 {
                continue;
            }
            let t = Instant::now();
            let mut out_f = Vec::new();
            m.generate(&ids, max_new, &eos, 8, &mut |w| { out_f.extend_from_slice(w); true })?;
            let tf = t.elapsed().as_secs_f64();
            let (plain, _) = m.plain(&ids, max_new, &eos)?;
            let ok = out == plain && out_f == plain;
            same += ok as usize;
            turns_n += 1;
            t_fresh += tf;
            t_reuse += tr;
            read_all += ids.len();
            read_kept += kept;
            println!("q{} turn {}: prompt {:5} words, {:5} kept | fresh {:5.2}s, reuse {:5.2}s | identical {}", q["question_id"], k + 1, ids.len(), kept, tf, tr, ok);
        }
        done += 1;
        if done == n_q {
            break;
        }
    }
    println!("{turns_n} follow-up turns: {same}/{turns_n} identical to plain | words kept {read_kept}/{read_all} | time fresh {t_fresh:.1}s, reuse {t_reuse:.1}s ({:.0}% less)",
             100.0 * (1.0 - t_reuse / t_fresh));
    Ok(())
}
