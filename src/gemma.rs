//! Gemma 4 (E2B / E4B): sliding (head 256) and full (head 512, proportional RoPE) attention, the last layers reading
//! an earlier layer's memory, sandwich norms, GELU, per-layer inputs, layer scalars. Guesser: Google's Gemma 4
//! assistant, a few small layers that read the target's memory and write one word per tree branch per round.
use crate::*;

/// Model-wide Gemma 4 settings.
#[derive(Clone)]
pub(crate) struct G4Cfg {
    pub ple: usize,        // per-layer input size (256)
    pub nl: usize,         // layers
    pub window: usize,     // sliding window
    pub ghd: usize,        // full-attention head size
    pub shared: usize,     // last layers that read an earlier layer's memory
    pub theta_s: f32,
    pub theta_f: f32,
    pub rot_f: f32,        // full-attention partial rotary factor
    pub sliding: Vec<bool>,
}

pub(crate) fn read_g4(c: &serde_json::Value) -> Option<G4Cfg> {
    if c["model_type"].as_str() != Some("gemma4_text") {
        return None;
    }
    let rp = &c["rope_parameters"];
    let n = c["num_hidden_layers"].as_u64()? as usize;
    Some(G4Cfg {
        ple: c["hidden_size_per_layer_input"].as_u64().unwrap_or(0) as usize,
        nl: n,
        window: c["sliding_window"].as_u64().unwrap_or(512) as usize,
        ghd: c["global_head_dim"].as_u64().unwrap_or(512) as usize,
        shared: c["num_kv_shared_layers"].as_u64().unwrap_or(0) as usize,
        theta_s: rp["sliding_attention"]["rope_theta"].as_f64().unwrap_or(1e4) as f32,
        theta_f: rp["full_attention"]["rope_theta"].as_f64().unwrap_or(1e6) as f32,
        rot_f: rp["full_attention"]["partial_rotary_factor"].as_f64().unwrap_or(0.25) as f32,
        sliding: (0..n).map(|i| c["layer_types"][i].as_str() != Some("full_attention")).collect(),
    })
}

/// Attention shape of a Gemma 4 layer kind (heads, sliding?), reading memory of layer `src`.
pub(crate) fn g4_at(g: &G4Cfg, hd_s: usize, h: usize, kvh: usize, sliding: bool, kvh_w: usize, src: usize) -> At {
    let hd = if sliding { hd_s } else { g.ghd };
    let rot = if sliding { hd } else { (hd as f32 * g.rot_f) as usize };
    At { trad: false, h, hd, kvh_w, kvh, src, w: if sliding { g.window } else { 0 }, theta: if sliding { g.theta_s } else { g.theta_f },
         rot, ph: hd / 2, ed: hd, scale: 1.0, vnorm: true, qs: hd, gate: false }
}

fn bf16_scalar(w: &Weights, name: &str) -> Result<f32> {
    Ok(bf16_to_f32(w.u16s(name)?[0]))
}

pub(crate) fn gemma_layer(gpu: &Gpu, w: &Weights, cfg: &Cfg, i: usize, p: &str, maxlen: usize) -> Result<Layer> {
    let g = cfg.gemma.as_ref().unwrap();
    let a = format!("{p}self_attn.");
    let m = format!("{p}mlp.");
    let first_shared = g.nl - g.shared;
    let own = i < first_shared;
    // a shared layer reads the last own-memory layer of its kind
    let src = if own { i } else { (0..first_shared).rev().find(|&j| g.sliding[j] == g.sliding[i]).context("no memory layer")? };
    let at = g4_at(g, cfg.hd, cfg.h, cfg.kvh, g.sliding[i], if own { cfg.kvh } else { 0 }, src);
    let kvbytes = cfg.kvh * maxlen * at.hd * 2;
    let qkv = if own {
        repack(gpu, &[w.q4(&format!("{a}q_proj"))?, w.q4(&format!("{a}k_proj"))?, w.q4(&format!("{a}v_proj"))?], cfg.d)
    } else {
        repack(gpu, &[w.q4(&format!("{a}q_proj"))?], cfg.d)
    };
    let down = w.q4(&format!("{m}down_proj"))?;
    let inter = w.shape(&format!("{m}down_proj.weight"))[1] * 8;
    let ple = if g.ple > 0 {
        Some((repack(gpu, &[w.q4(&format!("{p}per_layer_input_gate"))?], cfg.d),
              repack(gpu, &[w.q4(&format!("{p}per_layer_projection"))?], g.ple),
              gpu.buf_from(w.u16s(&format!("{p}post_per_layer_input_norm.weight"))?)))
    } else {
        None
    };
    Ok(Layer {
        sc: None,
        qkv,
        kv: None,
        o: repack(gpu, &[w.q4(&format!("{a}o_proj"))?], cfg.h * at.hd),
        gu: repack(gpu, &[w.q4(&format!("{m}gate_proj"))?, w.q4(&format!("{m}up_proj"))?], cfg.d),
        down: repack(gpu, &[down], inter),
        in_n: gpu.buf_from(w.u16s(&format!("{p}input_layernorm.weight"))?),
        post_n: gpu.buf_from(w.u16s(&format!("{p}pre_feedforward_layernorm.weight"))?),
        qn: gpu.buf_from(w.u16s(&format!("{a}q_norm.weight"))?),
        kn: if own { gpu.buf_from(w.u16s(&format!("{a}k_norm.weight"))?) } else { gpu.buf(16) },
        kc: if own { gpu.buf(kvbytes) } else { gpu.buf(16) },
        vc: if own { gpu.buf(kvbytes) } else { gpu.buf(16) },
        lin: None,
        at,
        g4: Some(G4L {
            pa_n: gpu.buf_from(w.u16s(&format!("{p}post_attention_layernorm.weight"))?),
            po_n: gpu.buf_from(w.u16s(&format!("{p}post_feedforward_layernorm.weight"))?),
            ple,
            scalar: bf16_scalar(w, &format!("{p}layer_scalar"))?,
        }),
    })
}

/// Model-level Gemma 4 pieces: the per-layer input table (stays in the file), its projection and the pass buffers.
pub(crate) struct G4M {
    pub ple_name: String,  // embed_tokens_per_layer in the checkpoint
    pub proj: QW,          // per_layer_model_projection: d -> nl*ple
    pub norm: Buf,         // per_layer_projection_norm
    pub pw: Buf,           // gathered rows of the table (4-bit), scales, biases
    pub ps: Buf,
    pub pb: Buf,
    pub raw: Buf,          // [rows][nl*ple]: the rows' own per-layer embeddings
    pub pp: Buf,           // [rows][nl*ple]: projection of the embedding
    pub pl: Buf,           // [rows][nl*ple]: mixed per-layer inputs
}

impl G4M {
    pub(crate) fn load(gpu: &Gpu, w: &Weights, lm: &str, cfg: &Cfg, r: usize) -> Result<G4M> {
        let g = cfg.gemma.as_ref().unwrap();
        let npl = g.nl * g.ple;
        Ok(G4M {
            ple_name: format!("{lm}model.embed_tokens_per_layer"),
            proj: repack(gpu, &[w.q4(&format!("{lm}model.per_layer_model_projection"))?], cfg.d),
            norm: gpu.buf_from(w.u16s(&format!("{lm}model.per_layer_projection_norm.weight"))?),
            pw: gpu.buf(r * npl / 2),
            ps: gpu.buf(r * npl / 64 * 2),
            pb: gpu.buf(r * npl / 64 * 2),
            raw: gpu.buf(r * npl * 4),
            pp: gpu.buf(r * npl * 4),
            pl: gpu.buf(r * npl * 4),
        })
    }
}

/// Google's Gemma 4 assistant: [target embedding | target hidden] -> pre -> 4 layers over the target's memory ->
/// norm -> sparse head (the top-32 of 2048 word clusters, their words scored exactly) and post -> the hidden its
/// children start from. All on the GPU in one pass.
pub(crate) struct Asst {
    pub cfg: Cfg,
    pub pre: QW,
    pub layers: Vec<Layer>,
    pub norm: Buf,
    pub post: QW,            // d -> d_target
    pub lm: Buf,             // its word table, bf16 [vocab][d]
    pub cent: Buf,           // bf16 [nc][d]
    pub order: Buf,          // i32 [nc][per]
    pub nc: usize,
    pub topc: usize,
    pub ain: Buf,            // [rows][2*d_target] input
    pub nh: Buf,             // [rows][d_target] next hidden
    pub lg: Buf,             // [rows][topc*per] candidate scores and words
    pub ids: Buf,
}

impl Asst {
    pub(crate) fn load(gpu: &Gpu, dir: &Path, target: &Stack) -> Result<Asst> {
        let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let tc = &c["text_config"];
        let mut cfg = read_cfg(tc);
        let tg = target.cfg.gemma.as_ref().context("assistant needs a Gemma 4 target")?;
        let g = read_g4(tc).context("assistant config")?;
        cfg.gemma = Some(G4Cfg { ple: 0, ..g.clone() });
        let w = Weights::open(dir)?;
        let bd = c["backbone_hidden_size"].as_u64().context("backbone_hidden_size")? as usize;
        // the target's last own-memory layer of each kind
        let first_shared = tg.nl - tg.shared;
        let src = |sl: bool| (0..first_shared).rev().find(|&j| tg.sliding[j] == sl).unwrap();
        let mut layers = Vec::new();
        for i in 0..g.nl {
            let p = format!("model.layers.{i}.");
            let a = format!("{p}self_attn.");
            let m = format!("{p}mlp.");
            let sl = g.sliding[i];
            let at = g4_at(&g, cfg.hd, cfg.h, target.layers[src(sl)].at.kvh_w, sl, 0, src(sl));
            let q4 = |n: &str| -> Result<Q4<'static>> { let sh = w.shape(&format!("{n}.weight")); Ok(quant4(w.u16s(&format!("{n}.weight"))?, sh[0], sh[1])) };
            layers.push(Layer {
                sc: None,
                qkv: repack(gpu, &[q4(&format!("{a}q_proj"))?], cfg.d),
                kv: None,
                o: repack(gpu, &[q4(&format!("{a}o_proj"))?], cfg.h * at.hd),
                gu: repack(gpu, &[q4(&format!("{m}gate_proj"))?, q4(&format!("{m}up_proj"))?], cfg.d),
                down: repack(gpu, &[q4(&format!("{m}down_proj"))?], cfg.inter),
                in_n: gpu.buf_from(w.u16s(&format!("{p}input_layernorm.weight"))?),
                post_n: gpu.buf_from(w.u16s(&format!("{p}pre_feedforward_layernorm.weight"))?),
                qn: gpu.buf_from(w.u16s(&format!("{a}q_norm.weight"))?),
                kn: gpu.buf(16),
                kc: gpu.buf(16),
                vc: gpu.buf(16),
                lin: None,
                at,
                g4: Some(G4L {
                    pa_n: gpu.buf_from(w.u16s(&format!("{p}post_attention_layernorm.weight"))?),
                    po_n: gpu.buf_from(w.u16s(&format!("{p}post_feedforward_layernorm.weight"))?),
                    ple: None,
                    scalar: bf16_scalar(&w, &format!("{p}layer_scalar"))?,
                }),
            });
        }
        let ord = w.raw("masked_embedding.token_ordering")?;
        let order: Vec<i32> = ord.chunks(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as i32).collect();
        let nc = c["num_centroids"].as_u64().context("num_centroids")? as usize;
        let topc = c["centroid_intermediate_top_k"].as_u64().unwrap_or(32) as usize;
        let per = order.len() / nc;
        let q4m = |n: &str| -> Result<QW> { let sh = w.shape(n); Ok(repack(gpu, &[quant4(w.u16s(n)?, sh[0], sh[1])], sh[1])) };
        Ok(Asst {
            pre: q4m("pre_projection.weight")?,
            layers,
            norm: gpu.buf_from(w.u16s("model.norm.weight")?),
            post: q4m("post_projection.weight")?,
            lm: gpu.buf_from(w.u16s("model.embed_tokens.weight")?),
            cent: gpu.buf_from(w.u16s("masked_embedding.centroids.weight")?),
            order: gpu.buf_from(&order),
            nc,
            topc,
            ain: gpu.buf(MAXR * 2 * bd * 4),
            nh: gpu.buf(MAXR * bd * 4),
            lg: gpu.buf(16 * topc * per * 4),
            ids: gpu.buf(16 * topc * per * 4),
            cfg,
        })
    }
}

fn rms_cpu(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * w).collect()
}

impl Model {
    /// Gemma 4 layer over the n rows in self.h (input norm already in xh). Attention reads memory `kv`.
    pub(crate) fn enc_gemma_layer(&mut self, e: &Enc, c: &Cfg, l: &Layer, kv: (&Buf, &Buf), li: usize, n: usize, n_real: usize,
                                  p0: usize, nb: usize) -> Result<()> {
        let g = l.g4.as_ref().unwrap();
        let b = [self.h.clone(), self.xh.clone(), self.xo.clone(), self.xs.clone(), self.qkv.clone(), self.gu.clone(), self.tmp.clone(),
                 self.tmp2.clone(), self.pos.clone()];
        let [h, xh, xo, xs, qkv, gu, tmp, tmp2, pos] = &b;
        self.enc_qmm(e, &l.qkv, [xh, xo, xs], qkv, n, false)?;
        self.enc_rope(e, c, l.at.h, qkv, l, n_real, p0, n, pos)?;
        self.enc_attn(e, &l.at, kv, n, nb)?;
        self.enc_qmm(e, &l.o, [xh, xo, xs], tmp, n, false)?;
        self.enc_add_norm(e, c, h, tmp, &g.pa_n, 1.0, n)?;
        self.enc_norm(e, c.d, c.eps, h, &l.post_n, n)?;
        self.enc_qmm(e, &l.gu, [xh, xo, xs], gu, n, false)?;
        let p = self.gpu.pipe("swiglu_prep", kernels::SWIGLU_PREP, &[("I", l.down.k.to_string()), ("GELU", "1".into())])?;
        e.setComputePipelineState(&p);
        bind(e, &[gu, xh, xo, xs, &self.rscale]);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n), sz(256));
        self.rs_next = true;
        self.enc_qmm(e, &l.down, [xh, xo, xs], tmp, n, false)?;
        match &g.ple {
            None => self.enc_add_norm(e, c, h, tmp, &g.po_n, g.scalar, n),
            Some((gate, proj, pn)) => {
                self.enc_add_norm(e, c, h, tmp, &g.po_n, 1.0, n)?;
                self.enc_prep_scaled(e, h, c.d, n)?;
                self.rs_next = true;
                self.enc_qmm(e, gate, [xh, xo, xs], tmp2, n, false)?;
                let gm = self.g4m.as_ref().unwrap().pl.clone();
                let ple = c.gemma.as_ref().unwrap().ple;
                let p = self.gpu.pipe("gelu_mul_prep", kernels::GELU_MUL_PREP, &[("PLE", ple.to_string()),
                    ("NLP", (c.gemma.as_ref().unwrap().nl * ple).to_string())])?;
                e.setComputePipelineState(&p);
                bind(e, &[tmp2, &gm, xh, xo, xs, xs, &self.rscale]);
                set_args(e, &[li as i32], 5);
                e.dispatchThreadgroups_threadsPerThreadgroup(sz(n), sz(256));
                self.rs_next = true;
                self.enc_qmm(e, proj, [xh, xo, xs], tmp, n, false)?;
                self.enc_add_norm(e, c, h, tmp, pn, g.scalar, n)
            }
        }
    }

    fn enc_add_norm(&mut self, e: &Enc, c: &Cfg, h: &Buf, t: &Buf, w: &Buf, scalar: f32, n: usize) -> Result<()> {
        let p = self.gpu.pipe("add_norm", kernels::ADD_NORM, &[("D", c.d.to_string()), ("EPS", format!("{:e}f", c.eps))])?;
        e.setComputePipelineState(&p);
        bind(e, &[h, t, w]);
        unsafe { e.setBytes_length_atIndex(NonNull::new(&scalar as *const f32 as *mut c_void).unwrap(), 4, 3) };
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n), sz(256));
        Ok(())
    }

    /// float rows [n][kd] -> xh/xo/xs (no norm).
    pub(crate) fn enc_prep(&mut self, e: &Enc, x: &Buf, kd: usize, n: usize) -> Result<()> {
        self.enc_prep_with(e, x, kd, n, false)
    }

    /// As enc_prep, scaling rows past half's range (the next matmul must multiply back: rs_next).
    fn enc_prep_scaled(&mut self, e: &Enc, x: &Buf, kd: usize, n: usize) -> Result<()> {
        self.enc_prep_with(e, x, kd, n, true)
    }

    fn enc_prep_with(&mut self, e: &Enc, x: &Buf, kd: usize, n: usize, scaled: bool) -> Result<()> {
        let p = self.gpu.pipe("prep", kernels::PREP, &[("KD", kd.to_string()), ("RSC", (scaled as i32).to_string())])?;
        e.setComputePipelineState(&p);
        bind(e, &[x, &self.xh, &self.xo, &self.xs, &self.rscale]);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n), sz(256));
        Ok(())
    }

    /// Per-layer inputs of the n rows (ids in self.ids, embeddings already in self.h) -> g4m.pl.
    pub(crate) fn enc_ple(&mut self, e: &Enc, n: usize) -> Result<()> {
        let c = self.t.cfg.clone();
        let g = c.gemma.as_ref().unwrap();
        let npl = g.nl * g.ple;
        let gm = self.g4m.as_ref().unwrap();
        let ids: Vec<i32> = read(&self.ids, n);
        let nm = &gm.ple_name;
        let (tw, ts, tb) = (self.emb.u32s(&format!("{nm}.weight"))?, self.emb.u16s(&format!("{nm}.scales"))?, self.emb.u16s(&format!("{nm}.biases"))?);
        let (rw, rs) = (npl / 8, npl / 64);
        write(&gm.pw, &ids.iter().flat_map(|&i| tw[i as usize * rw..(i as usize + 1) * rw].iter().copied()).collect::<Vec<u32>>());
        write(&gm.ps, &ids.iter().flat_map(|&i| ts[i as usize * rs..(i as usize + 1) * rs].iter().copied()).collect::<Vec<u16>>());
        write(&gm.pb, &ids.iter().flat_map(|&i| tb[i as usize * rs..(i as usize + 1) * rs].iter().copied()).collect::<Vec<u16>>());
        let b = [gm.pw.clone(), gm.ps.clone(), gm.pb.clone(), gm.raw.clone(), gm.pp.clone(), gm.pl.clone(), gm.norm.clone(), self.emb_ids.clone(), self.h.clone()];
        let [pw, ps, pb, raw, pp, pl, norm, ident, h] = &b;
        let p = self.gpu.pipe("embed", kernels::EMBED, &[("D", npl.to_string()), ("ESCALE", format!("{:e}f", (g.ple as f32).sqrt()))])?;
        e.setComputePipelineState(&p);
        bind(e, &[ident, pw, ps, pb, raw]);
        e.dispatchThreads_threadsPerThreadgroup(MTLSize { width: npl, height: n, depth: 1 }, sz(256));
        self.enc_prep(e, h, c.d, n)?;
        let proj = unsafe { &*(&self.g4m.as_ref().unwrap().proj as *const QW) };
        let (xh, xo, xs) = (self.xh.clone(), self.xo.clone(), self.xs.clone());
        self.enc_qmm(e, proj, [&xh, &xo, &xs], pp, n, false)?;
        let p = self.gpu.pipe("ple_mix", kernels::PLE_MIX, &[("NL", g.nl.to_string()), ("PLE", g.ple.to_string()),
            ("EPS", format!("{:e}f", c.eps)), ("PSCALE", format!("{:e}f", (c.d as f32).powf(-0.5)))])?;
        e.setComputePipelineState(&p);
        bind(e, &[pp, raw, norm, pl]);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n * g.nl), sz(32));
        Ok(())
    }

    /// The target's final normed hidden for row r of the last pass (the assistant's input).
    pub(crate) fn hidden_row(&self, r: usize) -> Vec<f32> {
        let d = self.t.cfg.d;
        let w: Vec<f32> = read::<u16>(&self.t.norm, d).iter().map(|&x| bf16_to_f32(x)).collect();
        rms_cpu(&self.hrows[r * d..(r + 1) * d], &w, self.t.cfg.eps)
    }

    /// One assistant round: rows (word, hidden) -> per row its top-2 (log-prob, word) and the hidden for its children.
    fn asst_step(&mut self, toks: &[i32], hids: &[Vec<f32>]) -> Result<(Vec<[(f32, i32); 2]>, Vec<Vec<f32>>)> {
        let a = unsafe { &*(self.asst.as_ref().unwrap() as *const Asst) };
        let bd = self.t.cfg.d;
        let ad = a.cfg.d;
        let n_real = toks.len();
        let p = self.p as i32;
        let n = self.write_rows(toks, &vec![p; n_real], &vec![[p, p, 0, 0]; n_real], &vec![[0; MAXD]; n_real]);
        // input rows: [target embedding * sqrt(d) | target hidden]
        let ep = &self.epre;
        let (tw, ts, tb) = (self.emb.u32s(&format!("{ep}.weight"))?, self.emb.u16s(&format!("{ep}.scales"))?, self.emb.u16s(&format!("{ep}.biases"))?);
        let sc = (bd as f32).sqrt();
        let mut x = vec![0f32; n * 2 * bd];
        for r in 0..n {
            let rr = r.min(n_real - 1);
            let t = toks[rr] as usize;
            for c in 0..bd {
                let q = (tw[t * bd / 8 + c / 8] >> (4 * (c % 8))) & 0xF;
                let v = bf16_to_f32(ts[t * bd / 64 + c / 64]) * q as f32 + bf16_to_f32(tb[t * bd / 64 + c / 64]);
                x[r * 2 * bd + c] = half::bf16::from_f32(v).to_f32() * sc;
            }
            x[r * 2 * bd + bd..(r + 1) * 2 * bd].copy_from_slice(&hids[rr]);
        }
        write(&a.ain, &x);
        let cb = self.gpu.queue.commandBuffer().context("cb")?;
        let e = cb.computeCommandEncoder().context("enc")?;
        self.enc_prep(&e, &a.ain, 2 * bd, n)?;
        let (xh, xo, xs, h) = (self.xh.clone(), self.xo.clone(), self.xs.clone(), self.h.clone());
        self.enc_qmm(&e, &a.pre, [&xh, &xo, &xs], &h, n, false)?;
        let nb = (self.p.min(self.maxlen) + MAXD).div_ceil(BS);
        for (li, l) in a.layers.iter().enumerate() {
            self.enc_norm(&e, ad, a.cfg.eps, &h, &l.in_n, n)?;
            let src = &self.t.layers[l.at.src];
            let kv = (src.kc.clone(), src.vc.clone());
            self.enc_gemma_layer(&e, &a.cfg, l, (&kv.0, &kv.1), li, n, 0, self.p, nb)?;
        }
        // head: final norm, every word's score, top 3 with log-probs; the hidden for children
        self.enc_norm(&e, ad, a.cfg.eps, &h, &a.norm, n)?;
        let (nh, tv, ti) = (a.nh.clone(), self.tv.clone(), self.ti.clone());
        let per = (a.order.length() as usize / 4) / a.nc;
        let ncand = a.topc * per;
        let p_head = self.gpu.pipe("asst_head", kernels::ASST_HEAD, &[("AD", ad.to_string()), ("NC", a.nc.to_string()),
            ("PER", per.to_string()), ("TOPC", a.topc.to_string())])?;
        e.setComputePipelineState(&p_head);
        bind(&e, &[&xh, &a.cent, &a.order, &a.lm, &a.lg, &a.ids]);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n_real), sz(256));
        self.enc_qmm(&e, &a.post, [&xh, &xo, &xs], &nh, n, false)?;
        let p_top = self.gpu.pipe("topk3", kernels::TOPK3, &[("V", ncand.to_string())])?;
        e.setComputePipelineState(&p_top);
        bind(&e, &[&a.lg, &tv, &ti]);
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(n_real), sz(256));
        e.endEncoding();
        let tg = Instant::now();
        cb.commit();
        cb.waitUntilCompleted();
        self.prof[0] += tg.elapsed().as_secs_f64();
        self.prof[2] += 1.0;
        let v: Vec<f32> = read(&self.tv, n_real * 3);
        let pos: Vec<i32> = read(&self.ti, n_real * 3);
        let cid: Vec<i32> = read(&a.ids, n_real * ncand);
        let t: Vec<i32> = (0..n_real * 3).map(|k| cid[(k / 3) * ncand + pos[k] as usize]).collect();
        let nhv: Vec<f32> = read(&a.nh, n_real * bd);
        let tops = (0..n_real).map(|r| [(v[r * 3], t[r * 3]), (v[r * 3 + 1], t[r * 3 + 1])]).collect();
        let nhs = (0..n_real).map(|r| nhv[r * bd..(r + 1) * bd].to_vec()).collect();
        Ok((tops, nhs))
    }

    /// Best-first tree from the assistant: a word joins only if its whole path is among the most probable ones; each
    /// round expands up to 2 nodes (one assistant pass for both).
    pub(crate) fn build_tree_asst(&mut self, root: i32, budget: usize) -> Result<(Vec<i32>, Vec<i32>)> {
        #[derive(PartialEq)]
        struct C(f32, usize, usize, i32, usize); // score, seq, parent, word, hidden index
        impl Eq for C {}
        impl PartialOrd for C { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
        impl Ord for C { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.0.total_cmp(&o.0).then(o.1.cmp(&self.1)) } }
        let (mut toks, mut parent, mut logp, mut dep) = (vec![root], vec![-1i32], vec![0f32], vec![0usize]);
        let mut hs: Vec<Vec<f32>> = vec![self.last_hn.clone()];
        let mut ex: Vec<(usize, usize)> = vec![(0, 0)]; // (node, hidden index)
        let mut heap = std::collections::BinaryHeap::new();
        let mut seq = 0;
        loop {
            if !ex.is_empty() {
                let (tops, nhs) = self.asst_step(&ex.iter().map(|&(i, _)| toks[i]).collect::<Vec<_>>(),
                                                 &ex.iter().map(|&(_, hi)| hs[hi].clone()).collect::<Vec<_>>())?;
                for (r, &(i, _)) in ex.iter().enumerate() {
                    hs.push(nhs[r].clone());
                    let hi = hs.len() - 1;
                    for &(l, t) in &tops[r] {
                        heap.push(C(logp[i] + l, seq, i, t, hi));
                        seq += 1;
                    }
                }
            }
            if heap.is_empty() || toks.len() >= budget {
                break;
            }
            ex.clear();
            while ex.len() < 2 && toks.len() < budget {
                let Some(C(s, _, p, t, hi)) = heap.pop() else { break };
                toks.push(t);
                parent.push(p as i32);
                logp.push(s);
                dep.push(dep[p] + 1);
                if dep[toks.len() - 1] < MAXD - 1 {
                    ex.push((toks.len() - 1, hi));
                }
            }
        }
        let mut order: Vec<usize> = (0..toks.len()).collect();
        order.sort_by_key(|&i| (dep[i], i));
        let mut new = vec![0usize; toks.len()];
        for (j, &o) in order.iter().enumerate() {
            new[o] = j;
        }
        Ok((order.iter().map(|&o| toks[o]).collect(), order.iter().map(|&o| if parent[o] >= 0 { new[parent[o] as usize] as i32 } else { -1 }).collect()))
    }
}
