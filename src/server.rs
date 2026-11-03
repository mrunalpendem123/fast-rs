//! OpenAI-compatible server for the native engine: `fast-rs serve MODEL [GUESSER] [PORT]`.
//! One engine thread owns the GPU state; each request streams the words of every checked pass as they land.
use crate::Model;
use anyhow::{Context, Result, anyhow};
use minijinja::{Environment, context};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;

enum Ev {
    Text(String),
    Done,
    Err(String),
}

struct Job {
    model: String,
    messages: Vec<Value>,
    max: usize,
    out: Sender<Ev>,
}

pub(crate) fn template(dir: &PathBuf) -> Result<String> {
    if let Ok(s) = std::fs::read_to_string(dir.join("chat_template.jinja")) {
        return Ok(s);
    }
    let tc: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))?)?;
    Ok(tc["chat_template"].as_str().context("no chat template")?.to_string())
}

pub(crate) fn render(env: &Environment, messages: &[Value]) -> Result<String> {
    let msgs: Vec<minijinja::Value> = messages.iter().map(|m| {
        let content = match &m["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""),
            _ => String::new(),
        };
        context! {role => m["role"].as_str().unwrap_or("user").to_string(), content => content}
    }).collect();
    Ok(env.get_template("chat")?.render(context! {messages => msgs, add_generation_prompt => true, enable_thinking => false})?)
}

/// Models this Mac can serve: (id shown in the chat app, model dir, guesser dir). Only pairs that fit 8 GB and
/// were checked exact are listed; `fast-rs serve MODELS_DIR` keeps the ones present on disk.
const CATALOG: &[(&str, &str, &str)] = &[
    ("qwen3-4b", "qwen3-4b-4bit", "qwen3-4b-dflash"),
    ("qwen3.5-4b", "qwen3.5-4b-4bit", "qwen3.5-4b-dflash"),
    ("gemma-4-e2b", "gemma-4-e2b-mlx", "gemma-4-e2b-assistant"),
    ("gemma-4-e4b", "gemma-4-e4b-mlx", "gemma-4-e4b-assistant"),
];

pub(crate) struct Entry {
    pub id: String,
    model: PathBuf,
    guesser: Option<PathBuf>,
}

struct Loaded {
    id: String,
    tok: Tokenizer,
    env: Environment<'static>,
    eos: Vec<i32>,
    m: Model,
}

fn load(e: &Entry) -> Result<Loaded> {
    let t0 = Instant::now();
    let tok = Tokenizer::from_file(e.model.join("tokenizer.json")).map_err(|x| anyhow!("tokenizer: {x}"))?;
    let src = template(&e.model)?;
    let mut env = Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template_owned("chat", src)?;
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(e.model.join("config.json"))?)?;
    let ecfg = if cfg["eos_token_id"].is_null() { &cfg["text_config"]["eos_token_id"] } else { &cfg["eos_token_id"] };
    let mut eos: Vec<i32> = match ecfg {
        Value::Array(v) => v.iter().filter_map(|x| x.as_i64()).map(|x| x as i32).collect(),
        Value::Number(n) => vec![n.as_i64().unwrap() as i32],
        _ => vec![],
    };
    for t in ["<|im_end|>", "<|endoftext|>", "<turn|>", "<end_of_turn>"] {
        if let Some(id) = tok.token_to_id(t) {
            eos.push(id as i32);
        }
    }
    let mut m = Model::load(&e.model, e.guesser.as_deref(), 4096)?;
    m.autotune()?;
    let warm: Vec<i32> = tok.encode(render(&env, &[json!({"role": "user", "content": "Hi"})])?, false).map_err(|x| anyhow!("{x}"))?
        .get_ids().iter().map(|&x| x as i32).collect();
    m.generate(&warm, 8, &eos, budget(), &mut |_| true)?;
    m.reuse = true; // follow-up messages keep the memory of the conversation so far (exact: same words as rereading it)
    m.turn_tok = ["<|im_start|>", "<|turn>", "<start_of_turn>"].iter().find_map(|t| tok.token_to_id(t)).map(|x| x as i32);
    eprintln!("loaded {} in {:.1}s", e.id, t0.elapsed().as_secs_f64());
    Ok(Loaded { id: e.id.clone(), tok, env, eos, m })
}

fn budget() -> usize {
    std::env::var("BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

fn engine(entries: Vec<Entry>, rx: Receiver<Job>, ready: Sender<String>) -> Result<()> {
    let mut cur = Some(load(&entries[0])?);
    let _ = ready.send(entries[0].id.clone());
    while let Ok(job) = rx.recv() {
        let r = (|| -> Result<()> {
            // switch models when the chat asks for another one: free the current one first (8 GB holds one at a time)
            if let Some(e) = entries.iter().find(|e| e.id == job.model) {
                if cur.as_ref().map(|l| l.id != e.id).unwrap_or(true) {
                    cur = None;
                    cur = Some(load(e)?);
                }
            }
            let l = cur.as_mut().context("no model loaded")?;
            let text = render(&l.env, &job.messages)?;
            let ids: Vec<i32> = l.tok.encode(text, false).map_err(|e| anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            let max = job.max.min(4096usize.saturating_sub(ids.len() + 32));
            let mut words: Vec<u32> = Vec::new();
            let mut sent = String::new();
            let t0 = Instant::now();
            let (tok, eos) = (&l.tok, &l.eos);
            let (_, passes) = l.m.generate(&ids, max, eos, budget(), &mut |w| {
                words.extend(w.iter().filter(|x| !eos.contains(x)).map(|&x| x as u32));
                let s = tok.decode(&words, true).unwrap_or_default();
                if s.len() > sent.len() && !s.ends_with('\u{fffd}') {
                    if job.out.send(Ev::Text(s[sent.len()..].to_string())).is_err() {
                        return false;
                    }
                    sent = s;
                }
                true
            })?;
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("{} | {} prompt words | {} words in {dt:.2}s = {:.1} words/s, {:.2} words/check", l.id,
                      ids.len(), words.len(), words.len() as f64 / dt, words.len() as f64 / passes.max(1) as f64);
            Ok(())
        })();
        let _ = job.out.send(match r { Ok(()) => Ev::Done, Err(e) => Ev::Err(e.to_string()) });
    }
    Ok(())
}

/// Server-sent events for one streaming request, read by tiny_http as the response body.
/// Each check lands several words at once; the pacer hands them out one word at a time at the rate they arrive,
/// so the text flows evenly instead of in bursts. It never falls behind: a backlog shortens the gap between words.
struct Sse {
    rx: Receiver<Ev>,
    id: String,
    model: String,
    done: bool,
    pending: std::collections::VecDeque<String>,
    finished: Option<Vec<u8>>,
    first: Option<Instant>,
    arrived: usize,
    last: Instant,
}

impl Sse {
    fn new(rx: Receiver<Ev>, model: String) -> Self {
        Sse { rx, id: "chatcmpl-fast".into(), model, done: false, pending: Default::default(),
              finished: None, first: None, arrived: 0, last: Instant::now() }
    }
    fn chunk(&self, delta: Value, finish: Value) -> Vec<u8> {
        let v = json!({"id": self.id, "object": "chat.completion.chunk", "created": 0, "model": self.model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        format!("data: {v}\n\n").into_bytes()
    }
    /// Split text into word-sized pieces, each carrying its leading whitespace.
    fn push(&mut self, t: &str) {
        let mut cur = String::new();
        let mut seen_char = false;
        for ch in t.chars() {
            if ch.is_whitespace() && seen_char {
                self.pending.push_back(std::mem::take(&mut cur));
                seen_char = false;
            }
            if !ch.is_whitespace() { seen_char = true; }
            cur.push(ch);
        }
        if !cur.is_empty() { self.pending.push_back(cur); }
        self.arrived = self.arrived.max(1);
        if self.first.is_none() { self.first = Some(Instant::now()); }
    }
    fn take(&mut self, ev: Ev) {
        match ev {
            Ev::Text(t) => { self.arrived += t.split_whitespace().count().max(1); self.push(&t); }
            Ev::Err(e) => {
                let mut b = self.chunk(json!({"content": format!("\n[engine error: {e}]")}), json!("stop"));
                b.extend(b"data: [DONE]\n\n");
                self.finished = Some(b);
            }
            Ev::Done => {
                let mut b = self.chunk(json!({}), json!("stop"));
                b.extend(b"data: [DONE]\n\n");
                self.finished = Some(b);
            }
        }
    }
    /// Gap between words: the arrival rate so far, shortened when words pile up.
    fn gap(&self) -> f64 {
        let rate = match self.first {
            Some(f) if self.arrived > 4 => self.arrived as f64 / f.elapsed().as_secs_f64().max(0.05),
            _ => 60.0,
        };
        let backlog = self.pending.len() as f64;
        let g = (1.0 / rate.max(5.0)) / (1.0 + backlog / 6.0);
        if self.finished.is_some() { g.min(0.008) } else { g.min(0.06) }
    }
}

impl Sse {
    /// The next piece of the event stream, paced; None when the answer is complete.
    fn next_chunk(&mut self) -> Option<Vec<u8>> {
        if self.done { return None; }
        loop {
            while let Ok(ev) = self.rx.try_recv() { self.take(ev); }
            if !self.pending.is_empty() {
                let due = self.last + std::time::Duration::from_secs_f64(self.gap());
                let now = Instant::now();
                if due > now { std::thread::sleep(due - now); }
                let w = self.pending.pop_front().unwrap();
                self.last = Instant::now();
                return Some(self.chunk(json!({"content": w}), Value::Null));
            } else if let Some(b) = self.finished.take() {
                self.done = true;
                return Some(b);
            } else {
                match self.rx.recv() {
                    Ok(ev) => self.take(ev),
                    Err(_) => self.take(Ev::Done),
                }
            }
        }
    }
}

fn json_resp(req: tiny_http::Request, v: Value, code: u16) {
    let h = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    let _ = req.respond(tiny_http::Response::from_string(v.to_string()).with_status_code(code).with_header(h));
}

pub fn run(model: String, guesser: Option<String>, port: u16) -> Result<()> {
    let p = PathBuf::from(&model);
    let entries: Vec<Entry> = if p.join("config.json").exists() {
        vec![Entry { id: p.file_name().unwrap().to_string_lossy().to_string(), model: p, guesser: guesser.map(PathBuf::from) }]
    } else {
        // a folder of models: serve every catalog pair found there, the first one loaded at start
        let mut v: Vec<Entry> = CATALOG.iter().filter(|(_, m, _)| p.join(m).join("config.json").exists())
            .map(|(id, m, g)| Entry { id: id.to_string(), model: p.join(m), guesser: Some(p.join(g)).filter(|g| g.exists()) }).collect();
        if let Some(first) = guesser.as_deref().and_then(|f| v.iter().position(|e| e.id == f)) { v.swap(0, first); }
        v
    };
    anyhow::ensure!(!entries.is_empty(), "no models found in {model}");
    let ids: Vec<String> = entries.iter().map(|e| e.id.clone()).collect();
    let (jtx, jrx) = channel::<Job>();
    let (rtx, rrx) = channel::<String>();
    std::thread::spawn(move || {
        if let Err(e) = engine(entries, jrx, rtx) {
            eprintln!("engine failed: {e:#}");
            std::process::exit(1);
        }
    });
    let id = rrx.recv().context("engine did not start")?;
    let jtx = Arc::new(Mutex::new(jtx));
    let server = tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| anyhow!("{e}"))?;
    eprintln!("ready: {} on http://127.0.0.1:{port}/v1 (loaded: {id})", ids.join(", "));
    for mut req in server.incoming_requests() {
        let path = req.url().split('?').next().unwrap_or("").trim_end_matches('/').to_string();
        match (req.method().as_str(), path.as_str()) {
            ("GET", "/v1/models") | ("GET", "/models") => {
                json_resp(req, json!({"object": "list", "data": ids.iter().map(|i| json!({"id": i, "object": "model", "owned_by": "local", "created": 0})).collect::<Vec<_>>()}), 200)
            }
            ("POST", p) if p.ends_with("/chat/completions") => {
                let mut body = String::new();
                let _ = req.as_reader().read_to_string(&mut body);
                let b: Value = serde_json::from_str(&body).unwrap_or(json!({}));
                let (otx, orx) = channel();
                let want = b["model"].as_str().unwrap_or(&id).to_string();
                let job = Job {
                    model: want.clone(),
                    messages: b["messages"].as_array().cloned().unwrap_or_default(),
                    max: b["max_tokens"].as_u64().or(b["max_completion_tokens"].as_u64()).unwrap_or(1024) as usize,
                    out: otx,
                };
                let _ = jtx.lock().unwrap().send(job);
                let (id2, stream) = (want, b["stream"].as_bool().unwrap_or(false));
                std::thread::spawn(move || {
                    if stream {
                        // write the stream straight to the socket and flush every word (a buffered response arrives in bursts)
                        let mut sse = Sse::new(orx, id2);
                        let mut w = req.into_writer();
                        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                        if w.write_all(head.as_bytes()).and_then(|_| w.flush()).is_err() { return; }
                        while let Some(c) = sse.next_chunk() {
                            if w.write_all(&c).and_then(|_| w.flush()).is_err() { break; } // client left: dropping sse stops the engine
                        }
                    } else {
                        let mut text = String::new();
                        let mut err = None;
                        for ev in orx {
                            match ev {
                                Ev::Text(t) => text += &t,
                                Ev::Err(e) => err = Some(e),
                                Ev::Done => break,
                            }
                        }
                        match err {
                            Some(e) => json_resp(req, json!({"error": e}), 500),
                            None => json_resp(req, json!({"id": "chatcmpl-fast", "object": "chat.completion", "created": 0, "model": id2,
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}), 200),
                        }
                    }
                });
            }
            _ => json_resp(req, json!({"error": "not found"}), 404),
        }
    }
    Ok(())
}
