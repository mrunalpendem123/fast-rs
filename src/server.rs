//! OpenAI-compatible server for the native engine: `fast-rs serve MODEL [GUESSER] [PORT]`.
//! One engine thread owns the GPU state; each request streams the words of every checked pass as they land.
use crate::Model;
use anyhow::{Context, Result, anyhow};
use minijinja::{Environment, context};
use serde_json::{Value, json};
use std::io::Read;
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

fn engine(model: PathBuf, guesser: Option<PathBuf>, rx: Receiver<Job>, ready: Sender<String>) -> Result<()> {
    let tok = Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let src = template(&model)?;
    let mut env = Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template_owned("chat", src)?;
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(model.join("config.json"))?)?;
    let mut eos: Vec<i32> = match &cfg["eos_token_id"] {
        Value::Array(v) => v.iter().filter_map(|x| x.as_i64()).map(|x| x as i32).collect(),
        Value::Number(n) => vec![n.as_i64().unwrap() as i32],
        _ => vec![],
    };
    for t in ["<|im_end|>", "<|endoftext|>", "<turn|>", "<end_of_turn>"] {
        if let Some(id) = tok.token_to_id(t) {
            eos.push(id as i32);
        }
    }
    let mut m = Model::load(&model, guesser.as_deref(), 4096)?;
    m.autotune()?;
    let budget: usize = std::env::var("BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    // warm up every kernel shape once
    let warm: Vec<i32> = tok.encode(render(&env, &[json!({"role": "user", "content": "Hi"})])?, false).map_err(|e| anyhow!("{e}"))?
        .get_ids().iter().map(|&x| x as i32).collect();
    m.generate(&warm, 8, &eos, budget, &mut |_| true)?;
    m.reuse = true; // follow-up messages keep the memory of the conversation so far (exact: same words as rereading it)
    let _ = ready.send(model.file_name().unwrap().to_string_lossy().to_string());
    while let Ok(job) = rx.recv() {
        let r = (|| -> Result<()> {
            let text = render(&env, &job.messages)?;
            let ids: Vec<i32> = tok.encode(text, false).map_err(|e| anyhow!("{e}"))?.get_ids().iter().map(|&x| x as i32).collect();
            let max = job.max.min(4096 - ids.len() - 32);
            let mut words: Vec<u32> = Vec::new();
            let mut sent = String::new();
            let t0 = Instant::now();
            let (_, passes) = m.generate(&ids, max, &eos, budget, &mut |w| {
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
            eprintln!("{} prompt words | {} words in {dt:.2}s = {:.1} words/s, {:.2} words/check",
                      ids.len(), words.len(), words.len() as f64 / dt, words.len() as f64 / passes.max(1) as f64);
            Ok(())
        })();
        let _ = job.out.send(match r { Ok(()) => Ev::Done, Err(e) => Ev::Err(e.to_string()) });
    }
    Ok(())
}

/// Server-sent events for one streaming request, read by tiny_http as the response body.
struct Sse {
    rx: Receiver<Ev>,
    buf: Vec<u8>,
    id: String,
    model: String,
    done: bool,
}

impl Sse {
    fn chunk(&self, delta: Value, finish: Value) -> Vec<u8> {
        let v = json!({"id": self.id, "object": "chat.completion.chunk", "created": 0, "model": self.model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        format!("data: {v}\n\n").into_bytes()
    }
}

impl Read for Sse {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        while self.buf.is_empty() && !self.done {
            match self.rx.recv() {
                Ok(Ev::Text(t)) => self.buf = self.chunk(json!({"content": t}), Value::Null),
                Ok(Ev::Err(e)) => {
                    self.buf = self.chunk(json!({"content": format!("\n[engine error: {e}]")}), json!("stop"));
                    self.buf.extend(b"data: [DONE]\n\n");
                    self.done = true;
                }
                Ok(Ev::Done) | Err(_) => {
                    self.buf = self.chunk(json!({}), json!("stop"));
                    self.buf.extend(b"data: [DONE]\n\n");
                    self.done = true;
                }
            }
        }
        let n = self.buf.len().min(out.len());
        out[..n].copy_from_slice(&self.buf[..n]);
        self.buf.drain(..n);
        Ok(n)
    }
}

fn json_resp(req: tiny_http::Request, v: Value, code: u16) {
    let h = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    let _ = req.respond(tiny_http::Response::from_string(v.to_string()).with_status_code(code).with_header(h));
}

pub fn run(model: String, guesser: Option<String>, port: u16) -> Result<()> {
    let (jtx, jrx) = channel::<Job>();
    let (rtx, rrx) = channel::<String>();
    let (mdir, gdir) = (PathBuf::from(&model), guesser.map(PathBuf::from));
    std::thread::spawn(move || {
        if let Err(e) = engine(mdir, gdir, jrx, rtx) {
            eprintln!("engine failed: {e:#}");
            std::process::exit(1);
        }
    });
    let id = rrx.recv().context("engine did not start")?;
    let jtx = Arc::new(Mutex::new(jtx));
    let server = tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| anyhow!("{e}"))?;
    eprintln!("ready: {id} on http://127.0.0.1:{port}/v1");
    for mut req in server.incoming_requests() {
        let path = req.url().split('?').next().unwrap_or("").trim_end_matches('/').to_string();
        match (req.method().as_str(), path.as_str()) {
            ("GET", "/v1/models") | ("GET", "/models") => {
                json_resp(req, json!({"object": "list", "data": [{"id": id, "object": "model", "owned_by": "local", "created": 0}]}), 200)
            }
            ("POST", p) if p.ends_with("/chat/completions") => {
                let mut body = String::new();
                let _ = req.as_reader().read_to_string(&mut body);
                let b: Value = serde_json::from_str(&body).unwrap_or(json!({}));
                let (otx, orx) = channel();
                let job = Job {
                    messages: b["messages"].as_array().cloned().unwrap_or_default(),
                    max: b["max_tokens"].as_u64().or(b["max_completion_tokens"].as_u64()).unwrap_or(1024) as usize,
                    out: otx,
                };
                let _ = jtx.lock().unwrap().send(job);
                let (id2, stream) = (id.clone(), b["stream"].as_bool().unwrap_or(false));
                std::thread::spawn(move || {
                    if stream {
                        let h = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap();
                        let sse = Sse { rx: orx, buf: Vec::new(), id: "chatcmpl-fast".into(), model: id2, done: false };
                        let _ = req.respond(tiny_http::Response::new(tiny_http::StatusCode(200), vec![h], sse, None, None));
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
