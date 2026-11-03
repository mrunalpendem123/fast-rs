//! Safetensors index: tensor name -> (file, byte offset, byte length, shape, dtype), read from the shard headers.
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub file: PathBuf,
    pub offset: u64,
    pub len: usize,
    pub shape: Vec<i32>,
    pub dtype: String,
}

pub fn index(dir: &Path) -> Result<HashMap<String, TensorInfo>> {
    let mut files: Vec<PathBuf> = Vec::new();
    let idx = dir.join("model.safetensors.index.json");
    if idx.exists() {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&idx)?)?;
        let mut set: Vec<String> = v["weight_map"].as_object().context("weight_map")?.values()
            .map(|x| x.as_str().unwrap().to_string()).collect();
        set.sort();
        set.dedup();
        files.extend(set.into_iter().map(|f| dir.join(f)));
    } else {
        files.push(dir.join("model.safetensors"));
    }
    let mut out = HashMap::new();
    for f in files {
        let mut fh = File::open(&f).with_context(|| format!("open {f:?}"))?;
        let mut n = [0u8; 8];
        fh.read_exact(&mut n)?;
        let n = u64::from_le_bytes(n);
        let mut h = vec![0u8; n as usize];
        fh.read_exact(&mut h)?;
        let v: serde_json::Value = serde_json::from_slice(&h)?;
        for (k, t) in v.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let o = t["data_offsets"].as_array().unwrap();
            let (a, b) = (o[0].as_u64().unwrap(), o[1].as_u64().unwrap());
            out.insert(k.clone(), TensorInfo {
                file: f.clone(),
                offset: 8 + n + a,
                len: (b - a) as usize,
                shape: t["shape"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect(),
                dtype: t["dtype"].as_str().unwrap().to_string(),
            });
        }
    }
    Ok(out)
}
