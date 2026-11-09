# Qwen3.5-4B 4-bit reference: greedy 128 words per test prompt with mlx-lm (our plain decoding should match closely).
import json, mlx.core as mx
from mlx_lm import load
import mlx_lm.models.lfm2 as L2
_san = L2.Model.sanitize
L2.Model.sanitize = lambda self, w: _san(self, {k.replace('language_model.', '', 1): v for k, v in w.items()})
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
P = "/Users/mrunalpendem/models/lfm2.5-2.6b-4bit"
model, tok = load(P)
old = json.load(open("/Users/mrunalpendem/Desktop/local-ai-research/fast-rs/prompts.json"))
T = json.load(open("/Users/mrunalpendem/models/qwen3-4b-4bit/tokenizer.json"))
out = {"eos": [124900, 124895], "prompts": []}
from tokenizers import Tokenizer
t4 = Tokenizer.from_file("/Users/mrunalpendem/models/qwen3-4b-4bit/tokenizer.json")
for p in old["prompts"]:
    text = t4.decode(p["ids"], skip_special_tokens=False)
    user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False)
    got = []
    for (t, _), _ in zip(generate_step(mx.array(ids), model, sampler=make_sampler(0.0)), range(128)):
        got.append(int(t))
        if t in out["eos"]: break
    out["prompts"].append({"name": p["name"], "ids": ids, "python_plain": got})
    print(p["name"], len(ids), repr(tok.decode(got[:40])), flush=True)
json.dump(out, open("/Users/mrunalpendem/Desktop/local-ai-research/fast-rs/promptsl.json", "w"))
