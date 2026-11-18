# fast-rs

**Exact speculative decoding for Apple Silicon.** Make the model faster, not different.

fast-rs is an LLM inference engine written in Rust with hand-written Metal kernels. It guesses several tokens ahead with a small drafter, checks the whole guess tree with the big model in one pass, and keeps only what the big model agrees with. Its output is **bit-identical to the same engine's plain greedy decoding**, checked on every answer.

Full [Spec-Bench](https://github.com/hemingkx/Spec-Bench) (all 480 questions, both MT-Bench turns), greedy, up to 1,024 new tokens per turn, on a 13-inch MacBook Pro with the plain M2 chip (10-core GPU, 8 GB):

| Model (4-bit) | Drafter | Plain tok/s | fast-rs tok/s | Speedup | Identical |
|---|---|---|---|---|---|
| LFM2.5-2.6B | LFM2.5 DSpark | 34.4 | 86.0 | **2.50×** | 480/480 |
| Qwen3.5-4B | z-lab DFlash | 21.5 | 49.5 | **2.30×** | 480/480 |
| Qwen3-4B | z-lab DFlash | 21.3 | 48.4 | **2.27×** | 480/480 |
| Gemma 4 E2B | Gemma assistant | 28.3 | 55.8 | **1.98×** | 480/480 |

Speed is new tokens divided by wall time, **including prompt reading**. Speedup is fast-rs over the same engine with speculation off. The raw per-question results are in [`results/`](results/).

## Why exactness is the hard part

On 4-bit models, speculative decoding usually changes the output. Checking 8 guessed tokens runs the matrix maths with 8 rows instead of 1, and the kernels add the numbers in a different order. Floating-point addition isn't associative, so on a near-tie the last bit can flip the top token, and the text drifts from there. This is reported in [llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618), [#27407](https://github.com/ggml-org/llama.cpp/issues/27407), [#31472](https://github.com/ggml-org/llama.cpp/issues/31472) and the [vLLM batch-invariance RFC #54506](https://github.com/vllm-project/vllm/issues/54506).

fast-rs uses **batch-invariant 4-bit kernels**: every row is reduced in the same order whatever the number of rows, so checking a tree of guesses computes exactly what one-token-at-a-time decoding would. `specbench` runs both paths on every question and records whether the tokens match.

## How it relates to recent results on Macs

- [Yemets et al., *Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon*](https://arxiv.org/abs/2610.31479) (NeurIPS 2026 On-Device Intelligence workshop) found every speculative method they tested slowed down Qwen3-4B 4-bit in MLX. On an M2 Pro chip, end-to-end: DFlash −32.5%, DDTree −28.0%. Their workload is different (100 chat prompts averaging 1,776 tokens with ~60-token answers), so this is not a matched comparison.
- [Chordiya, *Lossless but Not Free*](https://arxiv.org/abs/2607.17283) reported a best case of 1.61× on an Apple-silicon laptop, with 3 of 5 configurations slower, partly because a quantized Metal backend ran "parallel" verification serially.
- [Li, *Reference-Preserving Speculative Decoding on Consumer Systems*](https://arxiv.org/abs/2610.03228) preserved the greedy reference on a tested panel, but on an M4 the target alone was 1.35× faster than speculation.

## Limits

- **Plain decoding is slower than MLX on this Mac** (~26 vs ~34 tok/s on Qwen3-4B 4-bit, measured differently: llama-bench-style tg128 vs mlx-lm decode). The gain comes only from speculation. With speculation on, fast-rs measured ~2× MLX's speed in a 4-prompt test on Qwen3.5-4B (62.5 vs 32.0 tok/s).
- **The speedup depends on the task:** maths up to 3.57×, while RAG and summarization gain ~1.7× because ~30% of their time is spent reading the prompt.
- **One machine so far.** Every number above comes from one M2 MacBook Pro. Results from other Macs are welcome.
- Exactness means identical to **fast-rs's own** plain greedy decoding. Different engines (MLX, llama.cpp) can differ from each other at near-ties because their kernels round differently.
- Batch size 1 (one user, one conversation). macOS on Apple Silicon only.

## Quick start

Requirements: an Apple Silicon Mac, Rust (stable 1.93 or newer), and `pip install -U huggingface_hub` for the model download script.

```sh
cargo build --release
scripts/get-model.sh qwen3-4b            # 4-bit target + drafter, pinned revisions, into models/

# chat in the browser at http://localhost:8800 (OpenAI-compatible: POST /v1/chat/completions)
./target/release/fast-rs serve models/qwen3-4b-4bit models/qwen3-4b-dflash 8800
```

Models available through `scripts/get-model.sh`: `qwen3-4b`, `qwen3.5-4b`, `lfm2.5-2.6b`, `gemma-4-e2b`, `gemma-4-e4b`. Each model keeps its own licence; see its Hugging Face page.

## Reproduce the benchmark

```sh
# all 480 Spec-Bench questions: plain run, fast run, exactness check per answer
./target/release/fast-rs specbench models/qwen3-4b-4bit models/qwen3-4b-dflash \
    data/specbench/question.jsonl my-qwen3-4b.jsonl 1024 8

python3 scripts/summary.py my-qwen3-4b.jsonl   # per-category table
```

Arguments: `specbench MODEL DRAFTER QUESTIONS OUT [MAX_NEW=1024] [TREE_BUDGET=8]`. Each output line has the question, both speeds, the mean number of accepted tokens per check, and `identical` for every turn. A full run takes 3–6 hours on an 8 GB M2. Keep the Mac plugged in, keep the lid open, and close other heavy apps, because swapping distorts the timings.

**Sharing results from your Mac:** open a pull request adding your output file under `results/<chip>-<memory>/`, with your chip, GPU core count, memory and macOS version in the description.

## Commands

| Command | What it does |
|---|---|
| `serve MODEL [DRAFTER] [PORT]` | Chat UI and OpenAI-compatible API on `localhost` |
| `specbench MODEL DRAFTER QUESTIONS OUT [MAX_NEW] [BUDGET]` | Spec-Bench protocol, plain vs fast, with an exactness check |
| `bench MODEL DRAFTER\|- PROMPTS OUT [RUNS] [BUDGET] [MAX_NEW]` | llama-bench-style pp512 / tg128, then repeated timed runs on a tokenized prompt set, with an exactness check on every run |
| `prefixtest MODEL DRAFTER QUESTIONS [N] [MAX_NEW]` | Second turns that reuse the first turn's memory, checked against plain decoding from scratch |

On first run, fast-rs tunes its kernels for your GPU and caches the choice in `~/.cache/fast-rs/`.

## Layout

```
src/kernels.rs   Metal kernels (4-bit matmul, attention, linear-attention/conv layers, drafter heads)
src/main.rs      model loading, tree speculative decoding, benchmarks
src/gemma.rs     Gemma 4 specifics
src/server.rs    HTTP server, chat templates, stop tokens
src/chat.html    browser chat UI
data/specbench/  Spec-Bench questions (Apache-2.0, see NOTICE)
results/         raw per-question results behind the table above
```

## Credits

Spec-Bench questions by Xia et al. ([hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench), Apache-2.0). Drafters: [z-lab DFlash](https://huggingface.co/z-lab), [LiquidAI LFM2.5 DSpark](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark), [Google Gemma 4 assistants](https://huggingface.co/google). 4-bit weights from [mlx-community](https://huggingface.co/mlx-community). Bundled fonts and pdf.js are listed in [NOTICE](NOTICE).

## License

Apache-2.0. See [LICENSE](LICENSE).
