# LLaMA-2-from-scratch

Reproduced LLaMA-2 7B model from scratch for inference in PyTorch with RoPE embeddings, KV cache, grouped multi-query attention, and flexible decoding strategies including beam search, top-k, top-p, and more.

## Introduction

This repository showcases a ground-up implementation of Meta’s LLaMA-2 7B transformer architecture, optimized for inference. Inspired by [Umar Jamail’s walkthrough on YouTube](https://www.youtube.com/watch?v=oM4VmoabDAI), I extended the design to support:

- **All major decoding strategies**: greedy, top-p, top-k, random sampling, and beam search.
- **Grouped multi-query attention** with separate query and KV heads.
- **RoPE embeddings** and positionally-aware KV caching.
- **Efficient beam search** from scratch for improved generation quality.

All components, from rotary embeddings to the decoding logic, are implemented cleanly with an emphasis on readability, correctness, and extensibility.

## Features

### Architecture Features

- **Grouped Multi-Query Attention (GQA):** Scales attention with fewer KV heads, allowing LLaMA-2's efficient inference behavior.
- **RoPE (Rotary Positional Embeddings):** Captures relative positional information without explicit positional vectors.
- **KV Cache with Replay:** Enables auto-regressive decoding by storing keys and values across steps.
- **Modular Layer Design:** Implements each transformer block with a clear separation of attention and feedforward sublayers.
- **RMSNorm:** Used throughout, consistent with LLaMA-2’s normalization approach.

### Decoding Strategies

- **Greedy Sampling:** Picks the token with the highest probability at each step.
- **Top-p Sampling:** Samples from the smallest set of tokens whose cumulative probability ≥ *p* using `torch.multinomial`.
- **Top-k Sampling:** Samples from the top *k* tokens by probability using `torch.multinomial`.
- **Random Sampling:** Samples from the full distribution using `torch.multinomial`.
- **Beam Search:** Implements batched beam search. This decoding strategy keeps track of the most probable *k* options for each prompt at each decoded token.

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA or CPU/BFloat16 support
- `sentencepiece`, `tqdm`, `json`, `torch`, `typing`, etc.

Install via:
```bash
pip install -r requirements.txt
```

## How to Run Inference

Clone the repo and prepare the checkpoint and tokenizer:

```bash
git clone https://github.com/yourusername/llama-2-from-scratch.git
cd llama-2-from-scratch
```

Download weights and tokenizer from [official LLaMA website](https://www.llama.com/llama-downloads/) into the directory `Llama-2-7b/` with:
- `params.json`
- `*.pth` model weights
- `tokenizer.model`

Then run:
```bash
python inference.py
```

Example prompts and decoding strategies are hardcoded in the `__main__` block. You can modify them directly or use the `LLaMA.generate()` interface.

## Sample Generations

For sample outputs, please have a look at `output.txt`.

## What I Learned

- **RoPE Embeddings:** Learned how rotary position embeddings work by converting real-valued vectors into complex space, enabling relative position encoding without explicit positional embeddings. Understood how to precompute and apply RoPE efficiently across transformer layers.
- **Grouped Multi-Query Attention:** Implemented and validated the behavior of using fewer KV heads than query heads, reducing memory and improving inference speed while preserving attention performance.
- **KV-Cache:** Built a key-value cache system to persist attention memory across time steps. Understood how to index into the cache using current position and batch, and how to manage cache for batched inference and beam search.
- **Decoding Strategies:** Gained hands-on understanding of various decoding strategies (greedy, top-k, top-p, random) and their impact on output diversity and quality. Implemented temperature scaling and conditional token replacement based on prompt masking.
- **Specifically Beam Search:** Learned how to implement batched beam search from scratch, including:
  - Tracking parent beams and token ranks
  - Efficient beam score updates using log-prob addition for numerical stability
  - EOS masking and beam pruning
  - Selecting best beams post-generation without leaking prompt content

## Potential Future Improvements

- Add streaming/interactive generation loop for long-form tasks.
- Optimize with `torch.compile()` for even faster inference.
- Add quantization support (e.g. int8) for CPU/GPU speedups.
- Add attention masking to support batched prompts with variable lengths.
- Initiating KV-cache in one go instead of iterative initialization.

## References

- [Meta's LLaMA Paper](https://arxiv.org/abs/2307.09288)
- [Umar Jamail’s LLaMA-2 YouTube Tutorial](https://www.youtube.com/watch?v=oM4VmoabDAI)
- [Rotary Positional Embeddings](https://arxiv.org/abs/2104.09864)
- [GPT-style Beam Search](https://huggingface.co/blog/how-to-generate)
- [LLaMA Model Card](https://huggingface.co/meta-llama/Llama-2-7b)
