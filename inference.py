from __future__ import annotations

from sentencepiece import SentencePieceProcessor
from contextlib import redirect_stdout
from traceback import print_exc
from typing import Literal
from pathlib import Path
from tqdm import tqdm
import torch
import time
import json
import sys

from model import ModelArgs, Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args

    @staticmethod
    def build_model(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str) -> LLaMA:
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"No checkpoint files found in {checkpoints_dir}"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint from {chk_path}")
            checkpoint = torch.load(chk_path, map_location=device)
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.load(f)

        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda": torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else: torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded model in {time.time() - prev_time:.2f}s")

        return LLaMA(model, tokenizer, model_args)

    def generate(self, prompts: list[str], temperature: float = 1.0, max_gen_len = None, strategy: Literal["greedy", "beam", "random", "top_p", "top_k"] = "top_p", **kwargs) -> tuple[list[list[int]], list[str]]:
        assert strategy in ["greedy", "beam", "random", "top_p", "top_k"], f"Invalid strategy: {strategy}. Must be one of ['greedy', 'beam', 'random', 'top_p', 'top_k']"
        if strategy == "beam": default_kwargs = {"k": 2}
        elif strategy == "top_p": default_kwargs = {"p": 0.5}
        elif strategy == "top_k": default_kwargs = {"k": 50}
        else: default_kwargs = {}
        kwargs = default_kwargs | kwargs
        if strategy == "beam": assert kwargs["k"] > 1, "Beam search requires k > 1"
        elif strategy == "top_p": assert 0 < kwargs["p"] < 1, "Top-p sampling requires 0 < p < 1"
        elif strategy == "top_k": assert kwargs["k"] > 1, "Top-k sampling requires k > 1"

        assert temperature >= 0, f"Temperature must be >= 0, got {temperature}"
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        prompt_lengths = [len(prompt) for prompt in prompt_tokens]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        if strategy == "beam": assert batch_size*kwargs["k"] <= self.model_args.max_batch_size, f"batch size must be less than or equal to {self.model_args.max_batch_size}//{kwargs['k']} for beam search with k={kwargs['k']}"
        else: assert batch_size <= self.model_args.max_batch_size, f"batch size must be less than or equal to {self.model_args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.model_args.max_seq_len, f"prompt length must be less than or equal to {self.model_args.max_seq_len}"
        total_len = min(self.model_args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.model_args.device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.model_args.device)
        
        eos_reached = torch.tensor([False] * batch_size, device=self.model_args.device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise

        self.model.empty_kv_cache()

        if strategy == "beam":
            return self._beam(tokens, prompt_tokens_mask, prompt_lengths, max_gen_len, temperature, kwargs["k"])

        cur_iterator = tqdm(range(1, total_len), desc=f"Generating tokens using {strategy} with {temperature=}" + (f", p={kwargs["p"]}" if strategy == "top_p" else f", k={kwargs["k"]}" if strategy == "top_k" else ""))
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model(tokens[:, cur_pos-1:cur_pos], cur_pos)
                logits = logits[:, -1]
            if temperature > 0:
                logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            if temperature == 0 or strategy == "greedy":
                next_token = torch.argmax(probs, dim=-1)
            elif strategy == "top_p":
                next_token = self._sample_top_p(probs, kwargs["p"])
            elif strategy == "random":
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = self._sample_top_k(probs, kwargs["k"])
            
            next_token = next_token.view(-1)
            # Only replace token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for i, current_prompt_tokens in enumerate(tokens.tolist()):
            current_prompt_tokens = current_prompt_tokens[:prompt_lengths[i] + max_gen_len]
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text

    def _beam(self, tokens: torch.Tensor, prompt_tokens_mask: torch.Tensor, prompt_lengths: list[int], max_gen_len: int, temperature: float, k: int) -> tuple[list[list[int]], list[str]]:
        assert k > 1, "Beam size must be greater than 1 for beam search."
        batch_size, total_len = tokens.shape

        # Expand the tokens and masks to k beams per batch element
        expanded_tokens = tokens.repeat_interleave(k, dim=0)  # (batch_size * k, total_len)
        expanded_prompt_mask = prompt_tokens_mask.repeat_interleave(k, dim=0)  # (batch_size * k, total_len)

        # Initialize beam scores: (batch_size, k)
        beam_scores = torch.zeros((batch_size, k), device=self.model_args.device)
        beam_scores[:, 1:] = -float("inf")  # Only the first beam is active initially

        # Track whether each beam has reached EOS
        eos_reached = torch.zeros((batch_size, k), dtype=torch.bool, device=self.model_args.device)

        for cur_pos in tqdm(range(1, total_len), desc=f"Generating tokens using beam search with {temperature=}, {k=} beams"):
            # Get the logits for the next token
            with torch.no_grad():
                # Input is the tokens up to cur_pos-1
                logits = self.model(expanded_tokens[:, cur_pos-1:cur_pos], cur_pos)
                logits = logits[:, -1]

            # Apply temperature to logits
            if temperature > 0:
                logits /= temperature
            # use log probs instead of probs for numerical stability and to avoid underflow
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size * k, vocab_size)

            # For each beam, select top k tokens and their log probs
            topk_log_probs, topk_indices = log_probs.topk(k, dim=-1) # (batch_size * k, k)

            # Reshape to (batch_size, k, k)
            topk_log_probs = topk_log_probs.view(batch_size, k, k)

            # Compute new scores: beam_scores (batch, k, 1) + topk_log_probs (batch, k, k) => (batch, k, k)
            new_scores = beam_scores.unsqueeze(-1) + topk_log_probs

            # Flatten to (batch_size, k * k)
            new_scores_flat = new_scores.view(batch_size, -1)

            # Select top k candidates for each batch element
            topk_new_scores, topk_indices_flat = new_scores_flat.topk(k, dim=-1)  # (batch_size, k)

            # Determine parent beams and token ranks
            parent_beams = topk_indices_flat // k  # (batch_size, k)
            token_ranks = topk_indices_flat % k    # (batch_size, k)

            # Update beam scores
            beam_scores = topk_new_scores

            # Gather the token indices for the selected candidates
            beam_indices = (torch.arange(batch_size, device=self.model_args.device).unsqueeze(1) * k + parent_beams).view(-1)
            token_ranks = token_ranks.view(-1)

            # Gather the token IDs from topk_indices
            gathered_token_ids = topk_indices[beam_indices, token_ranks]

            # Apply prompt mask: if current position is part of the prompt, use the existing token
            current_prompt_mask = expanded_prompt_mask[:, cur_pos]
            existing_tokens = expanded_tokens[:, cur_pos]
            gathered_token_ids = torch.where(current_prompt_mask, existing_tokens, gathered_token_ids)

            # Update the expanded_tokens by copying the parent sequences and adding the new token
            expanded_tokens[:, :cur_pos] = expanded_tokens[beam_indices, :cur_pos]
            expanded_tokens[:, cur_pos] = gathered_token_ids

            # Check for EOS tokens in non-prompt positions
            eos_reached_new = (~current_prompt_mask) & (gathered_token_ids == self.tokenizer.eos_id)
            eos_reached_new = eos_reached_new.view(batch_size, k)

            # Update eos_reached and mask scores for completed beams
            eos_reached |= eos_reached_new
            beam_scores[eos_reached] = -float("inf")

            # Early stopping if all beams reached EOS
            if eos_reached.all():
                break

        # Select the best beam for each batch element
        best_beam_indices = beam_scores.argmax(dim=-1)  # (batch_size)
        best_beam_indices_expanded = (torch.arange(batch_size, device=self.model_args.device) * k) + best_beam_indices
        best_tokens = expanded_tokens[best_beam_indices_expanded]

        # Process the final tokens to remove padding and cut at EOS
        out_tokens = []
        out_text = []
        for i in range(batch_size):
            tokens_list = best_tokens[i].tolist()
            tokens_list = tokens_list[:prompt_lengths[i] + max_gen_len]
            if self.tokenizer.eos_id in tokens_list:
                eos_idx = tokens_list.index(self.tokenizer.eos_id)
                tokens_list = tokens_list[:eos_idx]
            out_tokens.append(tokens_list)
            out_text.append(self.tokenizer.decode(tokens_list))

        return out_tokens, out_text
        
    def _sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p 
        probs_sort[mask] = 0.0 
        probs_sort /= probs_sort.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token

    def _sample_top_k(self, probs: torch.Tensor, k: int) -> torch.Tensor:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sort[:, k:] = 0.0
        probs_sort /= probs_sort.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


if __name__ == "__main__":

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [
        "Roses are red, violets are",
        "Once upon a time in a land far far away",
        "Hi, I'm a large language model and I"
    ]
    
    model = LLaMA.build_model(
        checkpoints_dir="Llama-2-7b/",
        tokenizer_path="Llama-2-7b/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=6, # beam search requires batch size to be at least len(prompts)*beam_k
        device=device
    )

    max_new_tokens = 32
    temperature = 1.1

    with open("output.txt", "w", encoding="utf-8", errors="replace") as f:
        with redirect_stdout(f):
            try:
                print("Beam sampling:")
                out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=temperature, strategy="beam", k=2)
                assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
                print(f"\n{'-'*50}\n".join(out_texts))
                print("="*100)
            except Exception as e:
                print(f"Error during beam sampling: {e}", file=sys.stderr)
                print_exc()

            try:
                print("Greedy sampling:")
                out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=temperature, strategy="top_p", p=0.9)
                assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
                print(f"\n{'-'*50}\n".join(out_texts))
                print("="*100)
            except Exception as e:
                print(f"Error during greedy sampling: {e}", file=sys.stderr)
                print_exc()

            try:
                print("Top-p sampling:")
                out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=temperature, strategy="top_p", p=0.9)
                assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
                print(f"\n{'-'*50}\n".join(out_texts))
                print("="*100)
            except Exception as e:
                print(f"Error during top-p sampling: {e}", file=sys.stderr)
                print_exc()

            try:
                print("Top-k sampling:")
                out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=temperature, strategy="top_k", k=50)
                assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
                print(f"\n{'-'*50}\n".join(out_texts))
                print("="*100)
            except Exception as e:
                print(f"Error during top-k sampling: {e}", file=sys.stderr)
                print_exc()

            try:
                print("Random sampling:")
                out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=temperature, strategy="random")
                assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
                print(f"\n{'-'*50}\n".join(out_texts))
                print("="*100)
            except Exception as e:
                print(f"Error during random sampling: {e}", file=sys.stderr)
                print_exc()
