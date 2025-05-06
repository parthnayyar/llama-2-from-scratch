from __future__ import annotations

from sentencepiece import SentencePieceProcessor
from traceback import print_exc
from typing import Literal
from pathlib import Path
from tqdm import tqdm
import torch
import time
import json

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
        if strategy == "beam": default_kwargs = {"k": 3}
        elif strategy == "top_p": default_kwargs = {"p": 0.5}
        elif strategy == "top_k": default_kwargs = {"k": 50}
        else: default_kwargs = {}
        kwargs = default_kwargs | kwargs
        if strategy == "beam": assert kwargs["k"] > 1, "Beam search requires k > 1"
        elif strategy == "top_p": assert 0 < kwargs["p"] < 1, "Top-p sampling requires 0 < p < 1"
        elif strategy == "top_k": assert kwargs["k"] > 1, "Top-k sampling requires k > 1"


        assert temperature > 0, f"Temperature must be greater than 0, got {temperature}"
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.model_args.max_batch_size, f"batch size must be less than or equal to {self.model_args.max_batch_size}"
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

        if strategy == "beam":
            return self._beam(tokens, prompt_tokens_mask, temperature, kwargs["k"])

        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model(tokens[:, cur_pos-1:cur_pos], cur_pos)
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            if strategy == "top_p":
                next_token = self._sample_top_p(probs, kwargs["p"])
            elif strategy == "greedy":
                next_token = torch.argmax(probs, dim=-1)
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
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text
    
    def _beam(self, tokens: torch.Tensor, prompt_tokens_mask: torch.Tensor, temperature: float, k: int) -> tuple[list[list[int]], list[str]]: 
        # TODO
        ...
        

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

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LLaMA.build_model(
        checkpoints_dir="Llama-2-7b/",
        tokenizer_path="Llama-2-7b/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=4,
        device=device
    )

    prompts = [
        "Roses are red, violets are",
        "Once upon a time, there was a brave knight who",
        "7 + 5 =",
        """Complete the blank:

Q: The capital of France is ____
A: Paris

Q: The capital of Germany is ____
A: Berlin

Q: The capital of Italy is ____
A: Rome

Q: The capital of Spain is ____
A:"""
    ]

    prompts = [
        "Roses are red, violets are",
        "7 + 5 =",
        """Complete the blank:

Q: The capital of France is ____
A: Paris

Q: The capital of Germany is ____
A: Berlin

Q: The capital of Italy is ____
A: Rome

Q: The capital of Spain is ____
A:"""
    ]

    max_new_tokens = 4

    try:
        print("Greedy sampling")
        out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=0.9, strategy="top_p", p=0.9)
        assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
        print(f"\n{'-'*50}\n".join(out_texts))
        print("="*100)
    except Exception as e:
        print(f"Error during greedy sampling: {e}")
        print_exc()

    try:
        print("Random sampling")
        out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=0.9, strategy="random")
        assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
        print(f"\n{'-'*50}\n".join(out_texts))
        print("="*100)
    except Exception as e:
        print(f"Error during random sampling: {e}")
        print_exc()

    try:
        print("Top-p sampling")
        out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=0.9, strategy="top_p", p=0.9)
        assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
        print(f"\n{'-'*50}\n".join(out_texts))
        print("="*100)
    except Exception as e:
        print(f"Error during top-p sampling: {e}")
        print_exc()

    try:
        print("Top-k sampling")
        out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=0.9, strategy="top_k", k=3)
        assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
        print(f"\n{'-'*50}\n".join(out_texts))
        print("="*100)
    except Exception as e:
        print(f"Error during top-k sampling: {e}")
        print_exc()

    try:
        print("Beam sampling")
        out_tokens, out_texts = model.generate(prompts, max_gen_len=max_new_tokens, temperature=0.9, strategy="beam", k=2)
        assert len(out_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(out_tokens)}"
        print(f"\n{'-'*50}\n".join(out_texts))
        print("="*100)
    except Exception as e:
        print(f"Error during beam sampling: {e}")
        print_exc()

    