from __future__ import annotations

from sentencepiece import SentencePieceProcessor
from typing import Optional
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

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded model in {time.time() - prev_time:.2f}s")

        return LLaMA(model, tokenizer, model_args)
    

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
    print("all OK")