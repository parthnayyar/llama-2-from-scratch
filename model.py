from dataclasses import dataclass
from typing import Optional
import torch


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float=10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0, "head_dim must be even"
    # construct the thetas ("theta" parameters)
    # according to formula, theta_i = 10000^(-2i / head_dim) for i in [0, 1, ..., head_dim // 2 - 1]
    theta_numerator = torch.arange(0, head_dim, 2).float() # (head_dim // 2)
    theta **= -theta_numerator / head_dim # (head_dim // 2)
    theta = theta.to(device)
    # construct the positions ("m" parameters)
    m = torch.arange(seq_len, device=device) # (seq_len)
    # get m_i*theta_j for i in [0, 1, ..., seq_len - 1] and j in [0, 1, ..., head_dim // 2 - 1]
    freqs = torch.outer(m, theta).float() # (seq_len, head_dim // 2)
    # compute complex numbers from the frequencies with magnitude 1 and angle freqs
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # (seq_len, head_dim // 2)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    x_grouped = x.view(*x.shape[:-1], -1, 2) # (B, T, H, head_dim) -> (B, T, H, head_dim // 2, 2)
    x_grouped = x_grouped.float()
    x_complex = torch.view_as_complex(x_grouped) # (B, T, H, head_dim // 2, 2) -> (B, T, H, head_dim // 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) # (T, head_dim // 2) -> (1, T, 1, head_dim // 2, 2)
    x_rotated = x_complex * freqs_complex # (B, T, H, head_dim // 2) * (1, T, 1, head_dim // 2) -> (B, T, H, head_dim // 2)
    x_real = torch.view_as_real(x_rotated) # (B, T, H, head_dim // 2) -> (B, T, H, head_dim // 2, 2)
    # x_out = x_real.view(*x.shape[-1]) # (B, T, H, head_dim // 2, 2) -> (B, T, H, head_dim)
    x_out = x_real.view(*x.shape) # (B, T, H, head_dim // 2, 2) -> (B, T, H, head_dim)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, T, N_KV_H, H_D = x.shape # (B, T, n_kv_heads, head_dim)
    if n_rep == 1: return x
    x_expanded = x[:, :, :, None, :].expand(B, T, N_KV_H, n_rep, H_D) # (B, T, n_kv_heads, head_dim) -> (B, T, n_kv_heads, n_rep, head_dim)
    x_reshaped = x_expanded.view(B, T, N_KV_H * n_rep, H_D) # (B, T, n_kv_heads, n_rep, head_dim) -> (B, T, n_kv_heads * n_rep, head_dim)
    return x_reshaped


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # number of heads for queries
    n_kv_heads: Optional[int] = None # number of heads for K & V
    vocab_size: int = -1 # set when we load tokenizer
    multiple_of: int = 256 # make sure the model dimension is a multiple of this number
    ffn_dim_multiplier: Optional[int] = None # multiplier for the feedforward dimension
    norm_eps: float = 1e-5 # epsilon for rmsnorm normalization

    # needed for KV cache
    max_batch_size: int = 32 # max batch size for KV cache
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: torch.nn.Parameter = torch.nn.Parameter(torch.ones(dim)) # gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # implement RMSnorm (RMSnorm is similar to LayerNorm, but does no recentering, only scales downt the sd)
        x_float = x.float()
        norm = x_float / (torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True)) + self.eps)
        return self.weight * norm.type_as(x)
    

class SelfAttention(torch.nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_kv_heads: int = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads # number of heads for K & V
        self.n_heads_q: int = args.n_heads # number of heads for Q
        self.n_rep: int = self.n_heads_q // self.n_kv_heads # number of repetitions of the K & V to match the number of Q heads
        self.head_dim: int = args.dim // args.n_heads # dimension of each head

        self.wq: torch.nn.Linear = torch.nn.Linear(args.dim, args.dim, bias=False)
        self.wk: torch.nn.Linear = torch.nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv: torch.nn.Linear = torch.nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo: torch.nn.Linear = torch.nn.Linear(self.n_heads_q * self.head_dim, args.dim, bias=False)

        self.cache_k: torch.Tensor = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v: torch.Tensor = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert T == 1, "seq_len must be 1"

        xq = self.wq(x) # (B, T, D) -> (B, T, D)
        xk = self.wk(x) # (B, T, D) -> (B, T, n_kv_heads * head_dim)
        xv = self.wv(x) # (B, T, D) -> (B, T, n_kv_heads * head_dim)

        xq = xq.view(B, T, self.n_heads_q, self.head_dim) # (B, T, D) -> (B, T, n_heads_q, head_dim)
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim) # (B, T, n_kv_heads * head_dim) -> (B, T, n_kv_heads, head_dim)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim) # (B, T, n_kv_heads * head_dim) -> (B, T, n_kv_heads, head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device) # (B, T, n_heads_q, head_dim) -> (B, T, n_heads_q, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device) # (B, T, n_kv_heads, head_dim) -> (B, T, n_kv_heads, head_dim)

        # append to cache
        self.cache_k[:B, start_pos : start_pos+T] = xk
        self.cache_v[:B, start_pos : start_pos+T] = xv

        # retrieve from cache
        keys = self.cache_k[:B, :start_pos+T] # (B, start_pos, n_kv_heads, head_dim) 
        values = self.cache_v[:B, :start_pos+T] # (B, start_pos, n_kv_heads, head_dim)

        # repeat K and V to match the number of Q heads
        keys_repeated = repeat_kv(keys, self.n_rep) # (B, start_pos, n_heads_q, head_dim)
        values_repeated = repeat_kv(values, self.n_rep) # (B, start_pos, n_heads_q, head_dim)

        xq = xq.transpose(1, 2) # (B, T, n_heads_q, head_dim) -> (B, n_heads_q, T, head_dim)
        keys_repeated = keys_repeated.transpose(1, 2) # (B, start_pos, n_heads_q, head_dim) -> (B, n_heads_q, start_pos, head_dim)
        values_repeated = values_repeated.transpose(1, 2) # (B, start_pos, n_heads_q, head_dim) -> (B, n_heads_q, start_pos, head_dim)

        scores = torch.nn.functional.softmax(torch.matmul(xq, keys_repeated.transpose(-1, -2)) / self.head_dim**0.5, dim=-1).type_as(xq) # (B, n_heads_q, T, head_dim) @ (B, n_heads_q, head_dim, start_pos) -> (B, n_heads_q, T, start_pos)
        out = torch.matmul(scores, values_repeated) # (B, n_heads_q, T, start_pos) @ (B, n_heads_q, start_pos, head_dim) -> (B, n_heads_q, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, n_heads_q, T, head_dim) -> (B, T, D)
        out = self.wo(out) # (B, T, D) -> (B, T, D)
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_dim = int(8*args.dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of) # make sure the hidden dimension is a multiple of args.multiple_of

        self.w1 = torch.nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = torch.nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = torch.nn.functional.silu(self.w1(x)) # (B, T, D) -> (B, T, hidden_dim)
        x_V = self.w3(x) # (B, T, D) -> (B, T, hidden_dim)
        x = swish * x_V # (B, T, hidden_dim) * (B, T, hidden_dim) -> (B, T, hidden_dim)
        x = self.w2(x) # (B, T, hidden_dim) -> (B, T, D)
        return x
    
    
class EncoderBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads: int = args.n_heads
        self.dim: int = args.dim
        self.head_dim: int = args.dim // args.n_heads

        self.attention: SelfAttention = SelfAttention(args)
        self.feed_forward: FeedForward = FeedForward(args)

        self.attention_norm: RMSNorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm: RMSNorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex) # (B, T, D) + (B, T, D) -> (B, T, D)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(torch.nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size != -1, "vocab_size must be set"
        self.args: ModelArgs = args
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList([
            EncoderBlock(args) for _ in range(args.n_layers)
        ])

        self.norm: RMSNorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output: torch.nn.Linear = torch.nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex: torch.Tensor = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads, 
            self.args.max_seq_len * 2, 
            device=self.args.device
        )

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        B, T = tokens.shape
        assert T == 1, "tokens must be of shape (batch_size, 1)"

        # get token embeddings
        h = self.tok_embeddings(tokens) # (B, 1) -> (B, 1, D)

        # retrieve the pairs (m, theta) corresponding to the position
        freqs_complex = self.freqs_complex[start_pos : start_pos + 1] 

        # sequentially apply all layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
        