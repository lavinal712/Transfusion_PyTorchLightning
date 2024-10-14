from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float = 0.0):
        super().__init__()
        hidden_dim = multiple_of * ((int(2 * hidden_dim / 3) + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False, proj_drop: float = 0.0, attn_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        total_kv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=qkv_bias)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_drop = attn_drop
        self.dropout = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.num_kv_heads * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        values = values.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_drop if self.training else 0
        )            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.attention = Attention(dim, num_heads)
        self.feed_forward = FeedForward(dim, hidden_dim, 256)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.attention_norm = RMSNorm(dim, norm_eps)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, 
        input_pos: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, input_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(
        self, block_size: int, vocab_size: int, num_layers: int,
        dim: int, num_heads: int, mlp_ratio: float = 4.0, norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        hidden_dim = int(dim * mlp_ratio)
        self.layers = nn.ModuleList(TransformerBlock(dim, num_layers, hidden_dim) for _ in range(num_layers))
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(block_size, dim // num_heads)
    
    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, 
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.tok_embeddings(idx)
        if self.training:
            freqs_cis = self.freqs_cis[:x.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]

        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, input_pos, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits


if __name__ == "__main__":
    pass
