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
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (seq_len, head_dim // 2, 2)
    return cache.to(dtype=dtype)


def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 4)[: (n_elem // 4)].float() / n_elem))
    t = torch.arange(grid_size ** 2, device=freqs.device)
    x = t % grid_size
    y = t // grid_size
    x_freqs = torch.outer(x, freqs) # (grid_size ** 2, head_dim // 4)
    y_freqs = torch.outer(y, freqs) # (grid_size ** 2, head_dim // 4)
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis_real = torch.cat([x_cis.real, y_cis.real], dim=-1) # (grid_size ** 2, head_dim // 2)
    freqs_cis_imag = torch.cat([x_cis.imag, y_cis.imag], dim=-1) # (grid_size ** 2, head_dim // 2)
    cache = torch.stack([freqs_cis_real, freqs_cis_imag], dim=-1) # (grid_size ** 2, head_dim // 2, 2)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
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
