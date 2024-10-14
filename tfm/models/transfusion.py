from typing import Optional, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from omegaconf import OmegaConf
import sys
sys.path.append("/home/v-yuqianhong/Transfusion")
from tfm.modules.patch_embed import LinearEmbed, UNetEmbed 
from tfm.llama import RMSNorm, FeedForward, Attention, precompute_freqs_cis
from tfm.util import instantiate_from_config
from tfm.models.autoencoder import DiagonalGaussianDistribution


class TransfusionBlock(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        num_kv_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        norm_eps: float = 1e-5,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        ffn_drop: float = 0.0,
        multiple_of: int = 256,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.attn = Attention(
            dim, num_heads=num_heads, num_kv_heads=num_kv_heads, 
            qkv_bias=qkv_bias, proj_drop=proj_drop, attn_drop=attn_drop,
        )
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden_dim, multiple_of=multiple_of, dropout=ffn_drop)
    
    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, 
        input_pos: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis, input_pos, mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Transfusion(nn.Module):
    def __init__(
        self,
        block_config,
        block_size: int = 2048,
        num_layers: int = 12,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_layers = num_layers
        self.dim = block_config["dim"]
        self.num_heads = block_config["num_heads"]
        self.norm_eps = block_config["norm_eps"]

        self.blocks = nn.ModuleList([
            TransfusionBlock(**block_config) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(self.dim, self.norm_eps)

        self.freqs_cis = precompute_freqs_cis(block_size, self.dim // self.num_heads)
        self.causal_mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
    
    def forward(
            self, h: torch.Tensor, input_pos: Optional[torch.Tensor] = None,
            mask: torch.Tensor = None
        ) -> torch.Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if mask is None:
            mask = self.causal_mask[None, None, input_pos].to(h.device)
        if self.training:
            freqs_cis = self.freqs_cis[:h.shape[1]].to(h.device)
        else:
            freqs_cis = self.freqs_cis[input_pos].to(h.device)
        for block in self.blocks:
            h = block(h, freqs_cis, input_pos, mask)
        h = self.norm(h)
        return h
    

class TransformerDiffusion(pl.LightningModule):
    def __init__(
        self,
        transformer_config,
        encoder_decoder_config,
        denoiser_config,
        first_stage_config,
        tokenizer_config,
    ):
        super().__init__()
        self.model = instantiate_from_config(transformer_config)
        self.first_stage_model = self.instantiate_first_stage(first_stage_config)
    
    def _compute_mask(self, ):
        pass

    def forward(self, inputs: List[torch.Tensor]):
        pass

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
    
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.no_grad()
    def get_input(
        self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
        cond_key=None, return_original_cond=False, bs=None, return_x=False
    ):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, "b h w c -> b c h w")
        x = x.to(memory_format=torch.contiguous_format).float()
        
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if cond_key is None:
            cond_key = self.cond_stage_key
        if cond_key != self.first_stage_key:
            xc = batch[cond_key]
        else:
            xc = x

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)


if __name__ == "__main__":
    config = OmegaConf.load("configs/transfudion.yaml")
    model = instantiate_from_config(config).cuda()
    x = torch.randn(3, 256, 768).cuda()
    input_pos = torch.randint(0, 2048, (3,))
    print(model(x, input_pos).shape)
