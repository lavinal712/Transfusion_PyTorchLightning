import math

import torch
import torch.nn as nn

from tfm.modules.diffusionmodules.openaimodel import (
    zero_module, TimestepEmbedSequential, Upsample, Downsample, 
    ResBlock, AttentionBlock, 
)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LinearEmbed(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        grid_size = image_size // patch_size
        self.num_patches = grid_size ** 2
        
        self.encoder = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.decoder = nn.Linear(embed_dim, out_channels * patch_size ** 2)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )
        self.norm_layer = norm_layer
    
    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def encode(self, x, emb):
        if x.dim() == 3:
            x = x[None]
        assert x.shape[0] == emb.shape[0]
        x = self.encoder(x) # (batch_size, embed_dim, grid_size, grid_size)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        x = self.norm_layer(x) * (1 + scale) + shift
        x = x.flatten(2).transpose(1, 2) # (batch_size, num_patches, embed_dim)
        return x
    
    def decode(self, x):
        x = self.decoder(x) # (batch_size, num_patches, out_channels * patch_size ** 2)
        x = self.unpatchify(x)
        return x
    
    def forward(self, x, emb):
        x = self.encode(x, emb)
        x = self.decode(x)
        return x


class UNetEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        time_embed_dim,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1)
            )
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2
        
        self.input_block_final = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, time_embed_dim, 3, padding=1)),
        )
        self.output_block_start = nn.Sequential(
            nn.GroupNorm(32, time_embed_dim),
            nn.SiLU(),
            zero_module(nn.Conv2d(time_embed_dim, ch, 3, padding=1)),
        )
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    def encode(self, x, emb, y=None):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.input_block_final(h).flatten(2).transpose(1, 2)
        return h, hs
    
    def decode(self, x, emb, hs):
        bz = x.shape[0]
        h_ = w_ = int(x.shape[1] ** 0.5)
        assert h_ * w_ == x.shape[1]
        
        h = x.transpose(1, 2).reshape(bz, -1, h_, w_)
        h = self.output_block_start(h)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
    
    def forward(self, x, emb):
        h, hs = self.encode(x, emb)
        h = self.decode(h, emb, hs)
        return h
