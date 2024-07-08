import torch
import torch.nn as nn
import re
import math
from einops import rearrange, repeat

class PerceiverBlock_modified(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, mult = 4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        ff_dim = dim * mult
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, ff_dim, bias=False),
                nn.GELU(),
                nn.Linear(ff_dim, dim, bias=False),
            ]
        )

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        split_sizes = [x.shape[1], latents.shape[1]]
        residual_x = x
        x = self.norm_media(x)
        residual_latents = latents
        latents = self.norm_latents(latents)
        residual = torch.cat((residual_x, residual_latents), dim=-2)

        h = self.heads
        qkv_input = torch.cat((x, latents), dim=-2)
        q = self.to_q(qkv_input)
        k, v = self.to_kv(qkv_input).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out) + residual
        residual_out = out
        for layer in self.feed_forward:
            out = layer(out)
        return torch.split(out + residual_out, split_sizes, dim=1)  

class PerceiverResampler_modified(nn.Module):
    def __init__(
        self,
        dim,
        hidden_size,
        patch_num,
        depth = 6,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        ff_mult = 4,
        mlp_depth = 2,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.pos_emb = nn.Parameter(torch.randn(patch_num + num_latents, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(PerceiverBlock_modified(dim=dim, dim_head=dim_head, heads=heads, mult=ff_mult))

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latent = nn.LayerNorm(dim)
        
        modules = [nn.Linear(dim, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.projector = nn.Sequential(*modules)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, v, D)
        Returns:
            shape (b, n, D) where n is self.num_latents
        """
        b, v = x.shape[:2]
        N = self.pos_emb.shape[0] - self.latents.shape[0]

        x = x + self.pos_emb[:N]
        latents = self.latents + self.pos_emb[N:]

        # blocks
        latents = repeat(latents, "n d -> b n d", b=b)
        for block in self.layers:
            x, latents = block(x, latents)

        return self.projector(torch.cat((self.norm_media(x), self.norm_latent(latents)), dim=-2))

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'ours':
        return PerceiverResampler_modified(config.mm_hidden_size, config.hidden_size, config.mm_patch_num, num_latents = 128)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
