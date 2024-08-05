# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import DropPath, Mlp

from src.utils.pos_embed import get_sincos_pos_embed, get_2d_sincos_pos_embed


def expand_size(sz):
    if isinstance(sz, int):
        return [sz, sz]
    return sz


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding -- borrowed from https://pypi.org/project/timm/0.4.12/
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = expand_size(patch_size)

        # make it compatible with stem transformer
        if len(patch_size) > 2:
            patch_size = patch_size[-2:]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            x (torch.Tensor): input image/LMS, shape (*, in_chans, H, W)

        Returns:
            torch.Tensor: patch-embedded image, shape (*, H / patch_size, W / patch_size, embed_dim)
        """
        *batch_dims, in_chans, height, width = x.size()
        x = self.proj(x.view(-1, in_chans, height, width))
        if self.flatten:
            x = x.permute(0, 2, 3, 1)  # channels-last
        x = self.norm(x)
        return x.view(*batch_dims, *x.shape[-3:])


class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Implementation of the Multihead Attention compatible with Flash Attention and Nested Tensors.

        Args:
            x (torch.Tensor): input tensor, shape (batch_size, length, embed_dim).
                The length can be variable if a NestedTensor is passed.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # each one has shape batch_size, num_heads, seq_length, embed_dim

        x = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.attn_drop if self.training else 0.,
                                           scale=self.scale).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer encoder (M2D) implementation based on the MAE.
    """

    def __init__(self, img_size=224, patch_size=16,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size, self.patch_size = expand_size(img_size), expand_size(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)]

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.grid_size, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            FlashAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        embed_dim = self.pos_embed.size(-1)
        pos_embed = get_sincos_pos_embed(embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, return_layers: bool = False):
        r"""

        Args:
        """
        # embed patches
        batch_size, freq_patches, time_patches, embed_dim = x.size()

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, :, :time_patches, :]

        # flatten
        x = x.view(batch_size, -1, embed_dim)

        # apply Transformer blocks
        layers = []
        for blk in self.blocks:
            x = blk(x)
            if return_layers:
                layers.append(x)

        x = self.norm(x)
        if return_layers:
            layers.pop()  # replace the last feature with the normalized one.
            layers.append(x)

        if return_layers:
            return torch.stack(layers)

        return x
