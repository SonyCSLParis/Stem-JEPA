# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import logging
import os
import sys
from typing import Sequence

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from numpy import load

import torch
import torch.nn as nn

from .vision_transformer import PatchEmbed, VisionTransformer


log = logging.getLogger(__name__)


class MixEncoder(VisionTransformer):
    """ Vision Transformer encoder (M2D) implementation based on the MAE.
    """
    def __init__(self, in_chans: int = 1, img_size=224, patch_size=16,
                 embed_dim=1024, depth=24, num_heads=16,
                 to_keep_ratio: float = 1.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super(MixEncoder, self).__init__(img_size=img_size,
                                   patch_size=patch_size,
                                   embed_dim=embed_dim,
                                   depth=depth,
                                   num_heads=num_heads,
                                   mlp_ratio=mlp_ratio,
                                   norm_layer=norm_layer)

        self.to_keep_ratio = to_keep_ratio
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.initialize_weights()
    
    def initialize_weights(self):
        super(MixEncoder, self).initialize_weights()

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor, return_layers: bool = False):
        r"""

        Args:
        """
        # embed patches
        x = self.patch_embed(x)
        batch_size, freq_patches, time_patches, embed_dim = x.size()

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, :, :time_patches, :]

        original_shape = x.size()

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
            layers.append(x.view(*original_shape))

        if return_layers:
            return torch.stack(layers)

        return x.view(*original_shape)


class StemPredictor(VisionTransformer):
    """ Masked Modeling Duo (M2D) implementation based on the MAE.
    """
    def __init__(self, num_sources: int = 4, img_size=224, patch_size=16, encoder_embed_dim=1024,
                 embed_dim=512, depth=8, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__(img_size=img_size,
                         patch_size=patch_size,
                         embed_dim=embed_dim,
                         depth=depth,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         norm_layer=norm_layer)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.predictor_embed = nn.Linear(encoder_embed_dim, embed_dim, bias=True)
        self.predictor_pred = nn.Linear(embed_dim, encoder_embed_dim, bias=True)  # predict target embeddings

        self.source_embed = nn.Embedding(num_sources, embed_dim)

        self.initialize_weights()

    def forward(self,
                x: torch.Tensor,
                target_indices: torch.LongTensor) -> torch.Tensor:
        original_shape = x.size()

        # embed tokens
        x = self.predictor_embed(x)

        # add pos embed and flatten for making it a sequence
        x = torch.flatten(x + self.pos_embed[..., :x.size(-2), :], 1, -2)

        # concatenate source conditioning
        source_token = self.source_embed(target_indices)
        x = torch.cat((source_token, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # discard source conditioning
        x = x[:, 1:, :]

        x = self.norm(x)

        # predictor projection
        x = self.predictor_pred(x)

        return x.view(*original_shape)


class MlpPredictor(nn.Module):
    def __init__(self,
                 num_sources: int = 4,
                 embed_dim: int = 768,
                 hidden_dims: Sequence[int] = (),
                 activation_layer = nn.ReLU,
                 norm_layer = nn.LayerNorm,
                 conditioning: str = "concat") -> None:
        super().__init__()
        self.embed_dim = embed_dim

        if conditioning.startswith("concat"):
            elems = conditioning.split('-')
            if len(elems) == 1:
                src_token_dim = embed_dim
            elif len(elems) == 2:
                conditioning = elems[0]
                src_token_dim = int(elems[1])
            else:
                raise ValueError(f"Got an invalid predictor conditioning: {conditioning}")

        self.conditioning = conditioning

        if self.conditioning == "none":
            self.source_embed = None
        else:
            self.source_embed = nn.Embedding(num_sources, src_token_dim)

        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        dims = [embed_dim] + list(hidden_dims) + [embed_dim]

        if self.conditioning == "concat":
            dims[0] = embed_dim + src_token_dim
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation_layer())
        
        layers.pop()

        self.blocks = nn.Sequential(*layers)
    
    def forward(self,
                x: torch.Tensor,
                target_indices: torch.LongTensor | None = None) -> torch.Tensor:
        
        if self.conditioning == "concat":
            source_embeddings = self.source_embed(target_indices)
            bs, *_, embed_dim = source_embeddings.size()
            source_embeddings = source_embeddings.view(bs, 1, 1, embed_dim).expand(*x.shape[:-1], embed_dim)
            x = torch.cat((x, source_embeddings), dim=-1)

        x = self.blocks(x)

        return self.norm(x)
    
    @property
    def num_sources(self) -> int:
        return self.source_embed.num_embeddings
