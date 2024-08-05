"""Masked Modeling Duo (M2D) Wrapper for EVAR.

Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input
https://ieeexplore.ieee.org/document/10097236/

Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation
https://arxiv.org/abs/2305.14079
"""

from math import prod

import torch

from .ar_base import calculate_norm_stats, normalize_spectrogram
from .ar_lightning import ARLightning


class ARStems(ARLightning):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.mix_tracks = "Stem" not in self.backbone.__class__.__name__
        self.flattened = "MixEncoder" not in self.backbone.__class__.__name__

    def to_feature(self, batch_audio):
        r"""Converts audio batch into mel-spectrograms

        Args:
            batch_audio (torch.tensor): audio batch, shape (*, num_channels)
        """
        *batch_dims, num_channels = batch_audio.size()

        # raw -> spectrogram, and normalize
        x = self.to_spec(batch_audio.view(prod(batch_dims), num_channels))
        x = (x + torch.finfo().eps).log()

        _, *spec_dims = x.size()

        return x.view(*batch_dims, 1, *spec_dims)

    def encode_frames(self, batch_audio):
        if batch_audio.ndim > 2 and self.mix_tracks:
            batch_audio = batch_audio.mean(dim=1)

        x = self.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x)
        x = self.augment_if_training(x)

        # if we anyway plan to use only the last layer,
        # we do not save the intermediate layers for saving GPU memory
        if self.cfg.output_layers == [-1]:
            features = self.encode_lms(x, return_layers=False)
        
        else:
            hidden_states = self.encode_lms(x, return_layers=True)  # shape [L, B, T, D]
            
            # stack layer outputs
            states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in hidden_states]
            features = torch.cat(states_to_stack, axis=-1)

        return features.transpose(1, 2)  # [B, T, D] -> [B, D, T]

    def precompute_lms(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, lambda x: x, norm_stats=self.cfg.norm_stats)
        self.lms_mode = True

    def encode_frames_lms(self, batch_lms):
        x = normalize_spectrogram(self.norm_stats, batch_lms)
        x = self.augment_if_training(x)
        hidden_states = self.runtime.encode_lms(x, return_layers=True)
        # stack layer outputs
        states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in hidden_states]
        features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2)  # [B, T, D] -> [B, D, T]

    def encode_lms(self, lms: torch.Tensor, return_layers: bool = False):
        r"""
        Encodes an arbitrary log mel-spectrogram and returns the corresponding embeddings.

        M2D can only receive fixed-size spectrograms as inputs because of absolute positional encodings.
        One has therefore to split audio into chunks if its length differs from the training data.

        Args:

        Returns:
            torch.Tensor | List[torch.Tensor]: output embeddings,
                shape (batch_size, num_freq_patches * num_time_patches, embed_dim) if cfg.flat_features else
                shape (batch_size, num_time_patches, num_freq_patches * embed_dim)
        """
        x = lms

        patch_fbins = 5  # self.backbone.grid_size[0]  WARNING save that shit
        unit_frames = self.backbone.img_size[1]
        patch_frames = self.backbone.patch_size[1]
        embed_d = self.backbone.patch_embed.proj.out_channels
        pad_frames = (patch_frames - x.shape[-1] % patch_frames) % patch_frames
        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, pad_frames))

        # if hasattr(self.backbone, "time_pos_encoding") \
        #     and not self.backbone.time_pos_encoding.startswith("abs"):
        #     unit_frames = x.shape[-1]
        chunks = (x.shape[-1] + unit_frames - 1) // unit_frames

        embeddings = []

        # flatten all patch embeddings
        for i in range(chunks):
            emb = self.backbone.forward(
                x[..., i * unit_frames:(i + 1) * unit_frames],
                return_layers=return_layers
            )
            if not self.cfg.flat_features:
                # rearrange dimensions: (..., f*t, d) -> (..., t, f*d)
                if self.flattened:
                    emb = emb.view(*emb.shape[:-2], patch_fbins, -1, embed_d).transpose(-3, -2).flatten(-2)
                else:
                    emb = emb.view(*emb.shape[:-3], patch_fbins, -1, embed_d).transpose(-3, -2).flatten(-2)
                    pass

            embeddings.append(emb)

        # concatenate chunks in the time axis
        x = torch.cat(embeddings, axis=-2)

        return x if x.ndim == 3 else [x_ for x_ in x]
