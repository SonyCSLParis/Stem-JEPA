"""Masked Modeling Duo (M2D) Wrapper for EVAR.

Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input
https://ieeexplore.ieee.org/document/10097236/

Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation
https://arxiv.org/abs/2305.14079
"""

import traceback
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
from nnAudio.features import MelSpectrogram

try:  # we need to manually import src for hydra to re-instantiate the model
    import sys
    sys.path.append('..')
    import src
except Exception:
    print(f'(For M2D users) Build your EVAR in your M2D folder.')
    print('vvvv')
    traceback.print_exc()
    print('^^^^')
from .ar_base import BaseAudioRepr, calculate_norm_stats, normalize_spectrogram


def get_to_melspec(cfg):
    to_spec = MelSpectrogram(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.window_size,
        hop_length=cfg.hop_size,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        center=True,
        power=2,
        verbose=False,
    )
    return to_spec


class ARCLAP(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        weight_file = Path(cfg.weight_file)
        train_cfg = OmegaConf.load(weight_file.parents[1] / "config.yaml")
        self.backbone = instantiate(train_cfg.model.audio_encoder.encoder)
        self._init_weights(weight_file)
        self.backbone.eval()

        self.to_spec = get_to_melspec(cfg)

    def _init_weights(self, weight_file: str | Path) -> None:
        state_dict = torch.load(weight_file, map_location=torch.device('cpu'))["state_dict"]
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('audio_encoder.'):
                if k[14:].startswith("_orig_mod."):
                    encoder_state_dict[k[24:]] = v
                else:
                    encoder_state_dict[k[14:]] = v
        self.backbone.load_state_dict(encoder_state_dict)

    def to_feature(self, batch_audio):
        # raw -> spectrogram, and normalize
        x = self.to_spec(batch_audio)
        x = (x + torch.finfo().eps).log()
        x = x.unsqueeze(1)
        return x

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def encode_frames(self, batch_audio):
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

    def forward(self, batch_audio):
        if hasattr(self, 'lms_mode'):
            x = self.encode_frames_lms(batch_audio)
        else:
            x = self.encode_frames(batch_audio)
        return x.mean(dim=-1)  # [B, D, T] -> [B, D]

    def precompute_lms(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, lambda x: x)
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

        patch_fbins = self.backbone.grid_size[0]
        unit_frames = self.backbone.img_size[1]
        patch_frames = self.backbone.patch_size[1]
        embed_d = self.backbone.patch_embed.proj.out_channels
        pad_frames = (patch_frames - x.shape[-1] % patch_frames) % patch_frames
        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, pad_frames))

        if hasattr(self.backbone, "time_pos_encoding") \
            and not self.backbone.time_pos_encoding.startswith("abs"):
            unit_frames = x.shape[-1]
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
                emb = emb.view(*emb.shape[:-2], patch_fbins, -1, embed_d).transpose(-3, -2).flatten(-2)

            embeddings.append(emb)

        # concatenate chunks in the time axis
        x = torch.cat(embeddings, axis=-2)

        return x if x.ndim == 3 else [x_ for x_ in x]
