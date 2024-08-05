"""Masked Modeling Duo (M2D) Wrapper for EVAR.

Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input
https://ieeexplore.ieee.org/document/10097236/

Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation
https://arxiv.org/abs/2305.14079
"""
import os
import traceback

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

try:
    import sys
    sys.path.append("external/audiomae")
    import external.audiomae.models_mae as models_vit
    from external.audiomae.util.patch_embed import PatchEmbed_new
except ModuleNotFoundError as e:
    print(e)
    raise ModuleNotFoundError("Please go to folder evar/external and run the following command: "
                              "`git clone https://github.com/facebookresearch/AudioMAE.git audiomae`.")
except Exception:
    print('vvvv')
    traceback.print_exc()
    print('^^^^')
from .ar_base import BaseAudioRepr, calculate_norm_stats, normalize_spectrogram


def get_to_melspec(cfg):
    return partial(torchaudio.compliance.kaldi.fbank,
                   htk_compat=True,
                   sample_frequency=cfg.sample_rate,
                   use_energy=False,
                   window_type=cfg.window,
                   num_mel_bins=cfg.n_mels,
                   dither=0.,
                   frame_shift=1000 * cfg.hop_size // cfg.sample_rate)


class ARAudioMAE(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        print(cfg)

        self._init_weights(cfg.weight_file)
        self.backbone.eval()

        self._to_spec = get_to_melspec(cfg)

    def _init_weights(self, weight_file: str | Path) -> None:
        if not os.path.exists(weight_file):
            self._download_weights(weight_file)

        checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))

        # instantiate model, cf https://github.com/facebookresearch/AudioMAE/blob/main/main_finetune_as.py#L361C6-L361C6
        args = checkpoint["args"]
        model = models_vit.__dict__[args.model](
            # num_classes=args.nb_classes,
            # drop_path_rate=args.drop_path,
            # global_pool=args.global_pool,
            mask_2d=args.mask_2d,
            use_custom_patch=args.use_custom_patch,
            decoder_mode=0
        )
        if args.audio_exp:
            img_size = (1024, 128)  # 1024, 128
            in_chans = 1
            emb_dim = 768
            if args.model == "vit_small_patch16":
                emb_dim = 384
            if args.use_custom_patch:
                model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=16, in_chans=1, embed_dim=emb_dim,
                                                   stride=10)
                model.pos_embed = nn.Parameter(torch.zeros(1, 1212 + 1, emb_dim),
                                               requires_grad=False)  # fixed sin-cos embedding
            else:
                model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16, 16), in_chans=1,
                                                   embed_dim=emb_dim, stride=16)  # no overlap. stride=img_size=16
                num_patches = model.patch_embed.num_patches
                # num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
                model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim),
                                               requires_grad=False)  # fixed sin-cos embedding

        # we only use the encoder for downstream tasks
        state_dict = {k: v for k, v in checkpoint["model"].items() if not k.startswith("decoder_")}
        model.load_state_dict(state_dict, strict=False)

        self.backbone = model

    def to_spec(self, batch_audio):
        return torch.stack([self._to_spec(audio.unsqueeze_(0)) for audio in batch_audio])


    def to_feature(self, batch_audio):
        # raw -> spectrogram, and normalize
        x = self.to_spec(batch_audio)
        x.unsqueeze_(1)
        return x

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def encode_frames(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x)
        x = self.augment_if_training(x)
        hidden_states = self.encode_lms(x)

        return hidden_states.transpose(1, 2)  # [B, T, D] -> [B, D, T]

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
        hidden_states = self.encode_lms(x)

        return hidden_states.transpose(1, 2)  # [B, T, D] -> [B, D, T]

    def encode_lms(self, lms: torch.Tensor):
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

        img_size = self.backbone.patch_embed.img_size
        patch_size = self.backbone.patch_embed.patch_size
        grid_size = tuple(s // p for s, p in zip(img_size, patch_size))

        patch_fbins = grid_size[1]
        unit_frames = img_size[0]
        patch_frames = patch_size[0]
        embed_d = self.backbone.embed_dim
        # pad_frames = unit_frames - x.shape[-2] % unit_frames
        # if pad_frames > 0:
        #     x = torch.nn.functional.pad(x, (0, 0, 0, pad_frames))
        pad_frames = (patch_frames - x.shape[-2] % patch_frames) % patch_frames
        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_frames))

        chunks = (x.shape[-2] + unit_frames - 1) // unit_frames

        embeddings = []

        # flatten all patch embeddings
        for i in range(chunks):
            emb = self.backbone.forward_encoder_no_mask(x[..., i * unit_frames:(i + 1) * unit_frames, :])[:, 1:, :]
            if not self.cfg.flat_features:
                # rearrange dimensions: (..., t*f, d) -> (..., t, f*d)
                emb = emb.view(*emb.shape[:-2], -1, patch_fbins, embed_d).flatten(-2)

            embeddings.append(emb)

        # concatenate chunks in the time axis
        x = torch.cat(embeddings, axis=-2)

        return x if x.ndim == 3 else [x_ for x_ in x]

    @staticmethod
    def _download_weights(filename):
        url = "https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link"
        raise FileNotFoundError(f"Please download the weights and place them at location {filename}. "
                                f"Download link: {url}")
