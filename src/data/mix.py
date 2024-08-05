import json
import logging
import random
import os
from pathlib import Path
from typing import Any, List, Mapping, Tuple
from tqdm import trange

import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample

from lightning import LightningDataModule

from nnAudio.features import MelSpectrogram

from src.utils.audio import audio_info, audio_read
from src.utils.copy import copy_to_compute_node
from src.utils.running_stats import OnlineStatsCalculator


log = logging.getLogger(__name__)


class ToLMS(torch.nn.Module):
    def __init__(self, resample: int | None = None, **kwargs):
        super(ToLMS, self).__init__()
        norm_stats = kwargs.pop("norm_stats")
        if norm_stats is None:
            log.warning("No normalization stats provided, using (0, 1) by default")
            norm_stats = (0., 1.)
        self.mean, self.std = norm_stats

        if resample is not None:
            self.resample = Resample(kwargs["sr"], resample)
            kwargs["sr"] = resample
        else:
            self.resample = torch.nn.Identity()
        
        self.to_mel = MelSpectrogram(**kwargs)
        self.eps = torch.finfo().eps

    def forward(self, x: torch.Tensor):
        r"""Converts the input waveform into normalized log-scaled mel-spectrograms.
        If other args are passed, they are returned as is"""
        x = self.resample(x)
        x = self.to_mel(x)

        # log-scale
        x.add_(self.eps).log_()

        # normalize
        x.sub_(self.mean).div_(self.std)

        return x.unsqueeze(-3)


class MixStemDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 duration: float,
                 sample_rate: int = 44100,
                 silence_threshold: float = 1e-2,
                 num_sources_context: str = 'max',
                 num_trials: int = 1) -> None:
        self.data_path = Path(data_path)
        self.duration = duration
        self.num_frames = int(duration * sample_rate)
        self.sample_threshold = silence_threshold
        self.num_sources_context = num_sources_context
        self.num_trials = num_trials

        self.song_names = sorted([f.relative_to(self.data_path) for f in self.data_path.glob("**/*.wav")])

    def __len__(self):
        return len(self.song_names)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        r"""

        Args:
            idx (int): Index of the track to load

        Returns:
            torch.Tensor: stems or mix, depending on `self.mix_tracks`,
                shape (num_frames) if `self.mix_tracks` and (num_stems, num_frames) otherwise
        """
        song_path = self.data_path / self.song_names[idx]

        # Get the maximum number of frames among selected stems for both mixes
        duration = audio_info(song_path).duration

        for _ in range(self.num_trials):
            # Choose a random frame offset within the valid range for both mixes
            if duration > self.duration:
                offset = random.random() * (duration - self.duration)

                # Retrieve the stems
                stems, sr = audio_read(song_path, seek_time=offset, duration=self.duration, pad=False)
            
            else:
                stems, sr = audio_read(song_path, duration=self.duration, pad=True)

            # Even if there are theoretically always 4 stems in the track, in practice some of them can be inactive.
            # This has to be handled carefully

            # make sure we don't put silence as target
            stem_amplitudes = stems.abs().max(dim=1).values
            target_index = self.pick_index(stem_amplitudes)
            if target_index is None:
                continue

            num_stems = stems.size(0)
            if self.num_sources_context == "one":  # keep only one source for context
                mask = torch.zeros(num_stems, dtype=torch.bool)
                context_index = torch.randint(num_stems, ())
                if context_index == target_index:
                    context_index = (context_index + 1) % num_stems
                mask[context_index] = True

            elif self.num_sources_context == "one-plus":  # keep only one source for context
                candidates_mask = torch.ones(num_stems, dtype=torch.bool)
                candidates_mask[target_index] = False

                context_index = self.pick_index(stem_amplitudes[candidates_mask])
                if context_index is None:
                    continue

                mask = torch.zeros(num_stems, dtype=torch.bool)
                mask[context_index] = True
            
            elif self.num_sources_context == "uniform":
                stem_amplitudes[target_index] = 0

                active_sources = stem_amplitudes > self.sample_threshold
                num_active_sources = active_sources.sum()

                mask = torch.zeros(num_stems, dtype=torch.bool)

                if num_active_sources > 0:
                    num_context_sources = torch.randint(1, num_active_sources + 1, ())
                    context_sources = torch.multinomial(active_sources.float(), num_context_sources, replacement=False)
                    mask[context_sources] = True
        
            else:  # keep all sources but the target as context
                if self.num_sources_context != "max":
                    log.warning(f"Weird config for `num_sources_context`: {self.num_sources_context}")
                mask = torch.ones(num_stems, dtype=torch.bool)
                mask[target_index] = False

            context = stems[mask].sum(dim=0)
            target = stems[target_index].squeeze_(0)

            return context, target, target_index
        
        # if one cannot find a proper chunk fuck it we keep the last one
        log.warning(f"Unable to find a non-silent chunk in track {song_path}")
        target_index = torch.randint(stems.size(0), (1,))

        mask = torch.ones(stems.size(0), dtype=torch.bool)
        mask[target_index] = 0

        context = stems[mask].sum(dim=0)
        target = stems[target_index].squeeze_(0)

        return context, target, target_index

    def pick_index(self, amplitudes: torch.Tensor):
        candidates = (amplitudes > self.sample_threshold).nonzero()
        num_candidates = len(candidates)
        if num_candidates == 0:
            return None
        idx = torch.randint(num_candidates, ())
        return candidates[idx]

    def get_audio(self, idx):
        song_path = self.data_path / self.song_names[idx]
        stems, sr = audio_read(song_path)
        return stems


class MixStemsDatamodule(LightningDataModule):
    def __init__(self,
                 dataset: Mapping[str, Any],
                 dataloader: Mapping[str, Any],
                 transform: torch.nn.Module | None = None,
                 local_dir: bool = True,
                 compute_stats: bool = False):
        super(MixStemsDatamodule, self).__init__()
        self.dataset_kwargs = dataset

        # get batch size per device
        devices = dataloader.pop("devices", 1)
        if not isinstance(devices, int):
            devices = len(devices)
        batch_size = dataloader.pop("batch_size", 256) // devices

        self.dataloader_kwargs = dataloader
        self.dataloader_kwargs["batch_size"] = batch_size

        self.transform = transform or torch.nn.Identity()

        # placeholders
        self.local_dir = local_dir
        self.compute_stats = compute_stats
        self.dataset = None
    
    def prepare_data(self):
        if not self.local_dir:
            return

        data_path = self.dataset_kwargs.pop("data_path")
        local_data_path = copy_to_compute_node(data_path)
        
        self.dataset_kwargs["data_path"] = local_data_path

    def setup(self, stage: str | None = None):
        self.dataset = MixStemDataset(**self.dataset_kwargs)

        if self.compute_stats:
            self.compute_stats()

    def train_dataloader(self):
        # Find current device
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        # move transforms to the appropriate device
        self.transform = self.transform.to(device)

        return DataLoader(self.dataset, shuffle=True, **self.dataloader_kwargs)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        ctx, tgt, idx = batch
        return self.transform(ctx), self.transform(tgt), idx

    def compute_stats(self, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        running_stats = OnlineStatsCalculator()

        self.transform = self.transform.to(device)

        print("Computing dataset statistics...")
        try:
            for i in trange(len(self.dataset)):
                audio = self.dataset.get_audio(i)
                if audio.size(0) < 4:
                    continue
                spec = self.transform(audio.to(device))
                running_stats.update(spec.mean(dim=-1))
        finally:
            print("== Dataset statistics ==")
            print(f"mean: {running_stats.get_mean().cpu().item():.3f}, std: {running_stats.get_std().cpu().item():.3f}")
            exit(0)
