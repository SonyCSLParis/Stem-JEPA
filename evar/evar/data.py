"""Dataset handlings.

Balanced sampler is supported for multi-label tasks.
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchaudio

from .sampler import BalancedRandomSampler, InfiniteSampler


class BaseRawAudioDataset(Dataset):
    def __init__(self, unit_samples, tfms=None, random_crop=False):
        self.unit_samples = unit_samples
        self.tfms = tfms
        self.random_crop = random_crop

    def __len__(self):
        raise NotImplementedError

    def get_audio(self, index: int) -> torch.Tensor:
        raise NotImplementedError

    def get_label(self, index):
        return None # implement me

    def __getitem__(self, index):
        wav = self.get_audio(index) # shape is expected to be (self.unit_samples,)

        # Trim or stuff padding
        l = wav.size(1)
        if l > self.unit_samples:
            start = np.random.randint(l - self.unit_samples) if self.random_crop else 0
            wav = wav[:, start:start + self.unit_samples]
        elif l < self.unit_samples:
            wav = F.pad(wav, (0, self.unit_samples - l), mode='constant', value=0)
        wav = wav.to(torch.float)
        wav.squeeze_(0)

        # Apply transforms
        if self.tfms is not None:
            wav = self.tfms(wav)

        # Return item
        label = self.get_label(index)
        return wav if label is None else (wav, label)


class WavDataset(BaseRawAudioDataset):
    def __init__(self,
                 cfg,
                 split: str,
                 holdout_fold: int = 1,
                 label_col: str = "label",
                 always_one_hot=False,
                 random_crop=True,
                 classes=None):
        super().__init__(cfg.unit_samples, tfms=None, random_crop=random_crop)
        self.cfg = cfg
        self.split = split

        df = pd.read_csv(cfg.task_metadata).rename(columns={label_col: "label"})

        # Multi-fold, leave one out of cross validation.
        if 'split' not in df.columns:
            assert 'fold' in df.columns, '.csv needs to have either "split" or "fold" column...'
            # Mark split either 'train' or 'test', no 'val' or 'valid' used in this implementation.
            df['split'] = df.fold.apply(lambda f: 'test' if f == holdout_fold else 'train')
        df = df[df.split == split].reset_index()
        self.df = df
        self.multi_label = df.label.map(str).str.contains(',').sum() > 0

        if self.multi_label or always_one_hot:
            # one-hot
            oh_enc = MultiLabelBinarizer(classes=classes)
            self.labels = torch.tensor(oh_enc.fit_transform([str(ls).split(',') for ls in df.label]), dtype=torch.float32)
            self.classes = oh_enc.classes_
        elif self.df["label"].dtype == "float":
            # we map classes to log-tempo and rescale so that classes roughly correspond to tolerance of Acc1
            if label_col == "tempo":
                self.labels = torch.tensor(self.df.label).log().div(2 * np.log(1.04)).round().long()
                self.classes = torch.arange(self.labels.max() + 1)
            else:
                self.labels = torch.tensor(self.df.label, dtype=torch.float32)
                self.classes = None
                raise NotImplementedError
        else:
            # single valued gt values
            self.classes = sorted(df.label.unique()) if classes is None else classes
            self.labels = torch.tensor(df.label.map({l: i for i, l in enumerate(self.classes)}).values)

    def __len__(self):
        return len(self.df)

    def get_audio(self, index):
        filename = self.cfg.task_data + '/' + self.df.file_name.values[index]
        # print(index, filename)
        wav, sr = torchaudio.load(filename)
        assert sr == self.cfg.sample_rate, f'Convert .wav files to {self.cfg.sample_rate} Hz. {filename} has {sr} Hz.'
        return wav

    def get_label(self, index):
        return self.labels[index]


class ASSpectrogramDataset(WavDataset):
    """Spectrogram audio dataset class for M2D AudioSet 2M fine-tuning."""

    def __init__(self, cfg, split, always_one_hot=False, random_crop=True, classes=None):
        super().__init__(cfg, split, holdout_fold=1, always_one_hot=always_one_hot, random_crop=random_crop, classes=classes)

        self.df.file_name = self.df.file_name.str.replace('.wav', '.npy', regex=False)
        self.folder = Path(cfg.data_path)
        self.crop_frames = cfg.dur_frames
        self.random_crop = random_crop

        print(f'Dataset contains {len(self.df)} files without normalizing stats.')

    def get_audio_file(self, filename):
        lms = torch.tensor(np.load(filename))
        return lms

    def get_audio(self, index):
        filename = self.folder / self.df.file_name.values[index]
        return self.get_audio_file(filename)

    def complete_audio(self, lms):
        # Trim or pad
        start = 0
        l = lms.shape[-1]
        if l > self.crop_frames:
            start = int(torch.randint(l - self.crop_frames, (1,))[0]) if self.random_crop else 0
            lms = lms[..., start:start + self.crop_frames]
        elif l < self.crop_frames:
            pad_param = []
            for i in range(len(lms.shape)):
                pad_param += [0, self.crop_frames - l] if i == 0 else [0, 0]
            lms = F.pad(lms, pad_param, mode='constant', value=0)
        self.last_crop_start = start
        lms = lms.to(torch.float)

        return lms

    def __getitem__(self, index):
        lms = self.get_audio(index)
        lms = self.complete_audio(lms)
        # Return item
        label = self.get_label(index)
        return lms if label is None else (lms, label)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(crop_frames={self.crop_frames}, random_crop={self.random_crop}, '
        return format_string


def create_as_dataloader(cfg, batch_size, always_one_hot=False, balanced_random=False, pin_memory=True, num_workers=8):
    batch_size = batch_size or cfg.batch_size
    train_dataset = ASSpectrogramDataset(cfg, 'train', always_one_hot=always_one_hot, random_crop=True)
    valid_dataset = ASSpectrogramDataset(cfg, 'valid', always_one_hot=always_one_hot, random_crop=True,
        classes=train_dataset.classes)
    test_dataset = ASSpectrogramDataset(cfg, 'test', always_one_hot=always_one_hot, random_crop=False,
        classes=train_dataset.classes)

    weights = pd.read_csv('evar/metadata/weight_as.csv').weight.values
    train_sampler = WeightedRandomSampler(weights, num_samples=200000, replacement=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=pin_memory,
                                            num_workers=num_workers) if balanced_random else \
                   torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                                            num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)
    train_loader.lms_mode = True

    return train_loader, valid_loader, test_loader, train_dataset.multi_label


def create_dataloader(cfg,
                      fold: int = 1,
                      seed: int = 42,
                      batch_size=None,
                      label_col: str = "label",
                      always_one_hot=False,
                      balanced_random: bool = False,
                      pin_memory=True,
                      num_workers=2):
    r"""

    Args:
        cfg
        batch_size (int): batch size, if provided it overrides the one provided in the config
        always_one_hot (bool):
        balanced_random (bool): ???
    """
    if Path(cfg.task_metadata).stem == 'as':
        return create_as_dataloader(cfg, batch_size=batch_size, always_one_hot=always_one_hot, balanced_random=balanced_random, pin_memory=pin_memory, num_workers=num_workers)

    batch_size = batch_size or cfg.batch_size
    train_dataset = WavDataset(cfg,
                               'train',
                               holdout_fold=fold,
                               label_col=label_col,
                               always_one_hot=always_one_hot,
                               random_crop=True)
    valid_dataset = WavDataset(cfg,
                               'valid',
                               holdout_fold=fold,
                               label_col=label_col,
                               always_one_hot=always_one_hot,
                               random_crop=True,
                               classes=train_dataset.classes)
    test_dataset = WavDataset(cfg,
                              'test',
                              holdout_fold=fold,
                              label_col=label_col,
                              always_one_hot=always_one_hot,
                              random_crop=False,
                              classes=train_dataset.classes)

    train_sampler = BalancedRandomSampler(train_dataset, batch_size, seed) if train_dataset.multi_label else \
        InfiniteSampler(train_dataset, batch_size, seed, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=pin_memory,
                                            num_workers=num_workers) if balanced_random else \
                   partial(DataLoader,
                           dataset=train_dataset,
                           shuffle=False,  # since embeddings are cached we don't shuffle to preserve order
                           pin_memory=pin_memory,
                           num_workers=num_workers)
    valid_loader = partial(DataLoader,
                           dataset=valid_dataset,
                           shuffle=False,
                           pin_memory=pin_memory,
                           num_workers=num_workers)
    test_loader = partial(DataLoader,
                          dataset=test_dataset,
                          shuffle=False,
                          pin_memory=pin_memory,
                          num_workers=num_workers)

    return train_loader, valid_loader, test_loader, train_dataset.multi_label
