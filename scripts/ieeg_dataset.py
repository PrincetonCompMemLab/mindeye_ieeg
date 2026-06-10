"""Torch ``Dataset`` pairing rep-averaged iEEG features with bigG CLIP targets.

Distinct from the ``ieeg_preproc`` package (which produces the numpy
``(conditions, features)`` matrix): this is the thin, model-facing wrapper that MindEye2
consumes. It z-scores the iEEG features per feature — computing mean/std on the **train**
split and **reusing** them on val/test to avoid leakage — and yields tensors shaped for the
MindEye2 ridge layer: ``(1, F)`` per sample (leading ``seq_len=1``) alongside the
``(256, 1664)`` CLIP target.
"""

import torch
from torch.utils.data import Dataset


class iEEGDataset(Dataset):
    def __init__(self, ieeg_path, clip_path, mean=None, std=None, eps=1e-6):
        ieeg = torch.load(ieeg_path, weights_only=True).float()        # (N, F)
        self.clip = torch.load(clip_path, weights_only=True).float()    # (N, 256, 1664)

        # Train computes its own stats; val/test pass the train stats back in.
        self.mean = ieeg.mean(0) if mean is None else mean
        self.std = ieeg.std(0) if std is None else std
        # eps keeps near-constant features from blowing up / producing NaNs.
        self.ieeg = (ieeg - self.mean) / (self.std + eps)

    def __len__(self):
        return self.ieeg.shape[0]

    def __getitem__(self, idx):
        return self.ieeg[idx].unsqueeze(0), self.clip[idx]   # (1, F), (256, 1664)
