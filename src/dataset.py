"""PyTorch dataset joining per-clip feature arrays (.npy) with labels.csv (filename, label, subject_id)."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _valid_mask(arr: np.ndarray) -> np.ndarray:
    """Rows that are real frames, not the zero padding added by extract_features.py."""
    return np.abs(arr).sum(axis=1) > 0


class DeceptionDataset(Dataset):
    def __init__(self, features_dir: str, labels_csv: str, normalize: bool = True):
        self.features_dir = features_dir
        self.normalize = normalize
        labels = pd.read_csv(labels_csv)
        self.labels_df = labels[
            labels["filename"].apply(lambda f: os.path.exists(os.path.join(features_dir, f"{f}.npy")))
        ].reset_index(drop=True)

        if len(self.labels_df) == 0:
            raise RuntimeError(f"No .npy files in {features_dir} match the rows of {labels_csv}.")

        if normalize:
            self.fit_normalization()

    def fit_normalization(self, indices=None):
        """Compute per-feature z-score statistics, ignoring padding.

        Pass the training indices of each fold so test statistics never leak into training.
        """
        if indices is None:
            indices = range(len(self.labels_df))
        frames = []
        for i in indices:
            arr = np.load(os.path.join(self.features_dir, f"{self.labels_df.iloc[i]['filename']}.npy"))
            frames.append(arr[_valid_mask(arr)])
        stacked = np.concatenate(frames, axis=0)
        self.mean = stacked.mean(axis=0, keepdims=True)
        self.std = stacked.std(axis=0, keepdims=True) + 1e-6

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        arr = np.load(os.path.join(self.features_dir, f"{row['filename']}.npy"))
        mask = _valid_mask(arr)
        if self.normalize:
            arr = (arr - self.mean) / self.std
            arr[~mask] = 0.0
        return (torch.tensor(arr, dtype=torch.float32),
                torch.tensor(int(row["label"]), dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool))

    @property
    def n_features(self):
        return np.load(os.path.join(self.features_dir, f"{self.labels_df.iloc[0]['filename']}.npy")).shape[1]
