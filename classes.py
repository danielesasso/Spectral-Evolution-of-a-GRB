from pathlib import Path

import h5py
import torch
from torch import nn
from torch.utils.data import Dataset


class GRBHDF5Dataset(Dataset):
    """
    PyTorch dataset backed by the local ClassiPyGRB HDF5 cache.

    Expected HDF5 datasets:
        X: float array shaped (n_grbs, time, channels)
        y: binary labels, 0.0 for short and 1.0 for long
        names: GRB names
        t90: T90 durations
    """

    def __init__(self, h5_file: str | Path) -> None:
        self.h5_file = Path(h5_file)
        if not self.h5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file}")

        with h5py.File(self.h5_file, "r") as h5:
            required_datasets = {"X", "y", "names", "t90"}
            missing = required_datasets.difference(h5.keys())
            if missing:
                raise ValueError(f"HDF5 file is missing datasets: {sorted(missing)}")

            self.x = torch.tensor(h5["X"][:], dtype=torch.float32)
            self.y = torch.tensor(h5["y"][:], dtype=torch.float32)
            self.names = [decode_h5_string(name) for name in h5["names"][:]]
            self.t90 = [float(value) for value in h5["t90"][:]]

            if "channel_columns" in h5:
                self.channel_columns = [
                    decode_h5_string(channel) for channel in h5["channel_columns"][:]
                ]
            else:
                self.channel_columns = [f"channel_{idx}" for idx in range(self.x.shape[2])]

            self.label_rule = str(h5.attrs.get("label_rule", "0 short, 1 long"))
            self.cache_normalization = str(h5.attrs.get("normalize", "unknown"))

        if self.x.ndim != 3:
            raise ValueError(f"Expected X to have shape (n, time, channels), got {tuple(self.x.shape)}")
        if len(self.x) != len(self.y):
            raise ValueError("X and y contain different numbers of GRBs")

        self.labels = [int(value.item()) for value in self.y]
        self.samples = [
            (self.x[idx], self.y[idx], self.names[idx], self.t90[idx])
            for idx in range(len(self.y))
        ]

    @property
    def num_channels(self) -> int:
        return int(self.x.shape[2])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class GRBConvNet(nn.Module):
    """1D CNN over time; input is (batch, time, channels)."""

    def __init__(self, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden * 2, hidden * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, time, channels) -> (batch, channels, time)
        logits = self.classifier(self.features(x))
        return logits.squeeze(1)
