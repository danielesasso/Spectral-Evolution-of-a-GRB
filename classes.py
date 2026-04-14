from typing import Iterable
import torch
from ClassiPyGRB import SWIFT
from torch import nn
from torch.utils.data import Dataset

from functions import extract_t90, make_t90_lookup, resample_light_curve, zscore_per_grb

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class GRBDataset(Dataset):
    """
    PyTorch dataset for binary short/long GRB classification.

    Output:
        X: float tensor shaped (time, channels)
        y: float tensor scalar, 0.0 for short and 1.0 for long (T90 > 2 s)
    """

    def __init__(
        self,
        grb_names: Iterable[str],
        channel_columns: list[str],
        time_column: str,
        target_length: int,
        swift_resolution: int,
        normalize: str,
        skip_bad: bool,
    ) -> None:
        self.grb_names = list(grb_names)
        self.channel_columns = channel_columns
        self.time_column = time_column
        self.target_length = target_length
        self.swift = SWIFT(res=swift_resolution)
        self.t90_lookup = make_t90_lookup(self.swift.summary_table())
        self.samples: list[tuple[torch.Tensor, torch.Tensor, str, float]] = []
        self.skipped: list[tuple[str, str]] = []

        if normalize != "zscore":
            raise ValueError("Only per-GRB zscore normalization is currently implemented")

        grb_iterator = self.grb_names
        if tqdm is not None:
            grb_iterator = tqdm(self.grb_names, desc="Loading GRBs", unit="GRB")

        for name in grb_iterator:
            try:
                df = self.swift.obtain_data(name=name)
                if not hasattr(df, "columns"):
                    raise RuntimeError(str(df))
                t90 = extract_t90(df, grb_name=name, t90_lookup=self.t90_lookup)
                signal = resample_light_curve(
                    df,
                    target_length=target_length,
                    channel_columns=channel_columns,
                    time_column=time_column,
                )
                signal = zscore_per_grb(signal)
                label = 1.0 if t90 > 2.0 else 0.0
                x = torch.tensor(signal, dtype=torch.float32)
                y = torch.tensor(label, dtype=torch.float32)
                self.samples.append((x, y, name, t90))
            except Exception as exc:
                if not skip_bad:
                    raise
                self.skipped.append((name, str(exc)))

        if not self.samples:
            raise RuntimeError("No valid GRBs were loaded. Check names, columns, and T90 metadata.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, _, _ = self.samples[idx]
        return x, y


class GRBConvNet(nn.Module):
    """1D CNN over time; input is (batch, time, channels)."""

    def __init__(self, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, time, channels) -> (batch, channels, time)
        logits = self.classifier(self.features(x))
        return logits.squeeze(1)
