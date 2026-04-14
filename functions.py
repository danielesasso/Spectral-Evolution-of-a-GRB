import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# seed
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# uso di mps
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_grb_names(
    names: list[str] | None,
    names_file: str | Path | None,
    default_grb_names: list[str],
) -> list[str]:
    """Load GRB names from CLI values, a text file, or a CSV file."""
    loaded_names: list[str] = []

    if names:
        loaded_names.extend(names)

    if names_file:
        path = Path(names_file)
        if not path.exists():
            raise FileNotFoundError(f"Names file not found: {path}")

        if path.suffix.lower() == ".csv":
            table = pd.read_csv(path)
            candidate_columns = ["name", "grb", "GRB", "grb_name", "GRB_name"]
            name_column = next((col for col in candidate_columns if col in table.columns), None)
            if name_column is None:
                raise ValueError(
                    f"CSV names file must contain one of these columns: {candidate_columns}"
                )
            loaded_names.extend(table[name_column].dropna().astype(str).tolist())
        else:
            loaded_names.extend(
                line.strip() for line in path.read_text().splitlines() if line.strip()
            )

    cleaned = []
    seen = set()
    for name in loaded_names or default_grb_names:
        name = str(name).strip()
        if name and name not in seen:
            cleaned.append(name)
            seen.add(name)
    return cleaned


def make_t90_lookup(summary_table: pd.DataFrame) -> dict[str, float]:
    """Create a GRB name -> T90 lookup from the ClassiPyGRB Swift summary table."""
    required_columns = {"GRBname", "T90"}
    missing_columns = required_columns.difference(summary_table.columns)
    if missing_columns:
        raise ValueError(f"Swift summary table is missing columns: {sorted(missing_columns)}")

    table = summary_table.copy()
    table["GRBname"] = table["GRBname"].astype(str).str.strip()
    table["T90"] = pd.to_numeric(table["T90"], errors="coerce")
    table = table[table["GRBname"].ne("") & table["T90"].notna()]
    return dict(zip(table["GRBname"], table["T90"].astype(float)))


def extract_t90(
    df: pd.DataFrame,
    grb_name: str | None = None,
    t90_lookup: dict[str, float] | None = None,
) -> float:
    """Find a T90 value in common dataframe columns or metadata."""
    if grb_name and t90_lookup and grb_name in t90_lookup:
        return float(t90_lookup[grb_name])

    for key, value in df.attrs.items():
        if "t90" in str(key).lower():
            t90 = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if np.isfinite(t90):
                return float(t90)

    for column in df.columns:
        if "t90" in str(column).lower():
            values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()
            values = values[np.isfinite(values)]
            if values.size:
                return float(np.nanmedian(values))

    raise ValueError("No finite T90 value found")


def zscore_per_grb(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize each channel inside one GRB only; no global leakage."""
    mean = np.nanmean(signal, axis=0, keepdims=True)
    std = np.nanstd(signal, axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    normalized = (signal - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def pad_or_truncate(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Keep temporal order and pad short curves with zeros at the end."""
    if signal.shape[0] >= target_length:
        return signal[:target_length]

    padded = np.zeros((target_length, signal.shape[1]), dtype=np.float32)
    padded[: signal.shape[0]] = signal
    return padded


def resample_light_curve(
    df: pd.DataFrame,
    target_length: int,
    channel_columns: list[str],
    time_column: str,
) -> np.ndarray:
    """
    Convert variable-length multiband Swift curves to (target_length, channels).

    Assumption: when a valid time column exists, interpolation onto an evenly
    spaced grid is preferable to truncation because it preserves the full burst
    time span. If time is unusable, the code falls back to pad/truncate.
    """
    missing_channels = [col for col in channel_columns if col not in df.columns]
    if missing_channels:
        raise ValueError(f"Missing rate columns: {missing_channels}")

    rates = df[channel_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if rates.size == 0:
        raise ValueError("Empty light curve")

    if time_column not in df.columns:
        return pad_or_truncate(np.nan_to_num(rates, nan=0.0), target_length)

    time = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=np.float64)
    valid_rows = np.isfinite(time)
    if valid_rows.sum() < 2:
        return pad_or_truncate(np.nan_to_num(rates, nan=0.0), target_length)

    time = time[valid_rows]
    rates = rates[valid_rows]
    order = np.argsort(time)
    time = time[order]
    rates = rates[order]

    unique_time, unique_idx = np.unique(time, return_index=True)
    time = unique_time
    rates = rates[unique_idx]
    if time.size < 2 or np.isclose(time[0], time[-1]):
        return pad_or_truncate(np.nan_to_num(rates, nan=0.0), target_length)

    target_time = np.linspace(time[0], time[-1], target_length)
    resampled_channels = []
    for channel_idx in range(rates.shape[1]):
        channel = rates[:, channel_idx]
        finite = np.isfinite(channel)
        if finite.sum() == 0:
            resampled = np.zeros(target_length, dtype=np.float32)
        elif finite.sum() == 1:
            resampled = np.full(target_length, channel[finite][0], dtype=np.float32)
        else:
            resampled = np.interp(target_time, time[finite], channel[finite]).astype(np.float32)
        resampled_channels.append(resampled)

    return np.stack(resampled_channels, axis=1)


def split_dataset(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    """Random split after preprocessing; set seed for reproducibility."""
    n = len(dataset)
    if n < 3:
        raise ValueError("Need at least 3 valid labeled GRBs for train/val/test splits")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_size = max(1, int(n * train_ratio))
    val_size = max(1, int(n * val_ratio))
    if train_size + val_size >= n:
        train_size = n - 2
        val_size = 1

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def make_dataloaders(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def binary_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int, int]:
    preds = (torch.sigmoid(logits) >= 0.5).long()
    labels = targets.long()
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    return tp, tn, fp, fn


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    description: str | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    tp = tn = fp = fn = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        progress = loader
        if tqdm is not None:
            progress = tqdm(loader, desc=description, leave=False)

        for x, y in progress:
            x = x.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = y.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            batch_tp, batch_tn, batch_fp, batch_fn = binary_counts(logits.detach(), y.detach())
            tp += batch_tp
            tn += batch_tn
            fp += batch_fp
            fn += batch_fn

            if tqdm is not None:
                running_loss = total_loss / max(total_examples, 1)
                running_metrics = metrics_from_counts(tp, tn, fp, fn)
                progress.set_postfix(
                    loss=f"{running_loss:.4f}",
                    acc=f"{running_metrics['accuracy']:.3f}",
                    f1=f"{running_metrics['f1']:.3f}",
                )

    avg_loss = total_loss / max(total_examples, 1)
    return avg_loss, metrics_from_counts(tp, tn, fp, fn)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            description=f"Epoch {epoch:03d}/{epochs:03d} train",
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            description=f"Epoch {epoch:03d}/{epochs:03d} val",
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} acc={train_metrics['accuracy']:.3f} "
            f"f1={train_metrics['f1']:.3f} | "
            f"val loss={val_loss:.4f} acc={val_metrics['accuracy']:.3f} "
            f"f1={val_metrics['f1']:.3f}"
        )

    model.load_state_dict(best_state)
    return model

# valutazione e stampa metriche
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    criterion = nn.BCEWithLogitsLoss()
    loss, metrics = run_epoch(model, loader, criterion, device)
    true_short, true_long, pred_short, pred_long = collect_binary_label_counts(
        model,
        loader,
        device,
    )

    print("\nTest results")
    print(f"loss: {loss:.4f}")
    print(f"accuracy: {metrics['accuracy']:.3f}")
    print(f"precision: {metrics['precision']:.3f}")
    print(f"recall: {metrics['recall']:.3f}")
    print(f"f1: {metrics['f1']:.3f}")
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(
        [
            [int(metrics["tn"]), int(metrics["fp"])],
            [int(metrics["fn"]), int(metrics["tp"])],
        ]
    )
    print("true labels:")
    print(f"short GRBs: {true_short}")
    print(f"long GRBs: {true_long}")
    print("model predictions:")
    print(f"short GRBs: {pred_short}")
    print(f"long GRBs: {pred_long}")
    print("label rule: short = T90 <= 2 seconds, long = T90 > 2 seconds")


def collect_binary_label_counts(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[int, int, int, int]:
    model.eval()
    true_short = true_long = pred_short = pred_long = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            labels = y.long()

            true_short += int((labels == 0).sum().item())
            true_long += int((labels == 1).sum().item())
            pred_short += int((preds == 0).sum().item())
            pred_long += int((preds == 1).sum().item())

    return true_short, true_long, pred_short, pred_long
