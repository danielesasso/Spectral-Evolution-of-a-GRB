import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# main.py: reproducibility and device selection
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# testing_files/*.py: Swift metadata and raw light-curve extraction
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


def extract_light_curve_arrays(
    df: pd.DataFrame,
    channel_columns: list[str],
    time_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract variable-length light-curve arrays without padding, truncation, or resampling.

    This is intended for the raw HDF5 cache. Fixed-shape conversion should happen
    later, right before analysis/training.
    """
    missing_channels = [col for col in channel_columns if col not in df.columns]
    if missing_channels:
        raise ValueError(f"Missing rate columns: {missing_channels}")

    rates = df[channel_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if rates.size == 0:
        raise ValueError("Empty light curve")

    if time_column in df.columns:
        time = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=np.float64)
    else:
        time = np.arange(rates.shape[0], dtype=np.float64)

    valid_rows = np.isfinite(time)
    if valid_rows.sum() == 0:
        raise ValueError("No finite time values found")

    time = time[valid_rows]
    rates = rates[valid_rows]
    order = np.argsort(time)
    time = time[order]
    rates = rates[order]

    unique_time, unique_idx = np.unique(time, return_index=True)
    return unique_time.astype(np.float64), rates[unique_idx].astype(np.float32)


# main.py: fold creation and DataLoader setup
def dataset_labels(dataset: Dataset) -> list[int] | None:
    """Return binary labels without triggering item loading when possible."""
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return None

    labels: list[int] = []
    for sample in samples:
        label = sample[1]
        labels.append(int(float(label.item() if hasattr(label, "item") else label)))
    return labels


def make_stratified_folds(dataset: Dataset, k_folds: int, seed: int) -> list[list[int]]:
    """Create class-balanced folds for cross-validation."""
    if k_folds < 3:
        raise ValueError("Need at least 3 folds so train, validation, and test are separate")

    labels = dataset_labels(dataset)
    if labels is None or len(set(labels)) < 2:
        raise ValueError("Stratified k-fold requires binary labels on the dataset")

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    min_class_count = min(len(indices) for indices in by_class.values())
    if k_folds > min_class_count:
        raise ValueError(
            f"k_folds={k_folds} is too large for the smallest class ({min_class_count} samples)"
        )

    folds = [[] for _ in range(k_folds)]
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        for item_idx, dataset_idx in enumerate(class_indices):
            folds[item_idx % k_folds].append(dataset_idx)

    for fold in folds:
        rng.shuffle(fold)
    return folds


def make_fold_dataloaders(
    dataset: Dataset,
    folds: list[list[int]],
    fold_idx: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    global_normalize: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Use one fold for test, the next fold for validation, and the rest for training."""
    if not folds:
        raise ValueError("folds cannot be empty")

    fold_count = len(folds)
    test_fold = fold_idx % fold_count
    val_fold = (fold_idx + 1) % fold_count
    train_idx = [
        dataset_idx
        for current_fold, fold in enumerate(folds)
        if current_fold not in {test_fold, val_fold}
        for dataset_idx in fold
    ]
    rng = random.Random(seed + fold_idx)
    rng.shuffle(train_idx)

    if global_normalize:
        train_set, val_set, test_set = make_global_normalized_subsets(
            dataset,
            train_idx=train_idx,
            val_idx=folds[val_fold],
            test_idx=folds[test_fold],
        )
    else:
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, folds[val_fold])
        test_set = Subset(dataset, folds[test_fold])

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


def make_global_normalized_subsets(
    dataset: Dataset,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    eps: float = 1e-6,
) -> tuple[Dataset, Dataset, Dataset]:
    """Normalize X with mean/std computed from training GRBs only."""
    x_values = getattr(dataset, "x", None)
    if x_values is None:
        raise ValueError("Global normalization requires the dataset to expose an x tensor")

    train_x = x_values[train_idx]
    mean = train_x.mean(dim=(0, 1)).view(1, -1)
    std = train_x.std(dim=(0, 1), unbiased=False).view(1, -1)
    std = torch.where(std < eps, torch.ones_like(std), std)

    return (
        GlobalNormalizedSubset(dataset, train_idx, mean, std),
        GlobalNormalizedSubset(dataset, val_idx, mean, std),
        GlobalNormalizedSubset(dataset, test_idx, mean, std),
    )


class GlobalNormalizedSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[self.indices[idx]]
        return (x - self.mean) / self.std, y


# main.py: training and evaluation helpers
def binary_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[int, int, int, int]:
    preds = (torch.sigmoid(logits) >= threshold).long()
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
    threshold: float = 0.5,
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
            batch_tp, batch_tn, batch_fp, batch_fn = binary_counts(
                logits.detach(),
                y.detach(),
                threshold=threshold,
            )
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
    weight_decay: float = 1e-3,
    patience: int = 12,
) -> nn.Module:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, patience // 3),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    epochs_without_improvement = 0

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

        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train_loss:.4f} acc={train_metrics['accuracy']:.3f} "
                f"f1={train_metrics['f1']:.3f} | "
                f"val loss={val_loss:.4f} acc={val_metrics['accuracy']:.3f} "
                f"f1={val_metrics['f1']:.3f}"
            )

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping after {epoch} epochs; "
                f"best val loss={best_val_loss:.4f} f1={best_val_f1:.3f}"
            )
            break

    model.load_state_dict(best_state)
    return model


def find_best_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Tune the decision threshold on validation data instead of assuming 0.5."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits_list.append(model(x.to(device)).detach().cpu())
            targets_list.append(y.detach().cpu())

    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)

    best_threshold = 0.5
    best_metrics = metrics_from_counts(*binary_counts(logits, targets, threshold=best_threshold))
    for threshold in thresholds:
        metrics = metrics_from_counts(*binary_counts(logits, targets, threshold=float(threshold)))
        if (
            metrics["f1"] > best_metrics["f1"]
            or (
                metrics["f1"] == best_metrics["f1"]
                and metrics["accuracy"] > best_metrics["accuracy"]
            )
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def collect_binary_label_counts(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[int, int, int, int]:
    model.eval()
    true_short = true_long = pred_short = pred_long = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) >= threshold).long()
            labels = y.long()

            true_short += int((labels == 0).sum().item())
            true_long += int((labels == 1).sum().item())
            pred_short += int((preds == 0).sum().item())
            pred_long += int((preds == 1).sum().item())

    return true_short, true_long, pred_short, pred_long


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    criterion = nn.BCEWithLogitsLoss()
    loss, metrics = run_epoch(model, loader, criterion, device, threshold=threshold)
    true_short, true_long, pred_short, pred_long = collect_binary_label_counts(
        model,
        loader,
        device,
        threshold=threshold,
    )

    print("\nTest results")
    print(f"loss: {loss:.4f}")
    print(f"accuracy: {metrics['accuracy']:.3f}")
    print(f"precision: {metrics['precision']:.3f}")
    print(f"recall: {metrics['recall']:.3f}")
    print(f"f1: {metrics['f1']:.3f}")
    print(f"decision threshold: {threshold:.2f}")
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
    return loss, metrics
