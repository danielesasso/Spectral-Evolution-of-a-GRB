import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import torch

from classes import GRBConvNet, GRBHDF5Dataset
from functions import (
    evaluate_model,
    find_best_threshold,
    get_device,
    make_fold_dataloaders,
    make_stratified_folds,
    remove_trailing_zero_padding,
    set_seed,
    summarize_sequence_lengths,
    train_model,
)


# ---------------------------------------------------------------------------
# Configuration
# Edit paths, data choices, model hyperparameters, and training variables here.
# ---------------------------------------------------------------------------

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent  # Repository root used to build data paths.
    raw_data_dir: Path = project_root / "data" / "raw"  # Folder containing raw/local cached data.
    h5_file: Path = project_root / "data" / "processed" / "classipygrb" / "swift.hd5"  # Default training dataset.

    k_folds: int = 5  # Cross-validation folds; each GRB is tested once.
    global_normalize: bool = True  # Normalize X in memory using training-fold mean/std only.
    batch_size: int = 32  # Number of GRBs processed per optimizer step.
    epochs: int = 30  # Maximum number of full passes over the training set.
    learning_rate: float = 1e-3  # Optimizer step size; lower is slower but often steadier.
    weight_decay: float = 1e-3  # L2 regularization strength to reduce overfitting.
    early_stopping_patience: int = 20  # Stop after this many epochs without validation improvement.
    seed: int = 42  # Random seed for reproducible splits and initialization.
    num_workers: int = 0  # DataLoader worker processes; 

    hidden_channels: int = 32  # CNN width; larger can learn more but overfits more easily.
    dropout: float = 0.2  # Fraction of classifier activations dropped during training.
    no_padding: bool = False  # Remove fixed HDF5 padding and use variable-length samples.


CONFIG = Config()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(config: Config, channels: int):
    return GRBConvNet(
        channels=channels,
        hidden=config.hidden_channels,
        dropout=config.dropout,
        normalization="group" if config.no_padding else "batch",
        ceil_pooling=config.no_padding,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(config: Config, model, train_loader, val_loader, device):
    return train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        lr=config.learning_rate,
        device=device,
        weight_decay=config.weight_decay,
        patience=config.early_stopping_patience,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(model, val_loader, test_loader, device) -> dict[str, float]:
    threshold, val_metrics = find_best_threshold(model, val_loader, device=device)
    print(
        "\nValidation threshold tuning"
        f"\nbest threshold: {threshold:.2f}"
        f"\nval accuracy: {val_metrics['accuracy']:.3f}"
        f"\nval f1: {val_metrics['f1']:.3f}"
    )
    test_loss, test_metrics = evaluate_model(model, test_loader, device=device, threshold=threshold)
    return {"loss": test_loss, "threshold": threshold, **test_metrics}


def print_cross_validation_summary(fold_results: list[dict[str, float]]) -> None:
    print("\nCross-validation summary")
    for metric in [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "f1",
        "mcc",
        "threshold",
    ]:
        values = [result[metric] for result in fold_results]
        metric_mean = mean(values)
        metric_std = stdev(values) if len(values) > 1 else 0.0
        print(f"{metric}: {metric_mean:.2f} +/- {metric_std:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 1D CNN on cached Swift GRB light curves.")
    parser.add_argument(
        "h5_file",
        nargs="?",
        type=Path,
        default=CONFIG.h5_file,
        help="HDF5 file containing cached GRB light curves.",
    )
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--k-folds", type=int, default=CONFIG.k_folds)
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Jitter augmentation ratio in [0, 1], applied in-memory to the training split only.",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=0.0,
        help="Scaling augmentation ratio in [0, 1], applied in-memory to the training split only.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise augmentation ratio in [0, 1], applied in-memory to the training split only.",
    )
    parser.add_argument(
        "--no-global-normalize",
        action="store_true",
        help="Use X values exactly as stored in the HDF5 file.",
    )
    parser.add_argument(
        "--nopadding",
        "--nopaddiong",
        dest="no_padding",
        action="store_true",
        help=(
            "Remove trailing all-zero padding from each HDF5 sample. "
            "This uses batch size 1 so batches do not reintroduce padding."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIG
    config.epochs = args.epochs
    config.k_folds = args.k_folds
    config.global_normalize = not args.no_global_normalize
    config.no_padding = args.no_padding
    for flag_name, flag_value in {
        "--jitter": args.jitter,
        "--scaling": args.scaling,
        "--noise": args.noise,
    }.items():
        if flag_value < 0.0 or flag_value > 1.0:
            raise ValueError(f"{flag_name} must be in [0, 1], got {flag_value}")
    set_seed(config.seed)

    h5_path = args.h5_file
    if not h5_path.is_absolute() and not h5_path.exists():
        if h5_path == Path("swift.hd5"):
            h5_path = config.project_root / "data" / "processed" / "classipygrb" / h5_path.name
        else:
            h5_path = config.project_root / h5_path

    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {h5_path}\n"
            "Tip: use data/processed/classipygrb/swift.hd5 from the repository root."
        )

    dataset = GRBHDF5Dataset(h5_path)
    if args.no_padding:
        dataset = remove_trailing_zero_padding(dataset)

    short_count = sum(1 for label in dataset.labels if label == 0)
    long_count = sum(1 for label in dataset.labels if label == 1)
    length_summary = summarize_sequence_lengths(dataset)
    effective_batch_size = 1 if args.no_padding else config.batch_size

    print(f"HDF5 file: {h5_path}")
    print(f"Loaded GRBs: {len(dataset)}")
    print(f"Short GRBs: {short_count}")
    print(f"Long GRBs: {long_count}")
    if hasattr(dataset, "x"):
        print(f"Input shape: {tuple(dataset.x.shape)}")
    else:
        print("Input shape: variable length, no trailing zero padding")
    print(
        "Sequence length: "
        f"min={length_summary['min']:.0f}, "
        f"median={length_summary['median']:.0f}, "
        f"max={length_summary['max']:.0f}"
    )
    print(f"Channels: {', '.join(dataset.channel_columns)}")
    print(f"Label rule: {dataset.label_rule}")
    print(f"No padding mode: {args.no_padding}")
    print(f"Batch size: {effective_batch_size}")
    print(f"Jitter ratio: {args.jitter:.2f}")
    print(f"Scaling ratio: {args.scaling:.2f}")
    print(f"Noise ratio: {args.noise:.2f}")

    device = torch.device("cpu") if args.no_padding else get_device()
    print(f"Device: {device}")

    folds = make_stratified_folds(dataset, k_folds=config.k_folds, seed=config.seed)
    fold_results: list[dict[str, float]] = []
    for fold_idx in range(config.k_folds):
        print(f"\nFold {fold_idx + 1}/{config.k_folds}")
        set_seed(config.seed + fold_idx)
        train_loader, val_loader, test_loader = make_fold_dataloaders(
            dataset,
            folds=folds,
            fold_idx=fold_idx,
            batch_size=effective_batch_size,
            seed=config.seed,
            num_workers=config.num_workers,
            global_normalize=config.global_normalize,
            jitter_ratio=args.jitter,
            scaling_ratio=args.scaling,
            noise_ratio=args.noise,
        )
        model = build_model(config, channels=dataset.num_channels).to(device)
        model = run_training(config, model, train_loader, val_loader, device)
        fold_results.append(run_evaluation(model, val_loader, test_loader, device))

    print_cross_validation_summary(fold_results)


if __name__ == "__main__":
    main()
