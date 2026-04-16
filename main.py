import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from classes import GRBConvNet, GRBHDF5Dataset
from functions import (
    evaluate_model,
    find_best_threshold,
    get_device,
    make_fold_dataloaders,
    make_stratified_folds,
    set_seed,
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
    h5_file: Path = raw_data_dir / "classipygrb" / "swift_balanced_lightcurves.h5"  # Default cached GRB dataset.

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


CONFIG = Config()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(config: Config, channels: int):
    return GRBConvNet(
        channels=channels,
        hidden=config.hidden_channels,
        dropout=config.dropout,
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
    for metric in ["loss", "accuracy", "precision", "recall", "f1", "threshold"]:
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
        "--no-global-normalize",
        action="store_true",
        help="Use X values exactly as stored in the HDF5 file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIG
    config.epochs = args.epochs
    config.k_folds = args.k_folds
    config.global_normalize = not args.no_global_normalize
    set_seed(config.seed)

    dataset = GRBHDF5Dataset(args.h5_file)
    short_count = sum(1 for label in dataset.labels if label == 0)
    long_count = sum(1 for label in dataset.labels if label == 1)

    print(f"HDF5 file: {args.h5_file}")
    print(f"Loaded GRBs: {len(dataset)}")
    print(f"Short GRBs: {short_count}")
    print(f"Long GRBs: {long_count}")
    print(f"Input shape: {tuple(dataset.x.shape)}")
    print(f"Channels: {', '.join(dataset.channel_columns)}")
    print(f"Label rule: {dataset.label_rule}")

    device = get_device()
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
            batch_size=config.batch_size,
            seed=config.seed,
            num_workers=config.num_workers,
            global_normalize=config.global_normalize,
        )
        model = build_model(config, channels=dataset.num_channels).to(device)
        model = run_training(config, model, train_loader, val_loader, device)
        fold_results.append(run_evaluation(model, val_loader, test_loader, device))

    print_cross_validation_summary(fold_results)


if __name__ == "__main__":
    main()
