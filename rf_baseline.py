import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from classes import GRBHDF5Dataset
from functions import (
    make_fold_dataloaders,
    make_stratified_folds,
    metrics_from_counts,
    set_seed,
)


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent
    h5_file: Path = project_root / "data" / "processed" / "classipygrb" / "swift.hd5"

    k_folds: int = 5
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 0
    global_normalize: bool = True

    rf_n_estimators: int = 30


CONFIG = Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Random Forest on cached Swift GRB light curves.")
    parser.add_argument(
        "h5_file",
        nargs="?",
        type=Path,
        default=CONFIG.h5_file,
        help="HDF5 file containing cached GRB light curves.",
    )
    parser.add_argument("--k-folds", type=int, default=CONFIG.k_folds)
    parser.add_argument(
        "--epochs",
        "--epoch",
        dest="epochs",
        type=int,
        default=CONFIG.rf_n_estimators,
        help="Number of trees for Random Forest (default: 30).",
    )
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
    return parser.parse_args()


def flatten_batch(x_batch: np.ndarray) -> np.ndarray:
    """Convert (batch, time, channels) to (batch, time * channels)."""
    return x_batch.reshape(x_batch.shape[0], -1)


def loader_to_numpy(loader) -> tuple[np.ndarray, np.ndarray]:
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_np = x_batch.detach().cpu().numpy()
        y_np = y_batch.detach().cpu().numpy().astype(int)
        x_all.append(flatten_batch(x_np))
        y_all.append(y_np)
    return np.vstack(x_all), np.hstack(y_all)


def binary_counts_from_preds(preds: np.ndarray, targets: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    return tp, tn, fp, fn


def find_best_threshold_from_probs(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold = 0.5
    best_metrics = metrics_from_counts(*binary_counts_from_preds((probs >= 0.5).astype(int), targets))

    for threshold in thresholds:
        preds = (probs >= float(threshold)).astype(int)
        metrics = metrics_from_counts(*binary_counts_from_preds(preds, targets))
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


def binary_cross_entropy_loss(probs: np.ndarray, targets: np.ndarray, eps: float = 1e-7) -> float:
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(targets * np.log(probs) + (1 - targets) * np.log(1.0 - probs)))


def evaluate_rf_fold(
    model: RandomForestClassifier,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    val_probs = model.predict_proba(x_val)[:, 1]
    threshold, val_metrics = find_best_threshold_from_probs(val_probs, y_val)
    print(
        "\nValidation threshold tuning"
        f"\nbest threshold: {threshold:.2f}"
        f"\nval accuracy: {val_metrics['accuracy']:.3f}"
        f"\nval f1: {val_metrics['f1']:.3f}"
    )

    test_probs = model.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    tp, tn, fp, fn = binary_counts_from_preds(test_preds, y_test)
    test_metrics = metrics_from_counts(tp, tn, fp, fn)
    test_loss = binary_cross_entropy_loss(test_probs, y_test)

    true_short = int((y_test == 0).sum())
    true_long = int((y_test == 1).sum())
    pred_short = int((test_preds == 0).sum())
    pred_long = int((test_preds == 1).sum())

    print("\nTest results")
    print(f"loss: {test_loss:.4f}")
    print(f"accuracy: {test_metrics['accuracy']:.3f}")
    print(f"precision: {test_metrics['precision']:.3f}")
    print(f"recall: {test_metrics['recall']:.3f}")
    print(f"specificity: {test_metrics['specificity']:.3f}")
    print(f"balanced accuracy: {test_metrics['balanced_accuracy']:.3f}")
    print(f"f1: {test_metrics['f1']:.3f}")
    print(f"mcc: {test_metrics['mcc']:.3f}")
    print(f"decision threshold: {threshold:.2f}")
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print([[int(test_metrics["tn"]), int(test_metrics["fp"])], [int(test_metrics["fn"]), int(test_metrics["tp"])]])
    print("true labels:")
    print(f"short GRBs: {true_short}")
    print(f"long GRBs: {true_long}")
    print("model predictions:")
    print(f"short GRBs: {pred_short}")
    print(f"long GRBs: {pred_long}")
    print("label rule: short = T90 <= 2 seconds, long = T90 > 2 seconds")

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


def main() -> None:
    args = parse_args()
    config = CONFIG
    config.k_folds = args.k_folds
    config.rf_n_estimators = args.epochs
    config.global_normalize = not args.no_global_normalize

    if config.k_folds < 3:
        raise ValueError(f"--k-folds must be >= 3, got {config.k_folds}")
    if config.rf_n_estimators < 1:
        raise ValueError(f"--epochs must be >= 1, got {config.rf_n_estimators}")

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
        if h5_path in {Path("swift.hd5"), Path("swift.h5")}:
            h5_path = config.project_root / "data" / "processed" / "classipygrb" / h5_path.name
        else:
            h5_path = config.project_root / h5_path

    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {h5_path}\n"
            "Tip: use data/processed/classipygrb/swift.hd5 from the repository root."
        )

    dataset = GRBHDF5Dataset(h5_path)
    short_count = sum(1 for label in dataset.labels if label == 0)
    long_count = sum(1 for label in dataset.labels if label == 1)

    print(f"HDF5 file: {h5_path}")
    print(f"Loaded GRBs: {len(dataset)}")
    print(f"Short GRBs: {short_count}")
    print(f"Long GRBs: {long_count}")
    print(f"Input shape: {tuple(dataset.x.shape)}")
    print(f"Channels: {', '.join(dataset.channel_columns)}")
    print(f"Label rule: {dataset.label_rule}")
    print(f"Jitter ratio: {args.jitter:.2f}")
    print(f"Scaling ratio: {args.scaling:.2f}")
    print(f"Noise ratio: {args.noise:.2f}")
    print(f"Epochs: {config.rf_n_estimators}")
    print("Model: Random Forest")

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
            jitter_ratio=args.jitter,
            scaling_ratio=args.scaling,
            noise_ratio=args.noise,
        )

        x_train, y_train = loader_to_numpy(train_loader)
        x_val, y_val = loader_to_numpy(val_loader)
        x_test, y_test = loader_to_numpy(test_loader)

        model = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            random_state=config.seed + fold_idx,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        fold_results.append(evaluate_rf_fold(model, x_val, y_val, x_test, y_test))

    print_cross_validation_summary(fold_results)


if __name__ == "__main__":
    main()
