import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

# Keep BLAS/OpenMP threading conservative to avoid intermittent segfaults
# on some macOS Python/XGBoost builds.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import xgboost as xgb

from classes import GRBHDF5Dataset
from functions import (
    make_stratified_folds,
    metrics_from_counts,
    set_seed,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent
    h5_file: Path = project_root / "data" / "processed" / "classipygrb" / "swift.hd5"

    k_folds: int = 5
    seed: int = 42

    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    xgb_nthread: int = 1
    early_stopping_patience: int = 20


CONFIG = Config()


# ---------------------------------------------------------------------------
# Flattening and augmentation
# ---------------------------------------------------------------------------

def flatten_grb(x: np.ndarray) -> np.ndarray:
    """Flatten a single GRB from (time, channels) to (time * channels,)."""
    return x.reshape(-1)


def apply_jitter(x_flat: np.ndarray, jitter_std: float = 0.02) -> np.ndarray:
    """Apply jitter augmentation to flattened sample."""
    return x_flat + np.random.randn(*x_flat.shape) * jitter_std


def apply_scaling(x_flat: np.ndarray, scaling_std: float = 0.1) -> np.ndarray:
    """Apply scaling augmentation to flattened sample."""
    scale = np.clip(np.random.randn() * scaling_std + 1.0, a_min=0.1, a_max=None)
    return x_flat * scale


def apply_noise(x_flat: np.ndarray, noise_std: float = 0.03) -> np.ndarray:
    """Apply noise augmentation to flattened sample."""
    return x_flat + np.random.randn(*x_flat.shape) * noise_std


def augment_training_set(
    x_train: np.ndarray,
    y_train: np.ndarray,
    jitter_ratio: float = 0.0,
    scaling_ratio: float = 0.0,
    noise_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment training data in-memory by selecting and transforming samples per class."""
    rng = np.random.RandomState(seed)

    x_augmented = [x_train]
    y_augmented = [y_train]

    by_class = {}
    for idx, label in enumerate(y_train):
        by_class.setdefault(int(label), []).append(idx)

    augmentations = [
        ("jitter", jitter_ratio, apply_jitter),
        ("scaling", scaling_ratio, apply_scaling),
        ("noise", noise_ratio, apply_noise),
    ]

    for _, aug_ratio, aug_func in augmentations:
        if aug_ratio <= 0.0:
            continue

        selected_indices = []
        for class_indices in by_class.values():
            if not class_indices:
                continue
            shuffled = class_indices[:]
            rng.shuffle(shuffled)
            class_take = int(len(shuffled) * aug_ratio)
            if aug_ratio > 0.0 and class_take == 0:
                class_take = 1
            selected_indices.extend(shuffled[:class_take])

        x_aug_samples = []
        y_aug_samples = []
        for idx in selected_indices:
            x_aug = aug_func(x_train[idx].copy())
            x_aug_samples.append(x_aug)
            y_aug_samples.append(y_train[idx])

        x_augmented.append(np.array(x_aug_samples))
        y_augmented.append(np.array(y_aug_samples))

    x_train_aug = np.vstack(x_augmented)
    y_train_aug = np.hstack(y_augmented)
    return x_train_aug, y_train_aug


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_xgb_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
) -> xgb.XGBClassifier:
    """Train XGBoost on one fold."""
    model = xgb.XGBClassifier(
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        n_estimators=config.xgb_n_estimators,
        early_stopping_rounds=config.early_stopping_patience,
        nthread=config.xgb_nthread,
        random_state=config.seed,
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=False,
    )
    print_training_progress(model.evals_result(), log_every=10)
    return model


def print_training_progress(evals_result: dict[str, dict[str, list[float]]], log_every: int = 10) -> None:
    """Print train/validation loss every N boosting rounds."""
    train_loss = evals_result.get("validation_0", {}).get("logloss", [])
    val_loss = evals_result.get("validation_1", {}).get("logloss", [])
    if not train_loss or not val_loss:
        return

    total_rounds = min(len(train_loss), len(val_loss))
    for epoch_idx in range(1, total_rounds + 1):
        if epoch_idx % log_every == 0 or epoch_idx == total_rounds:
            print(
                f"epoch {epoch_idx:03d}/{total_rounds} | "
                f"train_loss={train_loss[epoch_idx - 1]:.4f} | "
                f"val_loss={val_loss[epoch_idx - 1]:.4f}"
            )


def evaluate_xgb_fold(
    model: xgb.XGBClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Evaluate XGBoost on validation and test sets."""
    test_logits = model.predict_proba(x_test)[:, 1]

    best_threshold = threshold
    best_f1 = 0.0

    for test_threshold in np.linspace(0.05, 0.95, 91):
        preds = (test_logits >= test_threshold).astype(int)
        tp, tn, fp, fn = binary_counts_from_preds(preds, y_test)
        metrics = metrics_from_counts(tp, tn, fp, fn)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(test_threshold)

    test_preds = (test_logits >= best_threshold).astype(int)
    tp, tn, fp, fn = binary_counts_from_preds(test_preds, y_test)
    test_metrics = metrics_from_counts(tp, tn, fp, fn)
    test_metrics["threshold"] = best_threshold
    test_metrics["loss"] = float(np.mean(np.log(1.0 + np.exp(-y_test * (2 * test_logits - 1)))))

    return best_threshold, test_metrics


def binary_counts_from_preds(
    preds: np.ndarray,
    targets: np.ndarray,
) -> tuple[int, int, int, int]:
    """Compute confusion matrix from predictions and targets."""
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    return tp, tn, fp, fn


def print_cross_validation_summary(fold_results: list[dict[str, float]]) -> None:
    """Print CV summary across all folds."""
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
    parser = argparse.ArgumentParser(description="Train XGBoost on cached Swift GRB light curves.")
    parser.add_argument(
        "h5_file",
        nargs="?",
        type=Path,
        default=CONFIG.h5_file,
        help="HDF5 file containing cached GRB light curves.",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=CONFIG.k_folds,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Jitter augmentation ratio in [0, 1] (default: 0).",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=0.0,
        help="Scaling augmentation ratio in [0, 1] (default: 0).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise augmentation ratio in [0, 1] (default: 0).",
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        dest="epochs",
        type=int,
        default=CONFIG.xgb_n_estimators,
        help="Number of XGBoost estimators (default: 100).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=CONFIG.xgb_nthread,
        help="XGBoost CPU threads (default: 1, safer on macOS).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIG
    config.k_folds = args.k_folds
    config.xgb_nthread = args.threads

    if config.k_folds < 3:
        raise ValueError(f"--k-folds must be >= 3, got {config.k_folds}")

    for flag_name, flag_value in {
        "--jitter": args.jitter,
        "--scaling": args.scaling,
        "--noise": args.noise,
    }.items():
        if flag_value < 0.0 or flag_value > 1.0:
            raise ValueError(f"{flag_name} must be in [0, 1], got {flag_value}")

    config.xgb_n_estimators = args.epochs
    if config.xgb_nthread < 1:
        raise ValueError(f"--threads must be >= 1, got {config.xgb_nthread}")
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
    print(f"K-folds: {config.k_folds}")
    print(f"Jitter ratio: {args.jitter:.2f}")
    print(f"Scaling ratio: {args.scaling:.2f}")
    print(f"Noise ratio: {args.noise:.2f}")
    print(f"Epochs: {config.xgb_n_estimators}")
    print(f"Threads: {config.xgb_nthread}")
    print(f"Model: XGBoost")

    x = dataset.x.numpy()
    y = np.array(dataset.labels, dtype=int)

    x_flat = np.array([flatten_grb(x[i]) for i in range(len(x))])

    folds = make_stratified_folds(dataset, k_folds=config.k_folds, seed=config.seed)
    fold_results: list[dict[str, float]] = []

    for fold_idx in range(config.k_folds):
        print(f"\nFold {fold_idx + 1}/{config.k_folds}")
        set_seed(config.seed + fold_idx)

        test_fold = fold_idx % len(folds)
        val_fold = (fold_idx + 1) % len(folds)
        train_idx = [
            dataset_idx
            for current_fold, fold in enumerate(folds)
            if current_fold not in {test_fold, val_fold}
            for dataset_idx in fold
        ]

        x_train = x_flat[train_idx]
        y_train = y[train_idx]
        x_val = x_flat[folds[val_fold]]
        y_val = y[folds[val_fold]]
        x_test = x_flat[folds[test_fold]]
        y_test = y[folds[test_fold]]

        if args.jitter > 0.0 or args.scaling > 0.0 or args.noise > 0.0:
            x_train, y_train = augment_training_set(
                x_train,
                y_train,
                jitter_ratio=args.jitter,
                scaling_ratio=args.scaling,
                noise_ratio=args.noise,
                seed=config.seed + fold_idx,
            )

        model = train_xgb_fold(x_train, y_train, x_val, y_val, config)
        threshold, fold_metrics = evaluate_xgb_fold(model, x_test, y_test)
        fold_results.append(fold_metrics)

        print(
            f"test loss={fold_metrics['loss']:.4f} "
            f"acc={fold_metrics['accuracy']:.3f} "
            f"f1={fold_metrics['f1']:.3f} "
            f"threshold={threshold:.2f}"
        )

    print_cross_validation_summary(fold_results)


if __name__ == "__main__":
    main()
