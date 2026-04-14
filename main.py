import argparse
from dataclasses import dataclass, field
from pathlib import Path

from classes import GRBConvNet, GRBDataset
from functions import (
    evaluate_model,
    get_device,
    load_grb_names,
    make_dataloaders,
    set_seed,
    train_model,
)


# ---------------------------------------------------------------------------
# Configuration
# Edit paths, data choices, model hyperparameters, and training variables here.
# ---------------------------------------------------------------------------

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent
    raw_data_dir: Path = project_root / "data" / "raw"
    swift_names_file: Path = raw_data_dir / "swift_grb_names.txt"

    time_column: str = "Time(s)"
    channel_columns: list[str] = field(
        default_factory=lambda: ["15-25keV", "25-50keV", "50-100keV", "100-350keV"]
    )

    target_length: int = 256
    swift_resolution: int = 64
    normalize: str = "zscore"
    skip_bad_grbs: bool = True

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0

    hidden_channels: int = 64
    dropout: float = 0.2


CONFIG = Config()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(config: Config):
    return GRBConvNet(
        channels=len(config.channel_columns),
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
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(model, test_loader, device) -> None:
    evaluate_model(model, test_loader, device=device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a 1D CNN on Swift GRB light curves.")
    parser.add_argument(
        "names_file",
        nargs="?",
        type=Path,
        default=CONFIG.swift_names_file,
        help="Text file containing one GRB name per line.",
    )
    parser.add_argument(
        "grb_head",
        nargs="?",
        default="all",
        help="Number of GRBs to use from the top of the names file, or 'all'.",
    )
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    return parser.parse_args()


def select_grb_names(names: list[str], grb_head: str) -> list[str]:
    if grb_head.lower() == "all":
        return names

    try:
        head = int(grb_head)
    except ValueError as exc:
        raise ValueError("GRB head must be a positive integer or 'all'") from exc

    if head <= 0:
        raise ValueError("GRB head must be greater than zero")

    return names[:head]


def main() -> None:
    args = parse_args()
    config = CONFIG
    config.epochs = args.epochs
    set_seed(config.seed)

    all_names = load_grb_names(None, args.names_file, [])
    names = select_grb_names(all_names, args.grb_head)
    print(f"Names file: {args.names_file}")
    print(f"Requested GRBs: {len(names)}")

    dataset = GRBDataset(
        names,
        channel_columns=config.channel_columns,
        time_column=config.time_column,
        target_length=config.target_length,
        swift_resolution=config.swift_resolution,
        normalize=config.normalize,
        skip_bad=config.skip_bad_grbs,
    )
    print(f"Loaded valid GRBs: {len(dataset)}")
    if dataset.skipped:
        print(f"Skipped GRBs: {len(dataset.skipped)}")
        for name, reason in dataset.skipped[:10]:
            print(f"  {name}: {reason}")
        if len(dataset.skipped) > 10:
            print("  ...")

    train_loader, val_loader, test_loader = make_dataloaders(
        dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    device = get_device()
    print(f"Device: {device}")
    model = build_model(config).to(device)
    model = run_training(config, model, train_loader, val_loader, device)
    run_evaluation(model, test_loader, device)


if __name__ == "__main__":
    main()
