import argparse
import random
from pathlib import Path

import pandas as pd
from ClassiPyGRB import SWIFT


# ---------------------------------------------------------------------------
# Configuration
# Edit paths and the short/long threshold here.
# ---------------------------------------------------------------------------

NAMES_FILE = Path("data/raw/swift_grb_names.txt")
OUTPUT_DIR = Path("data/processed/classipygrb")
T90_THRESHOLD_SECONDS = 2.0
RANDOM_SEED = 42


def load_names(names_file: Path) -> list[str]:
    if not names_file.exists():
        raise FileNotFoundError(f"Names file not found: {names_file}")

    names = []
    seen = set()
    for line in names_file.read_text().splitlines():
        name = line.strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def load_t90_table() -> pd.DataFrame:
    table = SWIFT.summary_table()
    required_columns = {"GRBname", "T90"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(f"Swift summary table is missing columns: {sorted(missing_columns)}")

    table = table[["GRBname", "T90"]].copy()
    table["GRBname"] = table["GRBname"].astype(str).str.strip()
    table["T90"] = pd.to_numeric(table["T90"], errors="coerce")
    table = table.dropna(subset=["GRBname", "T90"])
    return table.drop_duplicates(subset=["GRBname"], keep="first")


def summarize_classes(
    names: list[str],
    t90_table: pd.DataFrame,
    threshold_seconds: float,
) -> tuple[list[str], list[str]]:
    selected = pd.DataFrame({"GRBname": names})
    selected = selected.merge(t90_table, on="GRBname", how="left")

    missing_t90 = selected["T90"].isna()
    valid = selected[~missing_t90].copy()
    short = valid[valid["T90"] <= threshold_seconds]
    long = valid[valid["T90"] > threshold_seconds]

    print(f"Names file GRBs: {len(names)}")
    print(f"GRBs with T90: {len(valid)}")
    print(f"GRBs missing T90: {int(missing_t90.sum())}")
    print(f"Short GRBs (T90 <= {threshold_seconds:g} s): {len(short)}")
    print(f"Long GRBs  (T90 >  {threshold_seconds:g} s): {len(long)}")

    if len(valid):
        short_fraction = len(short) / len(valid)
        long_fraction = len(long) / len(valid)
        print(f"Short fraction: {short_fraction:.3f}")
        print(f"Long fraction: {long_fraction:.3f}")

    if len(short) and len(long):
        majority = max(len(short), len(long))
        minority = min(len(short), len(long))
        print(f"Class imbalance ratio: {majority / minority:.2f}:1")

    return short["GRBname"].tolist(), long["GRBname"].tolist()


def write_names(names: list[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(names) + "\n")


def write_balanced_files(
    short_names: list[str],
    long_names: list[str],
    output_dir: Path,
    seed: int,
) -> None:
    rng = random.Random(seed)
    sampled_long_names = long_names.copy()
    rng.shuffle(sampled_long_names)
    sampled_long_names = sampled_long_names[: len(short_names)]

    balanced_names = short_names + sampled_long_names
    rng.shuffle(balanced_names)

    write_names(short_names, output_dir / "swift_short_grb_names.txt")
    write_names(sampled_long_names, output_dir / "swift_long_grb_names_downsampled.txt")
    write_names(balanced_names, output_dir / "swift_balanced_grb_names.txt")

    print(f"Short GRB file: {output_dir / 'swift_short_grb_names.txt'}")
    print(f"Downsampled long GRB file: {output_dir / 'swift_long_grb_names_downsampled.txt'}")
    print(f"Balanced training names file: {output_dir / 'swift_balanced_grb_names.txt'}")
    print(f"Balanced file total: {len(balanced_names)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count short and long GRBs from a names file using ClassiPyGRB Swift T90 values."
    )
    parser.add_argument("head", nargs="?", default="all", help="Number of names to read, or 'all'.")
    parser.add_argument("--names-file", type=Path, default=NAMES_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=T90_THRESHOLD_SECONDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def select_head(names: list[str], head: str) -> list[str]:
    if head.lower() == "all":
        return names

    try:
        count = int(head)
    except ValueError as exc:
        raise ValueError("head must be a positive integer or 'all'") from exc

    if count <= 0:
        raise ValueError("head must be greater than zero")

    return names[:count]


def main() -> None:
    args = parse_args()
    names = select_head(load_names(args.names_file), args.head)
    t90_table = load_t90_table()
    short_names, long_names = summarize_classes(names, t90_table, args.threshold)
    write_balanced_files(short_names, long_names, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
