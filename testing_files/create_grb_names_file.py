import argparse
from pathlib import Path

import pandas as pd
from ClassiPyGRB import SWIFT


# ---------------------------------------------------------------------------
# Configuration
# Edit output path and filtering options here.
# ---------------------------------------------------------------------------

OUTPUT_FILE = Path("data/raw/swift_grb_names.txt")
ONLY_WITH_T90 = True
SORT_NAMES = True
LIMIT: int | None = None
CHECK_DOWNLOADS = False
SWIFT_RESOLUTION = 64
REQUIRED_COLUMNS = ["Time(s)", "15-25keV", "25-50keV", "50-100keV", "100-350keV"]


def get_swift_summary_names(only_with_t90: bool, sort_names: bool) -> list[str]:
    summary = SWIFT.summary_table()
    if "GRBname" not in summary.columns:
        raise ValueError("ClassiPyGRB Swift summary table does not contain a GRBname column")

    if only_with_t90:
        if "T90" not in summary.columns:
            raise ValueError("ClassiPyGRB Swift summary table does not contain a T90 column")
        summary = summary.copy()
        summary["T90"] = pd.to_numeric(summary["T90"], errors="coerce")
        summary = summary[summary["T90"].notna()]

    names = summary["GRBname"].dropna().astype(str).str.strip()
    names = [name for name in names if name]
    names = list(dict.fromkeys(names))

    if sort_names:
        names = sorted(names)

    return names

def keep_downloadable_names(names: list[str], swift_resolution: int) -> tuple[list[str], list[tuple[str, str]]]:
    swift = SWIFT(res=swift_resolution)
    valid_names: list[str] = []
    skipped: list[tuple[str, str]] = []

    for index, name in enumerate(names, start=1):
        print(f"[{index}/{len(names)}] Checking {name}")
        result = swift.obtain_data(name=name)
        if not isinstance(result, pd.DataFrame):
            skipped.append((name, str(result)))
            continue

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in result.columns]
        if missing_columns:
            skipped.append((name, f"missing columns: {missing_columns}"))
            continue

        valid_names.append(name)

    return valid_names, skipped


def write_names(names: list[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(names) + "\n")


def write_skipped(skipped: list[tuple[str, str]], output_file: Path) -> None:
    if not skipped:
        return

    skipped_file = output_file.with_suffix(".skipped.txt")
    lines = [f"{name}\t{reason}" for name, reason in skipped]
    skipped_file.write_text("\n".join(lines) + "\n")
    print(f"Skipped log written to: {skipped_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a text file with GRB names from the ClassiPyGRB Swift summary table."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--include-missing-t90", action="store_true")
    parser.add_argument("--no-sort", action="store_true")
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument(
        "--check-downloads",
        action="store_true",
        help="Slow: call SWIFT.obtain_data for each GRB and keep only downloadable light curves.",
    )
    parser.add_argument("--swift-resolution", type=int, default=SWIFT_RESOLUTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    names = get_swift_summary_names(
        only_with_t90=not args.include_missing_t90,
        sort_names=not args.no_sort,
    )

    if args.limit is not None:
        names = names[: args.limit]

    skipped: list[tuple[str, str]] = []
    if args.check_downloads:
        names, skipped = keep_downloadable_names(names, swift_resolution=args.swift_resolution)

    write_names(names, args.output)
    write_skipped(skipped, args.output)

    print(f"Written {len(names)} GRB names to: {args.output}")
    print(f"Use all names: python3 main.py --names-file {args.output}")
    print(f"Use first 100: head -n 100 {args.output} > data/raw/swift_grb_names_head100.txt")


if __name__ == "__main__":
    main()
