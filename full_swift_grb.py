import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from ClassiPyGRB import SWIFT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions import extract_light_curve_arrays, extract_t90, make_t90_lookup


DEFAULT_CHANNEL_COLUMNS = ["15-25keV", "25-50keV", "50-100keV", "100-350keV"]
THREAD_LOCAL = threading.local()


def summary_names_with_t90(summary: pd.DataFrame) -> list[str]:
    if "GRBname" not in summary.columns or "T90" not in summary.columns:
        raise ValueError("Swift summary table must contain GRBname and T90 columns")

    table = summary.copy()
    table["GRBname"] = table["GRBname"].astype(str).str.strip()
    table["T90"] = pd.to_numeric(table["T90"], errors="coerce")
    table = table[table["GRBname"].ne("") & table["T90"].notna()]
    return table["GRBname"].drop_duplicates().sort_values().tolist()


def get_thread_swift(swift_resolution: int) -> SWIFT:
    swift = getattr(THREAD_LOCAL, "swift", None)
    if swift is None or getattr(THREAD_LOCAL, "swift_resolution", None) != swift_resolution:
        swift = SWIFT(res=swift_resolution)
        THREAD_LOCAL.swift = swift
        THREAD_LOCAL.swift_resolution = swift_resolution
    return swift


def obtain_data_with_retries(
    swift: SWIFT,
    name: str,
    retries: int,
    retry_sleep_seconds: float,
) -> pd.DataFrame:
    attempts = max(1, retries)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            df = swift.obtain_data(name=name)
            if isinstance(df, pd.DataFrame):
                return df
            raise RuntimeError(str(df))
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            permanent = "404 client error" in message or "not found" in message
            if permanent or attempt == attempts:
                break

            print(f"  attempt {attempt}/{attempts} failed for {name}: {exc}")
            print(f"  retrying in {retry_sleep_seconds:g} seconds")
            time.sleep(retry_sleep_seconds)

    raise RuntimeError(str(last_error))


def trim_light_curve_by_t90_window(
    time_values: np.ndarray,
    rates: np.ndarray,
    t90: float,
    time_column: str,
    channel_columns: list[str],
) -> np.ndarray:
    light_curve = pd.DataFrame(rates, columns=channel_columns)
    light_curve.insert(0, time_column, time_values)

    signal_strength = light_curve[channel_columns].abs().sum(axis=1)
    peak_idx = int(signal_strength.idxmax())
    reference_time = float(light_curve.loc[peak_idx, time_column])

    t5 = reference_time - 0.5 * t90
    t95 = reference_time + 0.5 * t90
    window_start = t5 - 0.5 * t90
    window_end = t95 + 0.5 * t90

    trimmed = light_curve.loc[
        light_curve[time_column].between(window_start, window_end, inclusive="both")
    ]
    if trimmed.empty:
        return rates

    return trimmed[channel_columns].to_numpy(dtype=np.float32)


def download_swift_grbs(
    swift_resolution: int,
    time_column: str,
    channel_columns: list[str],
    t90_threshold: float,
    retries: int,
    retry_sleep_seconds: float,
    download_workers: int,
    trim_to_t90_window: bool,
    limit: int | None,
) -> tuple[list[str], list[np.ndarray], np.ndarray, np.ndarray, list[tuple[str, str]], int]:
    swift = SWIFT(res=swift_resolution)
    summary = swift.summary_table()
    t90_lookup = make_t90_lookup(summary)
    names = summary_names_with_t90(summary)
    requested_count = len(names)
    if limit is not None:
        names = names[:limit]

    if download_workers < 1:
        raise ValueError(f"--download-workers must be >= 1, got {download_workers}")

    def download_one(item: tuple[int, str]):
        index, name = item
        try:
            worker_swift = get_thread_swift(swift_resolution)
            df = obtain_data_with_retries(
                worker_swift,
                name=name,
                retries=retries,
                retry_sleep_seconds=retry_sleep_seconds,
            )
            t90 = extract_t90(df, grb_name=name, t90_lookup=t90_lookup)
            time_values, rates = extract_light_curve_arrays(
                df,
                channel_columns=channel_columns,
                time_column=time_column,
            )
            if trim_to_t90_window:
                rates = trim_light_curve_by_t90_window(
                    time_values=time_values,
                    rates=rates,
                    t90=t90,
                    time_column=time_column,
                    channel_columns=channel_columns,
                )
            label = 1 if t90 > t90_threshold else 0
            return index, name, rates.astype(np.float32), label, float(t90), None
        except Exception as exc:
            return index, name, None, None, None, str(exc)

    results: list[tuple[int, str, np.ndarray, int, float]] = []
    skipped: list[tuple[str, str]] = []
    indexed_names = list(enumerate(names, start=1))

    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = [executor.submit(download_one, item) for item in indexed_names]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, name, rates, label, t90, error = future.result()
            if error is None and rates is not None and label is not None and t90 is not None:
                results.append((index, name, rates, label, t90))
                print(f"[{completed}/{len(indexed_names)}] downloaded {name}")
            else:
                skipped.append((name, str(error)))
                print(f"[{completed}/{len(indexed_names)}] skipped {name}: {error}")

    results.sort(key=lambda item: item[0])
    kept_names = [name for _, name, _, _, _ in results]
    arrays = [rates for _, _, rates, _, _ in results]
    labels = np.asarray([label for _, _, _, label, _ in results], dtype=np.float32)
    t90_values = np.asarray([t90 for _, _, _, _, t90 in results], dtype=np.float32)
    return kept_names, arrays, labels, t90_values, skipped, requested_count


def pad_arrays(arrays: list[np.ndarray], target_length: int | None) -> np.ndarray:
    if not arrays:
        raise RuntimeError("No valid GRBs were downloaded")

    channels = arrays[0].shape[1]
    fixed_length = target_length if target_length is not None else max(array.shape[0] for array in arrays)
    if fixed_length <= 0:
        raise ValueError("--target-length must be > 0")

    x = np.zeros((len(arrays), fixed_length, channels), dtype=np.float32)
    for idx, rates in enumerate(arrays):
        if rates.shape[1] != channels:
            raise ValueError("All GRBs must have the same channel count")
        copy_len = min(rates.shape[0], fixed_length)
        x[idx, :copy_len, :] = rates[:copy_len, :]
    return x


def save_h5(
    output_file: Path,
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    t90: np.ndarray,
    channel_columns: list[str],
    skipped: list[tuple[str, str]],
    requested_count: int,
    swift_resolution: int,
    t90_threshold: float,
    trim_to_t90_window: bool,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_file, "w") as h5:
        h5.create_dataset("X", data=x, compression="gzip", compression_opts=4)
        h5.create_dataset("y", data=y)
        h5.create_dataset("names", data=np.asarray(names, dtype=object), dtype=string_dtype)
        h5.create_dataset("t90", data=t90)
        h5.create_dataset(
            "channel_columns",
            data=np.asarray(channel_columns, dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "skipped_names",
            data=np.asarray([name for name, _ in skipped], dtype=object),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "skipped_reasons",
            data=np.asarray([reason for _, reason in skipped], dtype=object),
            dtype=string_dtype,
        )
        h5.attrs["label_rule"] = "0 short, 1 long"
        h5.attrs["short_rule"] = f"T90 <= {t90_threshold:g} seconds"
        h5.attrs["long_rule"] = f"T90 > {t90_threshold:g} seconds"
        h5.attrs["source"] = "ClassiPyGRB SWIFT"
        h5.attrs["swift_resolution"] = swift_resolution
        h5.attrs["requested_grbs"] = requested_count
        h5.attrs["downloaded_grbs"] = len(names)
        h5.attrs["skipped_grbs"] = len(skipped)
        h5.attrs["trim_to_t90_window"] = bool(trim_to_t90_window)
        h5.attrs["target_length"] = int(x.shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all Swift GRBs and save full_grb.h5.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "classipygrb" / "full_grb.h5",
    )
    parser.add_argument("--swift-resolution", type=int, default=64)
    parser.add_argument("--time-column", default="Time(s)")
    parser.add_argument("--channel-columns", nargs="+", default=DEFAULT_CHANNEL_COLUMNS)
    parser.add_argument("--t90-threshold", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Debug only: save first N summary GRBs.")
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = args.output
    if not output_file.is_absolute():
        output_file = PROJECT_ROOT / output_file
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}. Use --overwrite to replace it.")

    names, arrays, labels, t90, skipped, requested_count = download_swift_grbs(
        swift_resolution=args.swift_resolution,
        time_column=args.time_column,
        channel_columns=args.channel_columns,
        t90_threshold=args.t90_threshold,
        retries=args.retries,
        retry_sleep_seconds=args.retry_sleep,
        download_workers=args.download_workers,
        trim_to_t90_window=not args.no_trim,
        limit=args.limit,
    )
    x = pad_arrays(arrays, target_length=args.target_length)

    save_h5(
        output_file=output_file,
        x=x,
        y=labels,
        names=names,
        t90=t90,
        channel_columns=args.channel_columns,
        skipped=skipped,
        requested_count=requested_count,
        swift_resolution=args.swift_resolution,
        t90_threshold=args.t90_threshold,
        trim_to_t90_window=not args.no_trim,
    )

    print("\nSaved full Swift GRB dataset")
    print(f"output: {output_file}")
    print(f"requested GRBs: {requested_count}")
    print(f"downloaded GRBs: {len(names)}")
    print(f"skipped GRBs: {len(skipped)}")
    print(f"short GRBs: {int((labels == 0).sum())}")
    print(f"long GRBs: {int((labels == 1).sum())}")
    print(f"X shape: {x.shape}")
    if skipped:
        print("\nSkipped download summary")
        for skipped_name, reason in skipped[:20]:
            print(f"{skipped_name}: {reason}")
        if len(skipped) > 20:
            print(f"... {len(skipped) - 20} more skipped")


if __name__ == "__main__":
    main()
