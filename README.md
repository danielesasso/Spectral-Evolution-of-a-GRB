# Spectral Evolution of a GRB

Pipeline for building Swift/BAT GRB datasets (ClassiPyGRB), converting them to training-ready HDF5 tensors, and training a CNN classifier for short vs long bursts.

## Environment

Conda environment name: `grb`

Recommended setup:

```bash
conda create -n grb python=3.10 -y
conda activate grb
conda install -c conda-forge numpy=1.24.2 pandas=2.0.0 scipy=1.10.1 scikit-learn=1.2.2 h5py=3.12.1 matplotlib=3.7.1 requests=2.29.0 -y
pip install ClassiPyGRB==1.0.0 torch torchvision torchaudio tqdm
```

## Data Pipeline

The workflow is split into three steps:

1. Build GRB name list
2. Build raw variable-length HDF5
3. Build processed fixed-shape HDF5 with `X`

### 1) Build names file

```bash
python testing_files/create_grb_names_file.py
```

Optional download validation:

```bash
python testing_files/create_grb_names_file.py --check-downloads
```

### 2) Build raw HDF5 cache

```bash
python testing_files/create_classipygrb_hdf5.py --overwrite
```

Output:

- `data/raw/classipygrb/swift_balanced_lightcurves.h5`

This raw file stores per-GRB variable-length datasets under:

- `lightcurves/<grb_name>/time`
- `lightcurves/<grb_name>/rates`

### 3) Build processed training HDF5

```bash
python testing_files/create_swift_processed_h5.py --overwrite
```

Output:

- `data/processed/classipygrb/swift.hd5`

This file contains fixed-shape tensors for training:

- `X` with shape `(n_grbs, time, channels)`
- `y`, `t90`, `names`, `channel_columns`

To cap sequence length (recommended if max length is too large):

```bash
python testing_files/create_swift_processed_h5.py --target-length 2048 --truncate --overwrite
```

## Training

Train with processed HDF5:

```bash
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120
```

Default training uses stratified k-fold cross-validation and prints per-fold + final summary metrics.

## Data Augmentation (CLI)

On-the-fly augmentations are supported during training (without modifying HDF5 files on disk).

Available augmentation flags:

- `--jitter`
- `--noise`
- `--scaling`

Examples:

```bash
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --noise --scaling
python3 main.py data/processed/classipygrb/swift.hd5 --epochs 120 --jitter --noise --scaling
```

Notes:

- Augmentations are applied only to training batches.
- Validation/test data remain untouched for fair benchmarking.
- You can combine multiple augmentation flags in the same run.

## Useful Diagnostics

Check longest-GRB row quality (all-zero vs non-zero rows):

```bash
python testing_files/print_longest_grb_row_quality.py
```

Notebook analysis:

- `preprocessing.ipynb`

## License

This project is distributed under the MIT License. See `LICENSE`.
