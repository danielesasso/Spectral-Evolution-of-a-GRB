# Spectral Evolution of a GRB

<p align="center">
  <img src="images/Traveler%20Visits%20GRB%20Short.gif" width="500">
</p>

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

Input files are already provided in this repository and are used as the starting point for preprocessing.
In particular, the raw variable-length cache and associated metadata are treated as fixed input, not regenerated during this workflow.

### Build processed training HDF5

```bash
python testing_files/create_swift_processed_h5.py --overwrite
```

Output:

- `data/processed/classipygrb/swift.hd5`

This file contains fixed-shape tensors for training:

- `X` with shape `(n_grbs, time, channels)`
- `y`, `t90`, `names`, `channel_columns`

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
- Augmentation ratios are customizable (for example `0.1`, `0.2`, `0.4`) and can be set per enabled augmentation.

## Useful Diagnostics

Check longest-GRB row quality (all-zero vs non-zero rows):

```bash
python testing_files/print_longest_grb_row_quality.py
```

Notebook analysis:

- `preprocessing.ipynb`

## License

This project is distributed under the MIT License. See `LICENSE`.
