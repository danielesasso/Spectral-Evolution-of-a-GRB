# Gamma-Ray-Burst

Initial structure for a workspace dedicated to the analysis of Gamma-Ray Bursts.

## Conda Environment

Conda environment name: `grb`

### Dependencies to Install

Main dependencies used in the project (aligned with the `grb` environment):  

- `python=3.11`
- `numpy==1.24.2`
- `pandas==2.0.0`
- `matplotlib==3.7.1`
- `scipy==1.10.1`
- `scikit-learn==1.2.2`
- `astropy==6.1.3`
- `h5py==3.12.1`
- `tables==3.9.2`
- `requests==2.29.0`
- `ClassiPyGRB==1.0.0`

Recommended installation:

```bash
conda create -n grb python=3.11 -y
conda activate grb
conda install -c conda-forge numpy=1.24.2 pandas=2.0.0 matplotlib=3.7.1 scipy=1.10.1 scikit-learn=1.2.2 astropy=6.1.3 h5py=3.12.1 tables=3.9.2 requests=2.29.0 -y
pip install ClassiPyGRB==1.0.0
```

## Project Structure

```text
Gamma-Ray-Burst/
├── data/
│   ├── raw/
│   │   ├── classipygrb/
│   │   └── fermi_gbm/
│   └── processed/
│       ├── classipygrb/
│       └── fermi_gbm/
├── experiments/
├── main.py
└── README.md
```

## A quick start:

Activate the environment:

```bash
conda activate grb
```

Run the main script with:

```bash
python3 main.py
```

## License

This project is distributed under the MIT License.  
For full details, see the `LICENSE` file.
