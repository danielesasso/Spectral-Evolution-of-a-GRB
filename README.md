# Gamma-Ray-Burst

Struttura iniziale per un workspace dedicato all'analisi di Gamma-Ray Burst.

## Ambiente conda

Ambiente creato: `grb`

### Dipendenze da installare

Dipendenze principali usate nel progetto (allineate all'ambiente `grb`):
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

Installazione consigliata:

```bash
conda create -n grb python=3.11 -y
conda activate grb
conda install -c conda-forge numpy=1.24.2 pandas=2.0.0 matplotlib=3.7.1 scipy=1.10.1 scikit-learn=1.2.2 astropy=6.1.3 h5py=3.12.1 tables=3.9.2 requests=2.29.0 -y
pip install ClassiPyGRB==1.0.0
```

## Struttura

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

## Uso rapido

Attiva l'ambiente:

```bash
conda activate grb
```

Esegui lo script principale:

```bash
python3 main.py
```

## Licenza

Questo progetto e distribuito sotto licenza MIT.
Per i dettagli completi, vedi il file `LICENSE`.