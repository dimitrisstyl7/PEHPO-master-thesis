# Notebooks

[⬅ Back to main README](../README.md#readme-files)

These notebooks prepare CICIDS2017 data for experiments and local testing. Run
their cells in order, with `notebooks/` as the working directory. Both notebooks
load their full input into memory and overwrite their output files when rerun.

## Preparation

1. Follow [`datasets/cicids2017/README.md`](../datasets/cicids2017/README.md)
   to download and place the raw CSV files in `datasets/cicids2017/`.
2. Run [`01_cicids_cleaning.ipynb`](01_cicids_cleaning.ipynb). It combines the
   raw files, cleans numeric features, encodes labels, and writes
   `datasets/cicids2017_cleaned.csv` and
   `datasets/cicids2017_label_mapping.json`.
3. Run [`02_cicids_sampling.ipynb`](02_cicids_sampling.ipynb). It reads both
   cleaning outputs and writes `datasets/cicids2017_sample.csv`.

The sampling notebook keeps 5% of each label by default, with a minimum of 500
rows per label or all available rows for smaller labels. The minimum increases
rare-label shares relative to the original data. Adjust `SAMPLE_FRACTION`,
`MIN_PER_CLASS`, and `SEED` in its configuration cell as needed. Its summary
compares original and sampled label distributions.

To use the sample for an experiment, set `DATASET_PATH` in `main.py` to
`datasets/cicids2017_sample.csv`.

For environment setup and notebook launch instructions, see the
[project README](../README.md#running-the-notebooks).
