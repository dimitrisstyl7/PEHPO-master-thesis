# Datasets

This directory holds dataset inputs and generated data for the notebooks and
experiments. Dataset CSV files are ignored by git.

- [`cicids2017/`](cicids2017/README.md): raw CICIDS2017 files and download
  instructions.
- `cicids2017_cleaned.csv`: cleaned features and encoded labels.
- `cicids2017_label_mapping.json`: mapping from original labels to integers.
- `cicids2017_sample.csv`: stratified sample for local testing. Its sampling
  floor increases the proportion of rare classes, so its test results may not
  represent the original dataset or deployment traffic.

See [`notebooks/README.md`](../notebooks/README.md) for preparation and
sampling instructions. Set `DATASET_PATH` in `main.py` to the desired CSV before
running an experiment.
