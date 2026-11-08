# Execution logs

The project runner writes console output and `logs/<YYYYMMDD_HHMMSS>/run.log`
for each run. The matching run ID identifies history and figures under
[`plots/`](../plots/README.md). Generated `.log` files are ignored by git.

After model selection, the log includes held-out test macro F1 and a per-class
precision, recall, F1, and support report. Class labels are encoded integers;
see [`datasets/cicids2017_label_mapping.json`](../datasets/cicids2017_label_mapping.json)
for their names. Test results are for final reporting, not model selection.
The local-testing sample has altered class proportions; see [`datasets/README.md`](../datasets/README.md).

Run the project from the repository root as described in the
[project README](../README.md#execution-plots).
