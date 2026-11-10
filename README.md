# Parallel Evolutionary Algorithm for Hyperparameter Optimization in Artificial Intelligence Models

## Python version

Python 3.13.x

## Clone repository

```bash
git clone https://github.com/dimitrisstyl7/PEHPO-master-thesis.git
```

## Virtual environment creation & activation ([uv](https://github.com/astral-sh/uv))

- Go to project directory

```bash
cd PEHPO-master-thesis
```

- Install `uv` (if not already installed)
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

- Create and activate the Python 3.13 virtual environment (`.venv`)
    ```bash
    uv venv --python 3.13 .venv && source .venv/bin/activate
    ```

## Install dependencies

Make sure the `.venv` virtual environment is active before installing dependencies.

### Running the project (recommended)

Install the already-pinned dependencies from `requirements.txt` using `uv pip`:

```bash
uv pip install -r requirements.txt
```

This is the standard setup path for users who want to run the project. It does not modify
`requirements.txt`.

### Updating dependencies

Only run these commands when intentionally changing the project's dependencies. First update
`requirements.in`, then use `uv`'s built-in compilation and synchronization tools:

```bash
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt
```

`uv pip compile` resolves and writes pinned versions to `requirements.txt`; `uv pip sync` makes the
active virtual environment match that file exactly, including removing packages not listed in it.

## Running the notebooks

After installing the project dependencies, install JupyterLab separately in the
active virtual environment:

```bash
uv pip install jupyterlab
```

From the project root, launch JupyterLab in the notebooks directory so relative
dataset paths resolve correctly:

```bash
cd notebooks
jupyter lab
```

Open the notebooks and run their cells in order. See
[`notebooks/README.md`](notebooks/README.md) for dataset prerequisites,
notebook order, sampling settings, and output paths.

## Running the tests

From the project root, run the same test command used by CI:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s .
```

This command uses the repository's configured source path and recursively executes
the categorized test suites under `tests/`.

## Running the main script

With the virtual environment active, run the optimizer from the repository root:

```bash
PYTHONPATH=src python main.py
```

## Execution plots

Each run saves `history.json` and four figures as PNG and PDF files under
`plots/<run-id>/`. See [`plots/README.md`](plots/README.md) for the run-folder
layout, figure details, and offline regeneration instructions.

## Execution logs

Each run saves its console output and final test report under `logs/<run-id>/`.
See [`logs/README.md`](logs/README.md) for details.
