# Parallel Evolutionary Algorithms for Hyperparameter Optimization in Artificial Intelligence Models

## Python version

Python 3.13.x

## Clone repository

```bash
git clone https://github.com/dimitrisstyl7/PEHPO-master-thesis.git
```

## Virtual environment creation ([venv](https://docs.python.org/3/library/venv.html))

- Go to project directory

```bash
cd PEHPO-master-thesis
```

- Create and activate virtual environment (`.venv`)
    - On Linux / Mac
       ```bash
       python3 -m venv .venv && source .venv/bin/activate
       ```
    - On Windows (PowerShell 7+)
        ```powershell
        python -m venv .venv && .venv\Scripts\Activate.ps1
        ```

## Install dependencies

Make sure the `.venv` virtual environment is active before installing dependencies.

### Running the project (recommended)

Install the already-pinned dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

This is the standard setup path for users who want to run the project. It does not modify
`requirements.txt`.

### Updating dependencies

Only run these commands when intentionally changing the project's dependencies. First update
`requirements.in`, then install [pip-tools](https://pypi.org/project/pip-tools/) and regenerate
the pinned lockfile:

```bash
python -m pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
```

`pip-compile` resolves and writes pinned versions to `requirements.txt`; `pip-sync` makes the
active virtual environment match that file exactly, including removing packages not listed in it.

## Running the tests

From the project root, run the same test command used by CI:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s .
```

This command uses the repository's configured source path and recursively executes
the categorized test suites under `tests/`.
