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

## Install dependency tooling ([pip-tools](https://pypi.org/project/pip-tools/))

Make sure the `.venv` virtual environment is active before running this command.

```bash
python -m pip install pip-tools
```

## Dependencies installation

Make sure the `.venv` virtual environment is active before installing/syncing dependencies.

```bash
pip-compile requirements.in && pip-sync requirements.txt
```
