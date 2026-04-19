# `generate_docs.sh`

This script generates HTML API docs for the `src` package with `pdoc`.

## Script lines explained

| Line  | Command                                                                       | Purpose                                                                              |
|-------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 1     | `#!/usr/bin/env bash`                                                         | Run script with Bash.                                                                |
| 2     | `set -eu`                                                                     | Stop on errors (`-e`) and unset vars (`-u`).                                         |
| 4     | `SCRIPT_DIR=...`                                                              | Get absolute path of script folder.                                                  |
| 5     | `REPO_ROOT=...`                                                               | Resolve repository root (`scripts/generate_docs/../..`).                             |
| 7     | `cd "${REPO_ROOT}"`                                                           | Run from repo root so paths resolve consistently.                                    |
| 9-17  | `if [[ -x .venv/bin/python ]] ... elif [[ -x .venv/Scripts/python.exe ]] ...` | Use repo virtualenv Python (Linux/macOS first, Windows fallback).                    |
| 15-18 | `if [[ ! -f "${REPO_ROOT}/src/__init__.py" ]] ...`                            | Ensure `src` is a package before generating docs.                                    |
| 21-33 | `python - "$@" <<'PY' ...`                                                    | Run inline Python that prepends `src` to `sys.path` and executes `pdoc` via `runpy`. |
| 31    | `sys.argv = ["pdoc", "--html", "--output-dir", "docs", ...`                   | Set default output directory to `docs` and pass through any extra CLI args.          |

## How to run

From repository root:

```bash
bash scripts/generate_docs/generate_docs.sh
```

Script always uses repo-local venv Python:

- `./.venv/bin/python` on Linux/macOS
- `./.venv/Scripts/python.exe` on Windows

Generated docs include:

- root package page at `docs/src/index.html`
- subpackage docs under `docs/src/<subpackage>/...`
