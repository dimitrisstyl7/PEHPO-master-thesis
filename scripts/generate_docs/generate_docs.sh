#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN=""
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/Scripts/python.exe"
else
  echo "Missing venv python at .venv/bin/python or .venv/Scripts/python.exe." >&2
  echo "Create .venv and install dependencies first." >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/src/__init__.py" ]]; then
  echo "Missing ${REPO_ROOT}/src/__init__.py. src must be a package." >&2
  exit 1
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" - "$@" <<'PY'
import os
import runpy
import sys

repo_root = os.getcwd()
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

sys.argv = ["pdoc", "--html", "--output-dir", "docs", "--force", "src", *sys.argv[1:]]
runpy.run_module("pdoc", run_name="__main__")
PY
