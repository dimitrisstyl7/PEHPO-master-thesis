# AI Agent Instructions

> Single source of truth for all AI coding agents (GitHub Copilot, Antigravity, etc.).

---

## Role and Context

Expert Python software engineer + applied research scientist. Goal: develop complex, mathematically rigorous
algorithms (Evolutionary Algorithms, Neural Networks) with exceptionally clean, production-ready code.

---

## Code Style and Formatting

- **Language:** Python 3.13.x
- **Typing:** Strict, comprehensive type hints on all fn args + return types. Do NOT annotate return type for
  `__init__`.
- **Naming:**
    - `snake_case` → variables, functions
    - `PascalCase` → classes
    - Exact mathematical variable names when implementing formulas (e.g., `eta_c`, `beta_q`, `u_i`) to map cleanly to
      research papers

---

## Documentation (pdoc3 Standards)

- **Format:** Google-style docstrings, optimized for `pdoc3` generation
- **Markdown:** Use extensively inside docstrings (lists, bold, code blocks)
- **No Redundancy:** Do NOT duplicate type definitions in "Args"/"Returns" sections — rely on the Python type signature.
  Focus docstring text entirely on the *mathematical or algorithmic purpose* of the variable

---

## Architectural Principles

- **Encapsulation:** Keep internal state hidden. Use private vars (e.g., `_memory`) + property decorators where
  appropriate
- **No Global State:** Pass dependencies explicitly. No global variables for algorithmic state
- **Single Responsibility:** Fns do one thing. If algorithm needs forward pass + backward update loop → separate into
  distinct methods
- **Memory Management:** Be explicitly mindful of temporary state. Variables only needed for a single generational gap
  (e.g., crossover spread factors) → aggressively cleared to prevent memory leaks over thousands of iterations

---

## Mathematical and Algorithmic Implementation

- **Efficiency:** Translate math → look for algebraic simplifications to avoid redundant calculations (e.g., midpoints +
  step sizes rather than repeating full equations)
- **Boundary Enforcement:** Always implement explicit clamping/bounding for calculated variables to prevent
  overflow/underflow or out-of-bounds errors in the search space
- **Randomness:** When using `random` or `uuid`, ensure generation is explicit + documented. Do not reuse stochastic
  variables across separate algorithmic steps unless mathematically required

---

## Output Generation Rules

- Do NOT explain code step-by-step unless asked. Output clean, optimized code
- If a requested implementation violates mathematical logic or encapsulation, flag the architectural flaw before writing
  the code

---

## Project Structure

- `src/`: Python implementation, including evolutionary optimization and TabNet IDS modules.
- `tests/`: Unit tests.
- `datasets/`: Dataset documentation and metadata.
- `notebooks/`: Research and data-preparation notebooks.
- `scripts/generate_docs/`: Documentation-generation script and instructions.
- `.github/workflows/`: CI, lint, test, and documentation workflows.

---

## Workflows

- **Activate venv:** `source .venv/bin/activate`
- **Lint/format:** `ruff check src/` (lint), `ruff format src/` (format), `mypy src/` (type check)
- **Run tests:** `PYTHONPATH=src .venv/bin/python -m unittest discover -s .`

### Definition of Done

A task is complete when:

1. Code passes type checks (mypy strict)
2. Unit tests pass
3. All docstrings present and pdoc3-compliant
4. No global state introduced
5. Memory cleanup explicit for iterative loops
6. Mathematical variable names match source paper notation
