---
name: code-review
description: Review pull requests and code changes in this repository for correctness, mathematical validity, test quality, type safety, and CI integrity. Use when reviewing a pull request, merge request, diff, or branch.
license: MIT
compatibility: Python 3.13 project using Ruff, mypy strict mode, unittest, evolutionary algorithms, and TabNet.
metadata:
  author: PEHPO-master-thesis
  version: "1.0"
---

Review changed code against its intended behavior and this repository's engineering
standards. Report only actionable, high-confidence issues introduced by the diff.

## Required context

1. Read `AGENTS.md` before reviewing.
2. Read the pull request description and linked issue when present.
3. Inspect changed code with its direct callers, tests, and configuration. Do not
   infer contracts from a diff alone.
4. Treat existing behavior as intentional unless repository documentation, tests,
   or direct call sites establish otherwise.

## Review priorities

Review in this order:

1. **Correctness:** Wrong outputs, broken control flow, invalid assumptions,
   regressions, data leakage, nondeterminism, boundary failures, and resource
   leaks.
2. **Algorithmic validity:** Mathematical expressions must match their stated
   source or contract; preserve variable meaning, bounds, sampling independence,
   and fitness/evaluation semantics.
3. **Data and model integrity:** Check feature/label separation, stratified split
   feasibility, train/validation/test isolation, reproducibility, device
   selection, model configuration, and metric inputs.
4. **Test validity:** Tests must fail for the behavior they claim to protect,
   isolate external dependencies, use realistic valid fixtures, and make
   assertions suited to their data types.
5. **Type safety and maintainability:** Preserve strict mypy compatibility,
   complete Python type hints, encapsulation, explicit dependencies, pdoc3-style
   docstrings, and repository formatting.
6. **CI and dependency integrity:** Workflow changes must use pinned project
   dependencies, run the documented commands, and trigger for every affected
   path.

## Evidence standard

Only report a finding when all apply:

- It is caused by changed code or makes a changed path incorrect.
- A concrete input, execution path, or incompatible contract demonstrates it.
- The proposed fix is specific and within pull request scope.
- The issue is not contradicted by executed tests, language semantics, or an
  existing contract.

When practical, reproduce disputed behavior with the smallest focused command
before reporting it. Do not treat a hypothetical future refactor as a current
bug.

For test assertions involving NumPy arrays, pandas objects, mocks, or nested
containers:

- Verify Python and library comparison semantics with the actual referenced
  objects before claiming the assertion raises or returns an ambiguous result.
- Distinguish a passing but less-clear assertion from a failing assertion.
- Report the former only as a low-severity maintainability improvement, not a
  correctness blocker.
- Prefer explicit identity or array-aware assertions when the tested contract
  requires identity or element-wise equality.

## Repository-specific checks

- Python source targets **3.13**. Functions require strict, complete type hints;
  omit a return annotation only on `__init__`.
- Do not accept new global mutable algorithm state. Dependencies and stochastic
  sources must be explicit.
- Iterative algorithms must release temporary per-generation state when it is no
  longer needed.
- Numerical values must enforce documented bounds and avoid overflow,
  underflow, and invalid search-space values.
- Randomized behavior must declare and preserve required seed semantics.
- Tests should use `unittest` and run with:

  ```bash
  PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
  ```

- Relevant checks are:

  ```bash
  .venv/bin/ruff check src/ tests/
  .venv/bin/mypy src/
  ```

## Finding format

Write findings in severity order. Every finding needs a file, exact changed line,
failure mode, and direct fix:

```text
<file>:L<line>: 🔴 bug: <concrete broken behavior>. <specific fix>.
```

Use `🟡 risk:` for a demonstrated fragile behavior and `🔵 nit:` only for
non-blocking clarity or style improvements. Do not report style preferences
already enforced by Ruff or formatters.

If no actionable findings remain after checking the relevant behavior, state
that the review found no blocking issues. Do not invent findings to increase
coverage.
