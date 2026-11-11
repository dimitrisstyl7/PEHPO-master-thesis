# GitHub Actions Workflows

[⬅ Back to main README](../../README.md#readme-files)

`ci.yml` is the entry-point workflow. It runs on pushes and pull requests
targeting `main` when a CI-relevant path changes, and supports a manual
`workflow_dispatch` run.

## Selective job execution

The `changes` job uses
[`dorny/paths-filter`](https://github.com/dorny/paths-filter) to select only
the affected reusable workflows:

| Changed path                          | Workflows run     |
|---------------------------------------|-------------------|
| `src/**`                              | lint, tests, docs |
| `tests/**`                            | lint, tests       |
| `requirements.in`, `requirements.txt` | lint, tests, docs |
| `mypy.ini`                            | lint, tests       |
| `scripts/generate_docs/**`            | docs              |
| `.github/workflows/**`                | lint, tests, docs |
| Manual dispatch                       | lint, tests, docs |

Requirements changes rebuild documentation because the documentation workflow
creates an environment from `requirements.txt`. Any workflow change runs the
full pipeline so modifications to workflow behavior, permissions, or reusable
workflow definitions receive complete validation.

## Workflow dependencies

- `lint.yml` runs Ruff linting, Ruff formatting checks, and mypy.
- `tests.yml` runs the unit-test suite.
- `docs.yml` builds documentation and deploys Pages only for pushes to `main`.

For source changes, the docs workflow waits for lint to succeed. For a
documentation-generator-only change, lint is skipped and docs runs directly.
If lint fails, docs does not run.

`ci.yml` grants `pull-requests: read` so the path filter can inspect changed
files on pull requests. The docs caller receives its additional Pages
deployment permissions only when the docs workflow is selected.
