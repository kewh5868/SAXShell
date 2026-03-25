# Contributing

## Local setup

Create a Python 3.12 environment and install the package in editable mode:

```bash
python -m pip install -e .
```

Install pre-commit and enable the hooks:

```bash
python -m pip install pre-commit
pre-commit install
```

## Running tests

Run the full suite:

```bash
pytest -q
```

For focused work, prefer the smallest test slice that exercises your change.

## Docs preview

Install the pinned docs dependencies and start the local server:

```bash
python -m pip install -r requirements/docs.txt
mkdocs serve
```

Build the site locally the same way CI does:

```bash
mkdocs build --strict
```

## Formatting and linting

The repository uses pre-commit hooks for:

- Black
- isort
- Flake8
- codespell
- Prettier for Markdown, JSON, and YAML
- docformatter

Run them manually if needed:

```bash
pre-commit run --all-files
```

## Branch and PR expectations

The current repository rules require:

- changes through pull requests
- passing status checks
- signed commits on protected branches

## Documentation expectations

When you add or change user-facing behavior:

- update the relevant task-based docs page
- keep README short and link back into the docs site
- avoid duplicating large how-to sections in multiple places

## TODO

TODO: expand this page with any future contributor conventions around release
notes or benchmark workflows if they become formalized.

TODO: document the remaining migration steps as SAXSShell is renamed from
SAXShell, including which internal package paths and compatibility shims are
still intentionally using the legacy name.
