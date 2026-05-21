# Contributing

Thanks for taking a look at `wandb-cache`.

## Development Setup

```bash
pip install -e ".[dev,examples]"
pre-commit install
```

## Checks

Run the offline test suite:

```bash
pytest
```

Run formatting and linting:

```bash
pre-commit run --all-files
```

Build and validate the package:

```bash
python -m build
python -m twine check dist/*
```

The examples use public W&B projects, but they still require normal W&B auth through `WANDB_API_KEY` or
`~/.netrc`.
