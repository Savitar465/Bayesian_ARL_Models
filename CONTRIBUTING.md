# Contributing

Thanks for contributing to this project.

## Getting Started

1. Fork and clone the repository.
2. Create and activate a virtual environment.
3. Install required packages:

```bash
pip install jupyter ipykernel pandas numpy matplotlib scipy pymc arviz
```

## Development Workflow

1. Create a branch from `main`:

```bash
git checkout -b feature/short-description
```

2. Keep changes focused and scoped to one purpose.
3. Put data files in `data/` and model artifacts in `models/`.
4. If behavior changes, update documentation in `README.md`.

## Notebook and Code Guidelines

- Keep notebook cells small and readable.
- Prefer deterministic runs (set random seeds when relevant).
- Avoid committing large generated outputs unless required.
- Use clear names for notebooks, scripts, and artifacts.

## Commit and Pull Request Guidelines

- Use clear commit messages in imperative form (for example: `Add ARL baseline notebook`).
- In pull requests, include:
  - what changed,
  - why it changed,
  - how reviewers can run it.
- Link related issues when applicable.

## Reporting Issues

When opening an issue, include:

- expected behavior,
- actual behavior,
- steps to reproduce,
- environment details (OS, Python version, key package versions).
