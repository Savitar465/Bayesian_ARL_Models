# Bayesian & ARL Models

Starter repository for experimenting with Bayesian and ARL-related model workflows in notebooks.

## Project Structure

```text
Bayesian&ARLModels/
├── data/           # Input datasets (.gitkeep keeps folder in git)
├── models/         # Saved model artifacts (.gitkeep keeps folder in git)
├── sample.ipynb    # Example notebook entry point
└── README.md
```

## Prerequisites

- Python 3.10+ recommended
- `pip`

## Setup

### PowerShell (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install jupyter ipykernel pandas numpy matplotlib scipy pymc arviz
```

### macOS/Linux (bash)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install jupyter ipykernel pandas numpy matplotlib scipy pymc arviz
```

## How to Run

1. Activate your virtual environment.
2. Start Jupyter in the repository root:

```bash
jupyter notebook
```

3. Open `sample.ipynb` and run cells.

## Data and Model Artifacts

- Put raw/intermediate datasets in `data/`.
- Save trained model outputs in `models/`.

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for development workflow and pull request guidelines.
