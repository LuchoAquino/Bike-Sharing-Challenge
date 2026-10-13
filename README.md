# challenge
machine learning challenge

## Setup

```bash
uv sync
uv run nbstripout --install   # once per clone: strips notebook outputs on commit
```

Then select the `.venv` kernel in your editor.

## Organisation

- `notebooks/01_preprocessing_eda.ipynb` — 3.1, writes cleaned data to `data/clean/` (parquet)
- `notebooks/02_clustering_trips.ipynb` — 3.2 A
- `notebooks/03_clustering_stations.ipynb` — 3.2 B
- `notebooks/04_prediction.ipynb` — 3.3, saves the trained model to `models/`
- `notebook.ipynb` — final submission, self-contained (no local imports)

Rules:
- One person edits a notebook at a time; work on your own branch and open a PR.
- After `git pull`, run `uv sync`.
- Add libraries with `uv add <lib>` and commit `pyproject.toml` + `uv.lock` together. On a `uv.lock` conflict, fix `pyproject.toml` then run `uv lock`.
