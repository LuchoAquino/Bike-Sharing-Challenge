# challenge
machine learning challenge

## Setup

```bash
uv sync
uv run nbstripout --install   # once per clone: strips notebook outputs on commit
```

Then select the `.venv` kernel in your editor.

## Organisation

Raw CSV files (`cities.csv`, `bikes.csv`, `bike_types.csv`, `stations.csv`, `stations_status.csv`, `trips_small.csv`) go in the project root (not versioned).

- `notebooks/01_preprocessing_eda.ipynb` — 3.1, EDA and cleaning of every dataset, writes the cleaned CSVs to `data_cleaned/`
- `notebooks/02_clustering_trips.ipynb` — 3.2 A, clustering of bike trips
- `notebooks/03_clustering_stations.ipynb` — 3.2 B, clustering of stations, writes to `output/part_b/`
- `notebooks/04_prediction.ipynb` — 3.3, supervised model (uses `data_external/` for weather and `bike_sharing_challenge_2026_eval/` for the evaluation set)

Run the notebooks in order: 01 creates the files that 02, 03 and 04 read.

Rules:
- One person edits a notebook at a time; work on your own branch and open a PR.
- After `git pull`, run `uv sync`.
- Add libraries with `uv add <lib>` and commit `pyproject.toml` + `uv.lock` together. On a `uv.lock` conflict, fix `pyproject.toml` then run `uv lock`.
