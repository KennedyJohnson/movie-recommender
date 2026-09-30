# Movie Recommender

Rate movies → ranked recommendations. Builds on the DSCI-4093 capstone (its notebooks/R code now live in `capstone/`; data not included). Site: `docs/` on GitHub Pages (legacy branch build, main /docs). The deployed model is **EASE** and runs fully in the browser; the FastAPI app in `api/` is optional/local only (old Render deploy in `render.yaml`).

## Layout
- `recsys/` - `data.py` (loads `data/<name>/*.parquet`), `models.py` (Popularity, BiasBaseline, FunkSVD, PMF, ImplicitALS, BPR, EASE), `metrics.py`.
- `scripts/download_data.py [ml-latest|ml-32m|ml-latest-small]` → `data/` (gitignored; MovieLens license forbids redistribution - never commit raw ratings).
- `scripts/evaluate.py` / `tune.py` → `results/`. Model params live in `evaluate.MODELS`.
- `scripts/fetch_posters.py` needs `TMDB_API_KEY`; caches `data/posters.json`. Movies without posters are dropped from export.
- `scripts/export_model.py ease` → `api/artifacts/{model.npz,movies.json}` (EASE pruned to top-200 neighbours/movie).
- `scripts/export_static.py` → `docs/data/{movies.json,meta.json,ease/<shard>.bin}` (100 rows/shard; int32 header+indptr+indices, float32 data).

## Site
`docs/index.html`, `docs/recommender.js` (JS port of `api/recommender.py`, EASE only - keep the two in sync), `docs/app.js` (UI; ratings in localStorage). Serve with `python -m http.server -d docs`.

## Automation
`.github/workflows/refresh.yml` - monthly: retrains only if GroupLens ml-latest Last-Modified differs from `docs/data/source_modified.txt` (or manual `force`). Needs repo secret `TMDB_API_KEY`. Commits `docs/data/last_updated.json` every run (keeps cron alive), requests a Pages build. Failure → `stale-data` issue.
