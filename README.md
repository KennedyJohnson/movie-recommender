# Movie Recommender

Rate a few movies and get a ranked list of what to watch next.

**Live demo:** https://kennedyjohnson.github.io/movie-recommender/

This project builds on my senior capstone, [DSCI 4093](https://github.com/KennedyJohnson/DSCI-4093). The capstone compared eight recommender algorithms on MovieLens 32M and found that only **SVDF** (Funk SVD) and **PMF** consistently beat the popularity baseline at every data size. Here I re-run that comparison on the latest MovieLens release, add implicit-feedback ALS, and deploy the best ranking model as a web app.

## Results

The data is the latest MovieLens release: 33.8M ratings. After keeping users and movies with 20 or more ratings, 204,263 users and 23,426 movies remain. A random 20% of each user's ratings is held out for testing. A test movie counts as a relevant hit if it was rated 4 stars or more.

| Model | RMSE ↓ | MAE ↓ | Precision@10 | Recall@10 | nDCG@10 | Catalog coverage |
|---|---|---|---|---|---|---|
| Popularity | – | – | 0.122 | 0.100 | 0.157 | 0.3% |
| Bias baseline (μ + b_u + b_i) | 0.856 | 0.651 | 0.001 | 0.001 | 0.002 | 0.1% |
| **SVDF (Funk SVD)** | **0.767** | **0.578** | 0.043 | 0.031 | 0.055 | 12.9% |
| PMF | 0.797 | 0.605 | 0.037 | 0.028 | 0.045 | 19.8% |
| **Implicit ALS** | – | – | **0.227** | **0.225** | **0.302** | 11.5% |

The raw numbers are in [results/results.json](results/results.json).

**SVDF is still the best at predicting a single rating.** Its RMSE is the lowest, which confirms the capstone's finding on fresh data.

**Implicit ALS is by far the best at building a ranked list.** Its nDCG@10 is more than 5 times SVDF's and nearly double the popularity baseline. The explicit models are trained only on the ratings people chose to give. When they rank the full catalog, they push up obscure movies that a few fans rated highly. Implicit ALS ([Hu, Koren & Volinsky, 2008](http://yifanhu.net/PUB/cf.pdf)) models whether someone rated a movie at all, and it treats unrated movies as weak negatives. That matches the "what should I watch next" task much better.

The app shows a ranked list, so it serves **Implicit ALS**.

## How it works

- **Training** ([recsys/models.py](recsys/models.py)): SVDF, PMF and implicit ALS are implemented from scratch with numba. SVDF and PMF use SGD. Implicit ALS uses a conjugate-gradient solver. Training on the full 32M filtered ratings takes about 3 minutes per model on a desktop CPU.
- **New users:** visitors aren't in the training data. The API holds the learned movie vectors fixed and solves the ALS least-squares problem for the visitor's own vector from the movies they rated 4 stars or more. This takes a few milliseconds per request, with no retraining ([api/recommender.py](api/recommender.py)).
- **Serving:** only the item factors (about 3 MB) are exported. The export covers the 12,087 movies that have at least 100 ratings and a TMDB poster. A FastAPI service ([api/main.py](api/main.py)) serves them to a static frontend ([docs/](docs/)).

## Run it locally

```bash
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r api/requirements.txt

python scripts/download_data.py                       # ml-latest from GroupLens (~1 GB unzipped)
python scripts/evaluate.py                            # model comparison -> results/
TMDB_API_KEY=... python scripts/fetch_posters.py      # poster paths from TMDB (free key)
python scripts/export_model.py als                    # train + export -> api/artifacts/

cd api && uvicorn main:app --port 8000                # API
cd docs && python -m http.server 5500                 # site at http://localhost:5500
```

## Deployment

- **Frontend:** GitHub Pages, served from `main` / `docs`.
- **API:** Render's free tier, configured in [render.yaml](render.yaml). The free instance sleeps when idle, so the first request after a quiet spell takes about 30 seconds.

## Data and attribution

This project uses the [MovieLens](https://grouplens.org/datasets/movielens/) dataset from GroupLens Research at the University of Minnesota. The raw data is not redistributed here. This is a non-commercial project and is not endorsed by the University of Minnesota or GroupLens Research.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4, Article 19 (December 2015), 19 pages. https://doi.org/10.1145/2827872

Movie posters are provided by [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB.

<img src="docs/tmdb-logo.svg" alt="TMDB logo" height="14">
