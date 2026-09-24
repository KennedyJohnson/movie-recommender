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
| Implicit ALS (defaults) | – | – | 0.227 | 0.225 | 0.302 | 11.5% |
| **Implicit ALS (tuned)** | – | – | 0.271 | 0.255 | 0.363 | 9.9% |
| **EASE** | – | – | **0.306** | **0.276** | **0.408** | 9.4% |
| BPR | – | – | 0.172 | 0.160 | 0.221 | 16.8% |

The raw numbers are in [results/results.json](results/results.json).

**SVDF is still the best at predicting a single rating.** Its RMSE is the lowest, which confirms the capstone's finding on fresh data.

**Implicit-feedback models are far better at building a ranked list.** Even untuned ALS scores more than 5 times SVDF's nDCG@10 and nearly double the popularity baseline. The explicit models are trained only on the ratings people chose to give. When they rank the full catalog, they push up obscure movies that a few fans rated highly. Implicit ALS ([Hu, Koren & Volinsky, 2008](http://yifanhu.net/PUB/cf.pdf)) models whether someone rated a movie at all, and it treats unrated movies as weak negatives. That matches the "what should I watch next" task much better.

**EASE is the best ranking model, and it's what the app serves.** See below for how we got there.

## Methods: finding the best ranking model

Settings were chosen on a validation split carved out of the training data ([scripts/tune.py](scripts/tune.py)), one setting at a time, so the test set played no part in tuning. Scores are nDCG@10.

| Step | Validation | Test |
|---|---|---|
| 1. ALS baseline: every rating counts as "watched" | 0.207 | 0.302 |
| 2. Use the star value: drop ratings below 3★, and give 5★ more confidence than 3★ | 0.230 | |
| 3. Stronger regularization (λ 0.1 → 5), and down-weight popular movies by (count / mean)^-0.25 | 0.240 | 0.363 |
| 4. Switch to EASE, trained on ratings of 3.5★ or more | 0.260 | 0.408 |
| 5. Prune EASE to 200 weights per movie so it fits free hosting (576 MB → 19 MB) | | **0.393** |

**What didn't help:** more latent factors (128 factors: 0.212, 200 factors: 0.196; both overfit), more conjugate-gradient steps (no change), and BPR ([Rendle et al., 2009](https://arxiv.org/abs/1205.2618)), which scored 0.221 on test.

**How EASE works.** EASE ([Steck, 2019](https://arxiv.org/abs/1905.03375)) is a linear item-to-item model with no latent factors.
- X is the binary user × movie "liked" matrix.
- EASE learns weights B that minimize ‖X − XB‖² + λ‖B‖², with B's diagonal fixed at 0 so a movie can't predict itself.
- The solution is closed-form: P = (XᵀX + λI)⁻¹, then B_ij = −P_ij / P_jj. Training is one matrix inversion (about 30 s).
- A user's score for movie j is Σ_i X_ui·B_ij, a weighted vote from every movie they liked.

Keeping only the 50, 100, 200 or 500 strongest weights per movie gives test nDCG@10 of 0.380, 0.387, 0.393 and 0.398. Even 50 beats tuned ALS.

## How it works

- **Training** ([recsys/models.py](recsys/models.py)): SVDF, PMF, implicit ALS, BPR and EASE are implemented from scratch with numpy and numba. SVDF, PMF and BPR use SGD, implicit ALS uses a conjugate-gradient solver, and EASE is solved in closed form. Training on the full 32M filtered ratings takes about 3 minutes per model on a desktop CPU.
- **New users:** visitors aren't in the training data, and EASE doesn't need them to be. A visitor's scores are the sum of the weight rows for the movies they rated 3.5★ or more, which takes under a millisecond with no retraining ([api/recommender.py](api/recommender.py)). The API can still serve ALS, SVDF or PMF (`export_model.py als|svdf|pmf`).
- **Serving:** only the pruned EASE weights are exported (12 MB compressed). The export covers the 12,087 movies that have at least 100 ratings and a TMDB poster. A FastAPI service ([api/main.py](api/main.py)) serves them to a static frontend ([docs/](docs/)).

## Run it locally

```bash
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r api/requirements.txt

python scripts/download_data.py                       # ml-latest from GroupLens (~1 GB unzipped)
python scripts/evaluate.py                            # model comparison -> results/
TMDB_API_KEY=... python scripts/fetch_posters.py      # poster paths from TMDB (free key)
python scripts/export_model.py ease                   # train + export -> api/artifacts/

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
