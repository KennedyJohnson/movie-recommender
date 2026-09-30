# DSCI 4093 capstone (2024): where this project started

My senior capstone, moved here from its own repository as reference for how this project picked its models. [Presentation slides](https://kennedyjohnson.github.io/DSCI%204093%20Presentation.pdf).

## Question

Which recommender algorithm best predicts how a user will rate a single movie, and does the answer hold as the amount of data grows?

## What was compared

- **Exploration** (`CountMoviesByGenre.ipynb`, `FrequencyOfRatings.ipynb`): movie counts by genre and the distribution of ratings in MovieLens 32M.
- **Neighborhood baselines** (`DSCI4093_MovieRecommender.ipynb`, Python / [LensKit](https://lkpy.readthedocs.io/)): user-user and item-item collaborative filtering on users with 100+ ratings, 5-fold cross-validation, RMSE.
- **Algorithm bake-off** (`Rcode/movieLens.Rmd`, R / [recommenderlab](https://cran.r-project.org/package=recommenderlab)): 8 algorithms (UBCF, IBCF, POPULAR, RANDOM, LIBMF, SVD, SVDF, ALS) on three random samples of increasing size, each scored by 5-fold cross-validation (10 known ratings per test user, 3+ stars counts as "good") on RMSE, MSE and MAE, plus ROC curves for top-N lists (`figures/roc-ubcf-ibcf-popular-random.png`).

## Finding

Only **SVDF** (Funk SVD) and **PMF** consistently beat the popularity baseline at every data size; the other algorithms didn't. For predicting a single movie's rating, matrix factorization was the approach to use.

## How that carried into this project

The main project re-ran the comparison on the newest MovieLens release (33.8M ratings), with a held-out test set ([../results/results.md](../results/results.md)):

- **Single-rating prediction: SVDF still wins** (RMSE 0.767, MAE 0.578, vs 0.797 for PMF and 0.856 for the bias baseline), confirming the capstone.
- **Ranked "what to watch next" lists are a different task.** Rating-error models rank the full catalog poorly (they push obscure movies a few fans loved), so the site serves **EASE**, an implicit-feedback model with 7x SVDF's nDCG@10. See the main [README](../README.md#methods-finding-the-best-ranking-model) for how it was chosen.

## Running it

The MovieLens data isn't included (its license doesn't allow redistribution). Download [ml-32m.zip](https://files.grouplens.org/datasets/movielens/ml-32m.zip) and unzip it into `capstone/ml-32m/`. The R analysis also reads `capstone/subsetData/ratings_{small,medium,large}.csv`, random samples of `ratings.csv`; recreate them locally. Both folders are git-ignored.
