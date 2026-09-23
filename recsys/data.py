from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Ratings:
    users: np.ndarray       # dense user index, int32
    items: np.ndarray       # dense item index, int32
    ratings: np.ndarray     # float32
    movie_ids: np.ndarray   # item index -> MovieLens movieId
    n_users: int
    n_items: int

    def subset(self, mask):
        return Ratings(self.users[mask], self.items[mask], self.ratings[mask],
                       self.movie_ids, self.n_users, self.n_items)


def load(name="ml-latest", min_item_ratings=20, min_user_ratings=20) -> Ratings:
    df = pd.read_parquet(DATA / name / "ratings.parquet", columns=["userId", "movieId", "rating"])
    # one pass each is enough in practice; filtering items barely changes user counts
    item_counts = df["movieId"].value_counts()
    df = df[df["movieId"].isin(item_counts.index[item_counts >= min_item_ratings])]
    user_counts = df["userId"].value_counts()
    df = df[df["userId"].isin(user_counts.index[user_counts >= min_user_ratings])]

    users, _ = pd.factorize(df["userId"])
    items, movie_ids = pd.factorize(df["movieId"])
    return Ratings(users.astype(np.int32), items.astype(np.int32),
                   df["rating"].to_numpy(np.float32), np.asarray(movie_ids),
                   int(users.max()) + 1, len(movie_ids))


def movies(name="ml-latest") -> pd.DataFrame:
    m = pd.read_parquet(DATA / name / "movies.parquet")
    links = pd.read_parquet(DATA / name / "links.parquet")[["movieId", "tmdbId"]]
    return m.merge(links, on="movieId", how="left")


def split(r: Ratings, test_frac=0.2, seed=4093):
    """Random per-rating holdout; every user keeps ~80% of their ratings for training."""
    test = np.random.default_rng(seed).random(len(r.ratings)) < test_frac
    return r.subset(~test), r.subset(test)
