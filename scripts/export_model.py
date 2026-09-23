"""Train the deployed model on all ratings and export small artifacts for the API.

Only movies with at least `min_ratings` ratings are exported, which keeps the artifacts
a few MB and keeps obscure titles with noisy factors out of recommendations.

    python scripts/export_model.py [svdf|pmf|als] [dataset] [min_ratings]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import data
from recsys.models import PMF, FunkSVD, ImplicitALS
from scripts.evaluate import MODELS

OUT = Path(__file__).resolve().parents[1] / "api" / "artifacts"
POSTERS = data.DATA / "posters.json"  # from scripts/fetch_posters.py
KINDS = {"svdf": (FunkSVD, "SVDF (Funk SVD)"), "pmf": (PMF, "PMF"), "als": (ImplicitALS, "Implicit ALS")}


def readable_title(title):
    """MovieLens files articles at the end ("Matrix, The" -> "The Matrix") and appends
    alternate titles in parentheses, which we drop for display."""
    title = re.sub(r"\s*\((a\.k\.a\. )?[^()]*\)$", "", title) or title
    return re.sub(r"^(.*?), (The|A|An)( \(.*\))?$", r"\2 \1\3", title)


def main(kind="svdf", name="ml-latest", min_ratings="100"):
    cls, label = KINDS[kind]
    r = data.load(name)
    print(f"Training {label} on {len(r.ratings):,} ratings")
    params = MODELS[label][1]
    model = cls(**params).fit(r.users, r.items, r.ratings, r.n_users, r.n_items)

    counts = np.bincount(r.items, minlength=r.n_items)
    means = np.bincount(r.items, weights=r.ratings, minlength=r.n_items) / counts
    posters = json.loads(POSTERS.read_text())
    meta = data.movies(name).set_index("movieId")
    tmdb_ids = meta["tmdbId"].reindex(r.movie_ids).to_numpy()
    has_poster = np.array([not np.isnan(t) and posters.get(str(int(t))) is not None for t in tmdb_ids])
    # every movie the site shows needs a cover, so drop the few TMDB has no poster for
    keep = np.flatnonzero((counts >= int(min_ratings)) & has_poster)
    keep = keep[np.argsort(-counts[keep])]  # most-rated first, so row 0.. is the onboarding list
    print(f"{(counts >= int(min_ratings)).sum() - len(keep)} movies dropped for missing posters")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "model.npz",
        kind=kind,
        Q=model.Q[keep].astype(np.float32),
        bi=(model.bi[keep] if hasattr(model, "bi") else np.zeros(len(keep))).astype(np.float32),
        mu=np.float32(getattr(model, "mu", 0.0)),
        reg=np.float32(params.get("reg", 0.1)),
        alpha=np.float32(params.get("alpha", 0.0)),
    )

    movies = []
    for idx in keep:
        mid = int(r.movie_ids[idx])
        row = meta.loc[mid]
        match = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", row["title"])
        tmdb = int(row["tmdbId"])
        movies.append({
            "id": mid,
            "title": readable_title(match.group(1) if match else row["title"]),
            "year": int(match.group(2)) if match else None,
            "genres": [] if row["genres"] == "(no genres listed)" else row["genres"].split("|"),
            "tmdb": tmdb,
            "poster": posters[str(tmdb)],
            "n": int(counts[idx]),
            "avg": round(float(means[idx]), 2),
        })
    (OUT / "movies.json").write_text(json.dumps(movies, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(movies):,} movies to {OUT}")


if __name__ == "__main__":
    main(*sys.argv[1:])
