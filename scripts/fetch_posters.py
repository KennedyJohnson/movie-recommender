"""Look up TMDB poster paths for MovieLens movies (cached in data/posters.json).

Needs a free TMDB API key: https://www.themoviedb.org/settings/api

    TMDB_API_KEY=... python scripts/fetch_posters.py [dataset] [min_ratings]
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import data

CACHE = data.DATA / "posters.json"


def main(name="ml-latest", min_ratings="100"):
    key = os.environ["TMDB_API_KEY"]
    counts = pd.read_parquet(data.DATA / name / "ratings.parquet", columns=["movieId"])["movieId"].value_counts()
    movies = data.movies(name)
    movies = movies[movies["movieId"].isin(counts.index[counts >= int(min_ratings)]) & movies["tmdbId"].notna()]

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [int(t) for t in movies["tmdbId"] if str(int(t)) not in cache]
    print(f"{len(todo):,} posters to look up ({len(cache):,} cached)")
    session = requests.Session()

    def lookup(tmdb_id):
        resp = session.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                           params={"api_key": key}, timeout=30)
        return tmdb_id, resp.json().get("poster_path") if resp.ok else None

    with ThreadPoolExecutor(16) as pool:
        for n, (tmdb_id, path) in enumerate(pool.map(lookup, todo), 1):
            cache[str(tmdb_id)] = path
            if n % 1000 == 0:
                print(f"  {n:,}/{len(todo):,}")
                CACHE.write_text(json.dumps(cache))
    CACHE.write_text(json.dumps(cache))
    print(f"{sum(v is not None for v in cache.values()):,} posters found")


if __name__ == "__main__":
    main(*sys.argv[1:])
