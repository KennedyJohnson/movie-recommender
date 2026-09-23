"""Download a MovieLens dataset from GroupLens and convert it to parquet.

The raw data is not committed: the MovieLens license forbids redistribution.

    python scripts/download_data.py              # ml-latest (refreshed periodically)
    python scripts/download_data.py ml-32m       # fixed research benchmark
    python scripts/download_data.py ml-latest-small
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / "data"
URL = "https://files.grouplens.org/datasets/movielens/{name}.zip"


def main(name: str = "ml-latest") -> None:
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL.format(name=name)} ...")
    resp = requests.get(URL.format(name=name), timeout=600)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for member in ("ratings.csv", "movies.csv", "links.csv"):
            with zf.open(f"{name}/{member}") as f:
                df = pd.read_csv(f)
            df.to_parquet(out / member.replace(".csv", ".parquet"), index=False)
            print(f"  {member}: {len(df):,} rows")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
