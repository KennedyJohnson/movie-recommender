"""Compare recommenders on a fresh MovieLens pull and write results/results.{json,md}.

    python scripts/evaluate.py [dataset]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys import data, metrics
from recsys.models import PMF, BiasBaseline, FunkSVD, ImplicitALS, Popularity

RESULTS = Path(__file__).resolve().parents[1] / "results"

MODELS = {
    "Popularity": (Popularity, {}),
    "Bias baseline": (BiasBaseline, {}),
    "SVDF (Funk SVD)": (FunkSVD, dict(factors=64, epochs=20, lr=0.005, reg=0.02)),
    "PMF": (PMF, dict(factors=64, epochs=20, lr=0.005, reg=0.05)),
    "Implicit ALS": (ImplicitALS, dict(factors=64, iterations=15, reg=0.1, alpha=10.0)),
}


def main(name="ml-latest"):
    r = data.load(name)
    train, test = data.split(r)
    print(f"{name}: {r.n_users:,} users x {r.n_items:,} movies, "
          f"{len(train.ratings):,} train / {len(test.ratings):,} test ratings")

    results = {}
    for label, (cls, params) in MODELS.items():
        print(f"\n== {label}")
        t0 = time.time()
        model = cls(**params).fit(train.users, train.items, train.ratings, r.n_users, r.n_items)
        row = {"train_seconds": round(time.time() - t0, 1)}
        if hasattr(model, "predict"):
            pred = model.predict(test.users, test.items)
            row.update(rmse=metrics.rmse(pred, test.ratings), mae=metrics.mae(pred, test.ratings))
        row.update(metrics.ranking(model, train, test))
        results[label] = row
        print("   ", {k: round(v, 4) for k, v in row.items()})

    RESULTS.mkdir(exist_ok=True)
    meta = {"dataset": name, "users": r.n_users, "movies": r.n_items,
            "train_ratings": len(train.ratings), "test_ratings": len(test.ratings)}
    (RESULTS / "results.json").write_text(json.dumps({"meta": meta, "models": results}, indent=2), encoding="utf-8")

    cols = ["rmse", "mae", "precision@10", "recall@10", "ndcg@10", "catalog_coverage", "train_seconds"]
    lines = ["| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for label, row in results.items():
        cells = [f"{row[c]:.4f}" if c in row and c != "train_seconds" else str(row.get(c, "–"))
                 for c in cols]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
