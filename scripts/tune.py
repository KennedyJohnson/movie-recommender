"""Greedy hyperparameter search for the ranking models on a validation split carved out
of the training data (the test split from evaluate.py is never touched).

    python scripts/tune.py [dataset]

Each ALS stage tries a few values of one knob, keeping the best-so-far for the rest;
writes results/tuning.{json,md}.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys import data, metrics
from recsys.models import BPR, EASE, ImplicitALS

RESULTS = Path(__file__).resolve().parents[1] / "results"

ALS_STAGES = [
    ("confidence", [dict(), dict(min_rating=3.5), dict(min_rating=3.0, graded=True),
                    dict(graded=True)]),
    ("alpha", [dict(alpha=a) for a in (3.0, 10.0, 30.0)]),
    ("reg", [dict(reg=r) for r in (0.1, 5.0, 50.0)]),
    ("factors", [dict(factors=f) for f in (64, 128, 200)]),
    ("cg_steps", [dict(cg_steps=c) for c in (3, 8)]),
    ("pop_beta", [dict(pop_beta=b) for b in (0.0, 0.25)]),
]
OTHERS = [
    ("EASE", EASE, [dict(reg=r) for r in (200.0, 1000.0, 4000.0)]),
    ("EASE (>=3.5 stars)", EASE, [dict(reg=1000.0, min_rating=3.5)]),
    ("BPR", BPR, [dict(), dict(factors=128, epochs=40)]),
]


def main(name="ml-latest"):
    r = data.load(name)
    train, _ = data.split(r)
    fit_set, val = data.split(train, seed=7)
    print(f"{len(fit_set.ratings):,} fit / {len(val.ratings):,} validation ratings")
    log = []

    def run(label, cls, params):
        t0 = time.time()
        model = cls(**params).fit(fit_set.users, fit_set.items, fit_set.ratings,
                                  r.n_users, r.n_items, verbose=False)
        row = dict(model=label, params=params, seconds=round(time.time() - t0, 1),
                   **metrics.ranking(model, fit_set, val, n_users=5_000))
        log.append(row)
        print(f"{label:22s} {json.dumps(params):60s} ndcg@10 {row['ndcg@10']:.4f}  "
              f"recall@10 {row['recall@10']:.4f}  cov {row['catalog_coverage']:.3f}  {row['seconds']}s",
              flush=True)
        return row

    best = {}
    for stage, options in ALS_STAGES:
        rows = [run(f"ALS/{stage}", ImplicitALS, {**best, **o}) for o in options]
        best = max(rows, key=lambda x: x["ndcg@10"])["params"]
    print("best ALS:", best)
    for label, cls, options in OTHERS:
        for o in options:
            run(label, cls, o)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tuning.json").write_text(json.dumps({"best_als": best, "runs": log}, indent=2),
                                         encoding="utf-8")
    lines = ["| Model | params | precision@10 | recall@10 | ndcg@10 | coverage | seconds |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {x['model']} | `{json.dumps(x['params'])}` | {x['precision@10']:.4f} | "
              f"{x['recall@10']:.4f} | {x['ndcg@10']:.4f} | {x['catalog_coverage']:.3f} | {x['seconds']} |"
              for x in log]
    (RESULTS / "tuning.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
