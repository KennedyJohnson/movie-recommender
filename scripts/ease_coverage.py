"""Sweep EASE's popularity debias (pop_beta) and report nDCG@10 vs catalog coverage.

    python scripts/ease_coverage.py [dataset]

Writes results/ease_coverage.{json,md}. Each setting is refit from scratch, unpruned and
with the top-200 pruning the site serves (scripts/export_model.py EASE_TOPK).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys import data, metrics
from recsys.models import EASE

RESULTS = Path(__file__).resolve().parents[1] / "results"
BETAS = [0.0, 0.1, 0.15, 0.175, 0.2, 0.25, 0.3]
TOPKS = [None, 200]


def main(name="ml-latest"):
    r = data.load(name)
    train, test = data.split(r)
    rows = []
    for topk in TOPKS:
        for beta in BETAS:
            t0 = time.time()
            model = EASE(reg=1000.0, min_rating=3.5, topk=topk, pop_beta=beta).fit(
                train.users, train.items, train.ratings, r.n_users, r.n_items, verbose=False)
            res = metrics.ranking(model, train, test)
            row = {"topk": topk or "none", "pop_beta": beta,
                   "ndcg@10": round(res["ndcg@10"], 4), "coverage": round(res["catalog_coverage"], 4),
                   "precision@10": round(res["precision@10"], 4), "recall@10": round(res["recall@10"], 4),
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(row, flush=True)
            del model

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ease_coverage.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["| EASE pruning | pop_beta | nDCG@10 | Catalog coverage | Precision@10 | Recall@10 | seconds |",
             "|---|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['topk']} | {x['pop_beta']} | {x['ndcg@10']:.4f} | {x['coverage']:.4f} | "
                     f"{x['precision@10']:.4f} | {x['recall@10']:.4f} | {x['seconds']} |")
    (RESULTS / "ease_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
