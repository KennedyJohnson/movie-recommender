"""Convert api/artifacts (EASE) into static files so the site runs on GitHub Pages with no server.

EASE scores = sum of the liked movies' weight rows, so the browser only needs the rows for the
movies a visitor rated. Rows are split into shards of SHARD rows; each shard is one small binary:
  int32 [nrows, nnz], int32 indptr[nrows+1], int32 indices[nnz], float32 data[nnz]

    python scripts/export_static.py
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART, OUT = ROOT / "api" / "artifacts", ROOT / "docs" / "data"
SHARD = 100

m = np.load(ART / "model.npz")
if str(m["kind"]) != "ease":
    raise SystemExit("static export supports the EASE model only (export with: python scripts/export_model.py ease)")
indptr, indices, data = m["B_indptr"], m["B_indices"], m["B_data"]
movies = json.loads((ART / "movies.json").read_text(encoding="utf-8"))

shutil.rmtree(OUT / "ease", ignore_errors=True)
(OUT / "ease").mkdir(parents=True)
for s, start in enumerate(range(0, len(movies), SHARD)):
    end = min(start + SHARD, len(movies))
    lo, hi = indptr[start], indptr[end]
    with open(OUT / "ease" / f"{s}.bin", "wb") as f:
        np.array([end - start, hi - lo], np.int32).tofile(f)
        (indptr[start:end + 1] - lo).astype(np.int32).tofile(f)
        indices[lo:hi].astype(np.int32).tofile(f)
        data[lo:hi].astype(np.float32).tofile(f)

(OUT / "movies.json").write_text(json.dumps(movies, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
(OUT / "meta.json").write_text(json.dumps({"shard": SHARD, "min_rating": float(m["min_rating"]),
                                           "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}))
print(f"{len(movies)} movies, {s + 1} shards -> {OUT}")
