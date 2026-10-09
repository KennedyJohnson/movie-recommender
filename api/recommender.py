"""Serve recommendations for brand-new users from exported item factors.

A visitor isn't in the training data, so we "fold them in": hold the item factors fixed
and solve the model's own regularized least-squares problem for their user vector.
That's one small (k x k) linear solve per request, with no retraining.
"""
import json
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


NEUTRAL = 3.0  # star rating that neither pulls similar movies up nor pushes them down


class Recommender:
    def __init__(self, artifacts: Path = ARTIFACTS):
        m = np.load(artifacts / "model.npz")
        self.kind = str(m["kind"])
        self.Q, self.bi = m["Q"], m["bi"]
        self.mu, self.reg, self.alpha = float(m["mu"]), float(m["reg"]), float(m["alpha"])
        # older artifacts predate graded ALS confidence: "liked" = 3.5+ stars, all weighted equally
        self.min_rating = float(m["min_rating"]) if "min_rating" in m else 3.5
        self.graded = bool(m["graded"]) if "graded" in m else False
        self.movies = json.loads((artifacts / "movies.json").read_text(encoding="utf-8"))
        self.index = {mv["id"]: i for i, mv in enumerate(self.movies)}
        if self.kind == "ease":  # pruned item-item weights as CSR arrays: row i = what movie i votes for
            self.B_data, self.B_indices, self.B_indptr = m["B_data"], m["B_indices"], m["B_indptr"]
        self.QtQ = self.Q.T @ self.Q
        norms = np.linalg.norm(self.Q, axis=1, keepdims=True)
        self.Q_unit = self.Q / np.maximum(norms, 1e-8)

    def _user_vector(self, idx, ratings):
        Q = self.Q[idx]
        k = Q.shape[1]
        if self.kind == "als":
            # Hu et al. closed form; weights mirror recsys.models.als_weights
            lo = self.min_rating
            w = (ratings - lo + 1.0) / (6.0 - lo) if self.graded else np.ones(len(ratings))
            liked = ratings >= lo
            if not liked.any():
                return None, 0.0
            Ql, c = Q[liked], 1.0 + self.alpha * w[liked]
            A = self.QtQ + (Ql.T * (c - 1.0)) @ Ql + self.reg * np.eye(k)
            return np.linalg.solve(A, Ql.T @ c), 0.0

        # explicit: fit [user bias, user factors] to residuals r - mu - b_i
        target = ratings - self.mu - self.bi[idx]
        use_bias = self.kind == "svdf"
        X = np.hstack([np.ones((len(idx), 1)), Q]) if use_bias else Q
        lam = max(self.reg * len(idx), 1.0)  # stronger prior when the user has rated little
        w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ target)
        return (w[1:], w[0]) if use_bias else (w, 0.0)

    def recommend(self, rated: dict[int, float], n=20, genre=None):
        known = [(self.index[m], r) for m, r in rated.items() if m in self.index]
        if not known:
            return self.popular(n, genre)
        idx = np.array([i for i, _ in known])
        ratings = np.array([r for _, r in known], np.float32)
        if self.kind == "ease":
            if not (ratings >= self.min_rating).any():
                return []
            # weight each rated movie by (stars - 3)^3: 4/2 stars nudge similar movies (+/-1), 5/1 stars move them hard (+/-8)
            w = (ratings - NEUTRAL) ** 3
            used, uw = idx[w != 0], w[w != 0]
            scores, bu = self._ease_rows(used, uw), 0.0
        else:
            pu, bu = self._user_vector(idx, ratings)
            if pu is None:  # nothing liked yet, so there's no taste signal to rank by
                return []
            scores = self.Q @ pu + self.bi
        scores[idx] = -np.inf
        if genre:
            scores[[genre not in mv["genres"] for mv in self.movies]] = -np.inf
        top = np.argsort(-scores)[:n]
        because = self._because(used, uw, top) if self.kind == "ease" else {}
        out = []
        for i in top:
            if not np.isfinite(scores[i]):
                break
            item = dict(self.movies[i])
            # ALS/EASE scores are ranking scores, not star ratings, so only explicit models predict stars
            if self.kind not in ("als", "ease"):
                item["predicted"] = round(float(np.clip(self.mu + bu + scores[i], 0.5, 5.0)), 2)
            if i in because:
                item["because"] = [self.movies[r]["title"] for r in because[i]]
            out.append(item)
        return out

    def _ease_rows(self, rows, weights=None):
        """EASE score = weighted sum of the given movies' weight rows (x_u @ B)."""
        sl = [slice(self.B_indptr[i], self.B_indptr[i + 1]) for i in rows]
        cols = np.concatenate([self.B_indices[s] for s in sl])
        weights = np.ones(len(sl)) if weights is None else weights
        vals = np.concatenate([self.B_data[s] * w for s, w in zip(sl, weights)])
        return np.bincount(cols, weights=vals, minlength=len(self.movies))

    def _because(self, rows, weights, targets, k=2):
        """For each target, the k rated movies whose weighted rows pushed it up the most (positive pushes only)."""
        best = {}  # target -> [(push, rated movie row), ...] sorted high to low, at most k entries
        targets = set(int(t) for t in targets)
        for r, w in zip(rows, weights):
            s = slice(self.B_indptr[r], self.B_indptr[r + 1])
            for t, v in zip(self.B_indices[s], self.B_data[s] * w):
                t = int(t)
                if t in targets and v > 0:
                    lst = best.setdefault(t, [])
                    lst.append((float(v), int(r)))
                    lst.sort(reverse=True)
                    del lst[k:]
        return {t: [r for _, r in lst] for t, lst in best.items()}

    def similar(self, movie_id: int, n=12):
        i = self.index[movie_id]
        sims = self._ease_rows([i]) if self.kind == "ease" else self.Q_unit @ self.Q_unit[i]
        sims[i] = -np.inf
        return [dict(self.movies[j], similarity=round(float(sims[j]), 3))
                for j in np.argsort(-sims)[:n]]

    def popular(self, n=40, genre=None, offset=0):
        pool = [mv for mv in self.movies if not genre or genre in mv["genres"]]
        return pool[offset:offset + n]  # movies.json is sorted by rating count

    def search(self, q: str, n=10):
        q = q.lower().strip()
        hits = [mv for mv in self.movies if q in mv["title"].lower()]
        hits.sort(key=lambda mv: (not mv["title"].lower().startswith(q), -mv["n"]))
        return hits[:n]
