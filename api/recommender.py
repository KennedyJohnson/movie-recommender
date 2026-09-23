"""Serve recommendations for brand-new users from exported item factors.

A visitor isn't in the training data, so we "fold them in": hold the item factors fixed
and solve the model's own regularized least-squares problem for their user vector.
That's one small (k x k) linear solve per request, with no retraining.
"""
import json
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


class Recommender:
    def __init__(self, artifacts: Path = ARTIFACTS):
        m = np.load(artifacts / "model.npz")
        self.kind = str(m["kind"])
        self.Q, self.bi = m["Q"], m["bi"]
        self.mu, self.reg, self.alpha = float(m["mu"]), float(m["reg"]), float(m["alpha"])
        self.movies = json.loads((artifacts / "movies.json").read_text(encoding="utf-8"))
        self.index = {mv["id"]: i for i, mv in enumerate(self.movies)}
        self.QtQ = self.Q.T @ self.Q
        norms = np.linalg.norm(self.Q, axis=1, keepdims=True)
        self.Q_unit = self.Q / np.maximum(norms, 1e-8)

    def _user_vector(self, idx, ratings):
        Q = self.Q[idx]
        k = Q.shape[1]
        if self.kind == "als":
            # Hu et al. closed form with only this user's liked movies as positives
            liked = ratings >= 3.5
            if not liked.any():
                return None, 0.0
            Ql = Q[liked]
            A = self.QtQ + self.alpha * Ql.T @ Ql + self.reg * np.eye(k)
            return np.linalg.solve(A, (1 + self.alpha) * Ql.sum(0)), 0.0

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
        pu, bu = self._user_vector(idx, ratings)
        if pu is None:  # nothing liked yet, so there's no taste signal to rank by
            return []

        scores = self.Q @ pu + self.bi
        scores[idx] = -np.inf
        if genre:
            scores[[genre not in mv["genres"] for mv in self.movies]] = -np.inf
        top = np.argsort(-scores)[:n]
        out = []
        for i in top:
            if not np.isfinite(scores[i]):
                break
            item = dict(self.movies[i])
            if self.kind != "als":
                item["predicted"] = round(float(np.clip(self.mu + bu + scores[i], 0.5, 5.0)), 2)
            out.append(item)
        return out

    def similar(self, movie_id: int, n=12):
        i = self.index[movie_id]
        sims = self.Q_unit @ self.Q_unit[i]
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
