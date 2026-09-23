"""Matrix factorization recommenders, implemented with numba.

- FunkSVD ("SVDF" in recommenderlab): biased MF trained with SGD on explicit ratings.
- PMF: probabilistic MF, i.e. unbiased MF with Gaussian priors (L2), trained with SGD
  on mean-centered ratings.
- ImplicitALS: Hu, Koren & Volinsky (2008) weighted MF on implicit interactions,
  solved with conjugate-gradient ALS (Takacs et al. 2011).
- Popularity: most-rated items, the baseline to beat.

Every model exposes `item_scores(user_idx)` -> (n_users_batch, n_items) for ranking.
"""
import numpy as np
import scipy.sparse as sp
from numba import njit, prange


# ---------------------------------------------------------------- explicit (SGD)

@njit(fastmath=True, cache=True)
def _sgd_epoch(users, items, ratings, order, mu, bu, bi, P, Q, lr, reg, reg_b, use_bias):
    k = P.shape[1]
    sq_err = 0.0
    for idx in order:
        u = users[idx]
        i = items[idx]
        pred = mu
        if use_bias:
            pred += bu[u] + bi[i]
        for f in range(k):
            pred += P[u, f] * Q[i, f]
        e = ratings[idx] - pred
        sq_err += e * e
        if use_bias:
            bu[u] += lr * (e - reg_b * bu[u])
            bi[i] += lr * (e - reg_b * bi[i])
        for f in range(k):
            pu = P[u, f]
            qi = Q[i, f]
            P[u, f] += lr * (e * qi - reg * pu)
            Q[i, f] += lr * (e * pu - reg * qi)
    return np.sqrt(sq_err / len(order))


class _SGDFactorizer:
    use_bias = True

    def __init__(self, factors=64, epochs=20, lr=0.005, reg=0.02, reg_bias=0.02, seed=4093):
        self.factors, self.epochs, self.lr = factors, epochs, lr
        self.reg, self.reg_bias, self.seed = reg, reg_bias, seed

    def fit(self, users, items, ratings, n_users, n_items, verbose=True):
        rng = np.random.default_rng(self.seed)
        ratings = ratings.astype(np.float32)
        self.mu = float(ratings.mean())
        self.bu = np.zeros(n_users, np.float32)
        self.bi = np.zeros(n_items, np.float32)
        self.P = rng.normal(0, 0.1, (n_users, self.factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, self.factors)).astype(np.float32)
        for epoch in range(self.epochs):
            order = rng.permutation(len(ratings))
            rmse = _sgd_epoch(users, items, ratings, order, self.mu, self.bu, self.bi,
                              self.P, self.Q, self.lr, self.reg, self.reg_bias, self.use_bias)
            if verbose:
                print(f"    epoch {epoch + 1:2d}/{self.epochs}  train RMSE {rmse:.4f}")
        return self

    def predict(self, users, items):
        pred = self.mu + np.einsum("ij,ij->i", self.P[users], self.Q[items])
        if self.use_bias:
            pred += self.bu[users] + self.bi[items]
        return np.clip(pred, 0.5, 5.0)

    def item_scores(self, users):
        scores = self.P[users] @ self.Q.T
        if self.use_bias:
            scores += self.bi
        return scores


class FunkSVD(_SGDFactorizer):
    use_bias = True


class PMF(_SGDFactorizer):
    use_bias = False


# ---------------------------------------------------------------- implicit (ALS)

@njit(parallel=True, fastmath=True, cache=True)
def _cg_als_step(indptr, indices, conf, X, Y, YtY, cg_steps):
    """Update every row of X given fixed Y, with confidence-weighted CG."""
    n, k = X.shape
    for u in prange(n):
        x = X[u].copy()
        r = -YtY @ x
        for j in range(indptr[u], indptr[u + 1]):
            y = Y[indices[j]]
            c = conf[j]
            r += (c - (c - 1.0) * np.dot(y, x)) * y
        p = r.copy()
        rs_old = np.dot(r, r)
        if rs_old < 1e-20:
            continue
        for _ in range(cg_steps):
            Ap = YtY @ p
            for j in range(indptr[u], indptr[u + 1]):
                y = Y[indices[j]]
                Ap += (conf[j] - 1.0) * np.dot(y, p) * y
            a = rs_old / np.dot(p, Ap)
            x += a * p
            r -= a * Ap
            rs_new = np.dot(r, r)
            if rs_new < 1e-20:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        X[u] = x


class ImplicitALS:
    """Treats every rating as an interaction ("watched"), ignoring its value."""

    def __init__(self, factors=64, iterations=15, reg=0.1, alpha=10.0, cg_steps=3, seed=4093):
        self.factors, self.iterations, self.reg = factors, iterations, reg
        self.alpha, self.cg_steps, self.seed = alpha, cg_steps, seed

    def fit(self, users, items, ratings, n_users, n_items, verbose=True):
        rng = np.random.default_rng(self.seed)
        conf = np.full(len(users), 1.0 + self.alpha, np.float32)
        ui = sp.csr_matrix((conf, (users, items)), shape=(n_users, n_items), dtype=np.float32)
        iu = ui.T.tocsr()
        self.P = rng.normal(0, 0.01, (n_users, self.factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.01, (n_items, self.factors)).astype(np.float32)
        eye = self.reg * np.eye(self.factors, dtype=np.float32)
        for it in range(self.iterations):
            _cg_als_step(ui.indptr, ui.indices, ui.data, self.P, self.Q,
                         self.Q.T @ self.Q + eye, self.cg_steps)
            _cg_als_step(iu.indptr, iu.indices, iu.data, self.Q, self.P,
                         self.P.T @ self.P + eye, self.cg_steps)
            if verbose:
                print(f"    iteration {it + 1:2d}/{self.iterations}")
        return self

    def item_scores(self, users):
        return self.P[users] @ self.Q.T


class Popularity:
    def fit(self, users, items, ratings, n_users, n_items, verbose=True):
        self.counts = np.bincount(items, minlength=n_items).astype(np.float32)
        return self

    def item_scores(self, users):
        return np.broadcast_to(self.counts, (len(users), len(self.counts))).copy()


class BiasBaseline(FunkSVD):
    """Global mean + user bias + item bias: the RMSE floor any MF should beat."""

    def __init__(self, epochs=10, lr=0.005, reg_bias=0.02, seed=4093):
        super().__init__(factors=0, epochs=epochs, lr=lr, reg_bias=reg_bias, seed=seed)
