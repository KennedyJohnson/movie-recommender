"""Matrix factorization recommenders, implemented with numba.

- FunkSVD ("SVDF" in recommenderlab): biased MF trained with SGD on explicit ratings.
- PMF: probabilistic MF, i.e. unbiased MF with Gaussian priors (L2), trained with SGD
  on mean-centered ratings.
- ImplicitALS: Hu, Koren & Volinsky (2008) weighted MF on implicit interactions,
  solved with conjugate-gradient ALS (Takacs et al. 2011).
- EASE: Steck (2019) closed-form item-item model; BPR: Rendle et al. (2009) pairwise ranking MF.
- Popularity: most-rated items, the baseline to beat.

Every model exposes `item_scores(user_idx)` -> (n_users_batch, n_items) for ranking.
"""
import numpy as np
import scipy.sparse as sp
import numba
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


def als_weights(ratings, min_rating=None, graded=False):
    """Per-rating confidence weight in [0, 1]. Ratings below `min_rating` get 0 (dropped,
    i.e. treated like unwatched); with `graded`, weight rises linearly up to 1 at 5 stars."""
    ratings = np.asarray(ratings, np.float32)
    lo = 0.5 if min_rating is None else min_rating
    w = (ratings - lo + 1.0) / (6.0 - lo) if graded else np.ones_like(ratings)
    return np.where(ratings >= lo, w, 0.0).astype(np.float32)


class ImplicitALS:
    """Hu et al. implicit ALS. By default every rating is an interaction ("watched"),
    ignoring its value; `min_rating`/`graded` fold the star value into the confidence."""

    def __init__(self, factors=64, iterations=15, reg=0.1, alpha=10.0, cg_steps=3,
                 min_rating=None, graded=False, pop_beta=0.0, seed=4093):
        self.factors, self.iterations, self.reg = factors, iterations, reg
        self.alpha, self.cg_steps, self.seed = alpha, cg_steps, seed
        self.min_rating, self.graded, self.pop_beta = min_rating, graded, pop_beta

    def fit(self, users, items, ratings, n_users, n_items, verbose=True):
        rng = np.random.default_rng(self.seed)
        w = als_weights(ratings, self.min_rating, self.graded)
        keep = w > 0
        users, items, w = users[keep], items[keep], w[keep]
        if self.pop_beta:  # down-weight blockbusters so they don't dominate the factors
            counts = np.bincount(items, minlength=n_items).astype(np.float32)
            w = w * (counts[items] / counts[counts > 0].mean()) ** -self.pop_beta
        conf = (1.0 + self.alpha * w).astype(np.float32)
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


def prune_topk(B, k, chunk=1024):
    """Keep the k largest-magnitude weights in each column of B (the strongest predictors
    of each target movie) as a sparse matrix, shrinking EASE from items^2 to items*k."""
    n = B.shape[0]
    rows, cols, vals = [], [], []
    for start in range(0, B.shape[1], chunk):
        block = B[:, start:start + chunk]
        top = np.argpartition(-np.abs(block), min(k, n - 1), axis=0)[:k]
        c = np.broadcast_to(np.arange(block.shape[1]), top.shape)
        rows.append(top.ravel()); cols.append((c + start).ravel()); vals.append(block[top, c].ravel())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=B.shape, dtype=np.float32)


class EASE:
    """Steck 2019: closed-form item-item linear autoencoder, B = I - P/diag(P) with
    P = (X'X + reg I)^-1. The dense item Gram matrix is items^2, so only the
    `max_items` most-interacted movies are modelled; the rest score -inf."""

    def __init__(self, reg=500.0, max_items=12_000, min_rating=None, topk=None):
        self.reg, self.max_items, self.min_rating, self.topk = reg, max_items, min_rating, topk

    def fit(self, users, items, ratings, n_users, n_items, verbose=True):
        if self.min_rating is not None:
            keep = ratings >= self.min_rating
            users, items = users[keep], items[keep]
        counts = np.bincount(items, minlength=n_items)
        self.cols = np.argsort(-counts)[:min(self.max_items, n_items)]
        remap = np.full(n_items, -1, np.int64)
        remap[self.cols] = np.arange(len(self.cols))
        m = remap[items] >= 0
        X = sp.csr_matrix((np.ones(m.sum(), np.float32), (users[m], remap[items[m]])),
                          shape=(n_users, len(self.cols)))
        if verbose:
            print(f"    Gram + inverse over {len(self.cols):,} items")
        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        B = P / -np.diag(P)
        B[np.diag_indices_from(B)] = 0.0
        B = B.astype(np.float32)
        self.B = prune_topk(B, self.topk) if self.topk else B
        self.X, self.n_items = X, n_items
        return self

    def item_scores(self, users):
        out = np.full((len(users), self.n_items), -np.inf, np.float32)
        s = self.X[users] @ self.B
        out[:, self.cols] = s.toarray() if sp.issparse(s) else s
        return out


@njit(parallel=True, fastmath=True)
def _bpr_epoch(indptr, indices, P, Q, lr, reg, n_samples, seed):
    """Hogwild BPR-SGD: sample (u, liked i, random j), push x_ui above x_uj."""
    n_users, n_items = P.shape[0], Q.shape[0]
    n_threads = numba.get_num_threads()
    per = n_samples // n_threads
    for t in prange(n_threads):
        np.random.seed(seed + t)
        for _ in range(per):
            u = np.random.randint(n_users)
            lo, hi = indptr[u], indptr[u + 1]
            if hi == lo:
                continue
            i = indices[np.random.randint(lo, hi)]
            j = np.random.randint(n_items)
            pu, qi, qj = P[u].copy(), Q[i], Q[j]
            x = np.dot(pu, qi - qj)
            g = 1.0 / (1.0 + np.exp(x))
            P[u] += lr * (g * (qi - qj) - reg * pu)
            Q[i] += lr * (g * pu - reg * qi)
            Q[j] += lr * (-g * pu - reg * qj)


class BPR:
    """Rendle et al. 2009 Bayesian Personalized Ranking with uniform negative sampling."""

    def __init__(self, factors=64, epochs=30, lr=0.05, reg=0.001, min_rating=None, seed=4093):
        self.factors, self.epochs, self.lr, self.reg = factors, epochs, lr, reg
        self.min_rating, self.seed = min_rating, seed

    def fit(self, users, items, ratings, n_users, n_items, verbose=True):
        if self.min_rating is not None:
            keep = ratings >= self.min_rating
            users, items = users[keep], items[keep]
        ui = sp.csr_matrix((np.ones(len(users), np.float32), (users, items)), shape=(n_users, n_items))
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0, 0.1, (n_users, self.factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, self.factors)).astype(np.float32)
        for ep in range(self.epochs):
            _bpr_epoch(ui.indptr, ui.indices, self.P, self.Q, np.float32(self.lr),
                       np.float32(self.reg), len(users), self.seed + 1000 * ep)
            if verbose:
                print(f"    epoch {ep + 1:2d}/{self.epochs}")
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
