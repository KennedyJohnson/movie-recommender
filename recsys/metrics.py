import numpy as np
import scipy.sparse as sp


def rmse(pred, actual):
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(pred, actual):
    return float(np.mean(np.abs(pred - actual)))


def ranking(model, train, test, k=10, n_users=20_000, relevant=4.0, batch=1_000, seed=4093):
    """Precision/recall/nDCG@k over sampled users: rank every unseen item, and count
    a hit when it's in the user's held-out ratings at or above `relevant` stars."""
    shape = (train.n_users, train.n_items)
    seen = sp.csr_matrix((np.ones(len(train.users), bool), (train.users, train.items)), shape=shape)
    rel_mask = test.ratings >= relevant
    liked = sp.csr_matrix((np.ones(rel_mask.sum(), bool),
                           (test.users[rel_mask], test.items[rel_mask])), shape=shape)

    eligible = np.flatnonzero(np.diff(liked.indptr) > 0)
    users = np.random.default_rng(seed).choice(eligible, min(n_users, len(eligible)), replace=False)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    prec, rec, ndcg, covered = [], [], [], set()

    for start in range(0, len(users), batch):
        ub = users[start:start + batch]
        scores = np.asarray(model.item_scores(ub), dtype=np.float32)
        s = seen[ub]
        scores[np.repeat(np.arange(len(ub)), np.diff(s.indptr)), s.indices] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.take_along_axis(scores, top, 1).argsort(1)[:, ::-1]
        top = np.take_along_axis(top, order, 1)
        covered.update(top.ravel().tolist())

        lb = liked[ub]
        hits = np.take_along_axis(lb.toarray(), top, 1)
        n_liked = np.diff(lb.indptr)
        prec.append(hits.sum(1) / k)
        rec.append(hits.sum(1) / n_liked)
        ideal = np.cumsum(discounts)[np.minimum(n_liked, k) - 1]
        ndcg.append((hits * discounts).sum(1) / ideal)

    return {f"precision@{k}": float(np.concatenate(prec).mean()),
            f"recall@{k}": float(np.concatenate(rec).mean()),
            f"ndcg@{k}": float(np.concatenate(ndcg).mean()),
            "catalog_coverage": len(covered) / train.n_items}
