"""Fixed-representation samplers (proposal §3.2). Each returns pool row indices to label."""
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression


def random_sample(n_pool: int, budget: int, seed: int) -> np.ndarray:
    """(a) Uniform random sampling: the primary baseline."""
    return np.random.default_rng(seed).choice(n_pool, size=budget, replace=False)


def query_seeded(pool_emb: np.ndarray, query_emb: np.ndarray, budget: int) -> np.ndarray:
    """(b) Top-`budget` pool items by cosine similarity to the predicate embedding.

    Embeddings are L2-normalized, so the dot product is the cosine similarity.
    """
    sims = pool_emb @ query_emb
    top = np.argpartition(-sims, budget)[:budget]
    return top[np.argsort(-sims[top])]


def lts_lite(pool_emb: np.ndarray, cluster_ids: np.ndarray, oracle: Callable[[np.ndarray], np.ndarray],
             budget: int, seed: int, batch: int = 25, decay: float = 0.99) -> np.ndarray:
    """Competing baseline: simplified LTS (Barbosa et al. 2025, Alg. 1).

    Thompson sampling over k-means clusters of ONE embedding. Each round: draw from
    Beta(1 + wins, 1 + losses) per non-exhausted cluster, pick the argmax cluster, and label
    `batch` items from it - model-predicted positives first (LR on labels so far), the rest
    random. Reward = 1 if the batch contains a positive. Wins/losses decay by `decay` each round.

    Deviation from the paper: the reward is "found a positive" rather than "validation F1
    improved" (no gold validation set, no per-round fine-tuning), matching EmbedSeek's reward
    so the two bandits differ only in their arms (clusters vs embedding spaces).
    `oracle(idx)` returns labels for idx; labels are never read before being requested.
    """
    rng = np.random.default_rng(seed)
    k = int(cluster_ids.max()) + 1
    wins, losses = np.zeros(k), np.zeros(k)
    unlabeled = np.ones(len(pool_emb), dtype=bool)
    labeled, labels = [], []

    while len(labeled) < budget:
        open_clusters = np.unique(cluster_ids[unlabeled])
        draws = rng.beta(1 + wins[open_clusters], 1 + losses[open_clusters])
        arm = open_clusters[np.argmax(draws)]

        cand = np.flatnonzero(unlabeled & (cluster_ids == arm))
        n = min(batch, budget - len(labeled), len(cand))
        y_so_far = np.array(labels)
        if y_so_far.size and 0 < y_so_far.sum() < y_so_far.size:
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(pool_emb[labeled], y_so_far)
            scores = clf.decision_function(pool_emb[cand])
            predicted_pos = cand[scores > 0][np.argsort(-scores[scores > 0])][:n]
            rest = rng.choice(np.setdiff1d(cand, predicted_pos), n - len(predicted_pos), replace=False)
            pick = np.concatenate([predicted_pos, rest])
        else:
            pick = rng.choice(cand, n, replace=False)

        y = oracle(pick)
        labeled.extend(pick.tolist())
        labels.extend(y.tolist())
        unlabeled[pick] = False

        wins *= decay
        losses *= decay
        if y.sum() > 0:
            wins[arm] += 1
        else:
            losses[arm] += 1

    return np.array(labeled)
