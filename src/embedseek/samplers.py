"""Samplers: fixed-representation baselines (proposal §3.2) and the two-level bandit (docs/07).

Each returns pool row indices to label.
"""
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


def multi_query(pool_embs: list[np.ndarray], query_embs: list[np.ndarray], budget: int,
                batch: int = 25) -> np.ndarray:
    """Query-seeded search over several encoders, with no feedback: a control for the bandit.

    Encoders take turns; each adds its top-`batch` not-yet-chosen rows by cosine to its own query.
    """
    sims = [emb @ q for emb, q in zip(pool_embs, query_embs)]
    free = np.ones(len(sims[0]), dtype=bool)
    chosen = []
    while len(chosen) < budget:
        for s in sims:
            n = min(batch, budget - len(chosen))
            pick = np.argsort(-np.where(free, s, -np.inf))[:n]
            free[pick] = False
            chosen.extend(pick.tolist())
    return np.array(chosen)


def _pick_from_cluster(pool_emb, cand, n, labeled, labels, rng) -> np.ndarray:
    """`n` of the candidate rows: model-predicted positives first (LR on the labels so far), the rest random."""
    y_so_far = np.array(labels)
    if y_so_far.size and 0 < y_so_far.sum() < y_so_far.size:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(pool_emb[labeled], y_so_far)
        scores = clf.decision_function(pool_emb[cand])
        predicted_pos = cand[scores > 0][np.argsort(-scores[scores > 0])][:n]
        rest = rng.choice(np.setdiff1d(cand, predicted_pos), n - len(predicted_pos), replace=False)
        return np.concatenate([predicted_pos, rest])
    return rng.choice(cand, n, replace=False)


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
        pick = _pick_from_cluster(pool_emb, cand, n, labeled, labels, rng)

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


def hierarchical_lts(pool_embs: list[np.ndarray], cluster_ids: list[np.ndarray],
                     oracle: Callable[[np.ndarray], np.ndarray], budget: int, seed: int,
                     batch: int = 25, decay: float = 0.99, share_labels: bool = True,
                     warm_start: list[np.ndarray] | None = None, warm_batch: int | None = None,
                     flat: bool = False, encoder_reward: str = "binary",
                     trace: list | None = None) -> np.ndarray:
    """Ours: two-level Thompson sampling, encoder -> cluster (docs/07, "Idea 1").

    `pool_embs[m]` and `cluster_ids[m]` are encoder m's embedding and k-means clusters of the same
    pool rows. Each round: draw from Beta(1 + wins, 1 + losses) per encoder and pick the argmax
    encoder m*, then run one `lts_lite` round inside m* (Thompson-sample a cluster, label `batch`
    rows from it, ranked by an LR on m*'s embedding). Reward = 1 if the batch contains a positive;
    it updates m* and the cluster. Wins/losses decay by `decay` each round at both levels.
    With one encoder and the defaults this is exactly `lts_lite`.

    share_labels: a labeled batch also updates the clusters its rows fall into in every OTHER
        encoder. A cluster holding a fraction f of the batch gets f of a win if any of its rows is
        positive, else f of a loss (for m*'s own cluster f = 1, the usual update).
    warm_start:   one query embedding per encoder. Before Thompson sampling starts, each encoder
        labels its top-`warm_batch` rows (default `batch`) by cosine to its query; these count
        against the budget and update the stats like any other round. With the default this is
        the first round of `multi_query`.
    flat:         ablation without the hierarchy: every (encoder, cluster) pair is one arm.
    encoder_reward: "binary" (default) gives the chosen encoder a win if the batch has a positive.
        "rate" gives it r = positives / batch as a fractional win and 1 - r as a loss, so an
        encoder whose batches are richer in positives pulls ahead even when every encoder finds
        some. Only the encoder level changes; clusters keep the binary reward. Ignored if `flat`.
    trace:        if a list, one (encoder, cluster, n_labeled, n_pos) tuple is appended per round
        (cluster = -1 for warm-start rounds), for arm-allocation analysis.
    """
    if encoder_reward not in ("binary", "rate"):
        raise ValueError(f"encoder_reward must be 'binary' or 'rate', got {encoder_reward!r}")
    rng = np.random.default_rng(seed)
    n_enc = len(pool_embs)
    ks = [int(c.max()) + 1 for c in cluster_ids]
    wins, losses = [np.zeros(k) for k in ks], [np.zeros(k) for k in ks]
    enc_wins, enc_losses = np.zeros(n_enc), np.zeros(n_enc)
    unlabeled = np.ones(len(pool_embs[0]), dtype=bool)
    labeled, labels = [], []

    def label_and_update(m, arm, pick):
        y = oracle(pick)
        labeled.extend(pick.tolist())
        labels.extend(y.tolist())
        unlabeled[pick] = False
        if trace is not None:
            trace.append((m, arm, len(pick), int(y.sum())))

        for stats in (*wins, *losses, enc_wins, enc_losses):
            stats *= decay
        if encoder_reward == "rate":
            enc_wins[m] += y.mean()
            enc_losses[m] += 1 - y.mean()
        else:
            (enc_wins if y.sum() > 0 else enc_losses)[m] += 1
        for e in range(n_enc) if share_labels else [m]:
            c = cluster_ids[e][pick]
            share = np.bincount(c, minlength=ks[e]) / len(pick)          # fraction of the batch per cluster
            hit = np.bincount(c, weights=y, minlength=ks[e]) > 0
            wins[e] += np.where(hit, share, 0)
            losses[e] += np.where(hit, 0, share)

    if warm_start is not None:
        for m, query_emb in enumerate(warm_start):
            n = min(warm_batch or batch, budget - len(labeled))
            if n == 0:
                break
            sims = np.where(unlabeled, pool_embs[m] @ query_emb, -np.inf)
            label_and_update(m, -1, np.argsort(-sims)[:n])

    while len(labeled) < budget:
        open_clusters = [np.unique(cluster_ids[e][unlabeled]) for e in range(n_enc)]
        if flat:
            draws = [rng.beta(1 + wins[e][open_clusters[e]], 1 + losses[e][open_clusters[e]]) for e in range(n_enc)]
            m = int(np.argmax([d.max() for d in draws]))
            arm = open_clusters[m][np.argmax(draws[m])]
        else:
            m = int(np.argmax(rng.beta(1 + enc_wins, 1 + enc_losses))) if n_enc > 1 else 0
            arm = open_clusters[m][np.argmax(rng.beta(1 + wins[m][open_clusters[m]], 1 + losses[m][open_clusters[m]]))]

        cand = np.flatnonzero(unlabeled & (cluster_ids[m] == arm))
        n = min(batch, budget - len(labeled), len(cand))
        label_and_update(m, arm, _pick_from_cluster(pool_embs[m], cand, n, labeled, labels, rng))

    return np.array(labeled)
