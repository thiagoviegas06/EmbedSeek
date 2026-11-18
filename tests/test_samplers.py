"""Samplers: right size, no duplicates, correct ranking, and no peeking at labels."""
import numpy as np
import pytest

from embedseek.samplers import hierarchical_lts, lts_lite, multi_query, query_seeded, random_sample


def unit_rows(n, d, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_random_sample_size_unique_and_seeded():
    a = random_sample(10_000, 500, seed=3)
    assert len(a) == 500 and len(np.unique(a)) == 500
    assert np.array_equal(a, random_sample(10_000, 500, seed=3))      # reproducible
    assert not np.array_equal(a, random_sample(10_000, 500, seed=4))  # seed matters


def test_random_sample_positive_rate_matches_prevalence():
    # 0.3% prevalence, B=1000, 500 seeds: mean positives should be ~3 (binomial sd of the mean ~0.08).
    y = np.zeros(100_000, dtype=int)
    y[:300] = 1
    found = [y[random_sample(len(y), 1000, s)].sum() for s in range(500)]
    assert np.mean(found) == pytest.approx(3.0, abs=0.3)


def test_query_seeded_returns_true_top_b_in_order():
    pool, q = unit_rows(5_000, 32), unit_rows(1, 32, seed=1)[0]
    got = query_seeded(pool, q, 100)
    brute = np.argsort(-(pool @ q))[:100]
    assert np.array_equal(got, brute)


class SpyOracle:
    """Returns labels only for requested indices and records every request."""

    def __init__(self, y):
        self.y, self.requested = y, []

    def __call__(self, idx):
        self.requested.extend(np.asarray(idx).tolist())
        return self.y[idx]


@pytest.mark.parametrize("budget", [10, 100, 333])
def test_lts_lite_budget_uniqueness_and_no_peeking(budget):
    rng = np.random.default_rng(0)
    pool = unit_rows(3_000, 16)
    clusters = rng.integers(0, 20, size=len(pool))
    y = (rng.random(len(pool)) < 0.05).astype(int)
    oracle = SpyOracle(y)

    idx = lts_lite(pool, clusters, oracle, budget, seed=0)

    assert len(idx) == budget and len(np.unique(idx)) == budget
    # Every label the sampler saw was one it requested, each exactly once, and nothing else.
    assert sorted(oracle.requested) == sorted(idx.tolist())


def test_lts_lite_exploits_a_positive_cluster():
    # All positives live in cluster 0; a working bandit must find far more than random would.
    rng = np.random.default_rng(1)
    pool = unit_rows(10_000, 16)
    clusters = rng.integers(0, 20, size=len(pool))
    y = np.zeros(len(pool), dtype=int)
    in0 = np.flatnonzero(clusters == 0)
    y[in0[: len(in0) // 2]] = 1  # ~250 positives, 2.5% overall
    found = np.mean([y[lts_lite(pool, clusters, lambda i: y[i], 400, seed=s)].sum() for s in range(5)])
    assert found > 3 * 0.025 * 400  # random would find ~10


# ---- Two-level bandit ---------------------------------------------------------------------------

def two_encoders(n, seed=0):
    """Two unrelated embeddings and clusterings of the same n rows."""
    rng = np.random.default_rng(seed)
    return [unit_rows(n, 16, seed=1), unit_rows(n, 24, seed=2)], [rng.integers(0, 20, n), rng.integers(0, 20, n)]


@pytest.mark.parametrize("budget", [10, 100, 333])
@pytest.mark.parametrize("opts", [{}, {"share_labels": False}, {"flat": True}, {"warm": True},
                                  {"flat": True, "warm": True, "share_labels": False},
                                  {"encoder_reward": "rate"}, {"encoder_reward": "rate", "warm": True}])
def test_hierarchical_lts_budget_uniqueness_and_no_peeking(budget, opts):
    embs, clusters = two_encoders(3_000)
    y = (np.random.default_rng(0).random(3_000) < 0.05).astype(int)
    oracle = SpyOracle(y)
    opts = dict(opts)
    warm = [e[0] for e in embs] if opts.pop("warm", False) else None
    trace = []

    idx = hierarchical_lts(embs, clusters, oracle, budget, seed=0, warm_start=warm, trace=trace, **opts)

    assert len(idx) == budget and len(np.unique(idx)) == budget
    assert sorted(oracle.requested) == sorted(idx.tolist())
    assert sum(t[2] for t in trace) == budget and sum(t[3] for t in trace) == y[idx].sum()


def test_hierarchical_lts_with_one_encoder_is_lts_lite():
    rng = np.random.default_rng(0)
    pool, clusters = unit_rows(3_000, 16), rng.integers(0, 20, 3_000)
    y = (rng.random(3_000) < 0.05).astype(int)
    for seed in range(3):
        assert np.array_equal(hierarchical_lts([pool], [clusters], lambda i: y[i], 300, seed),
                              lts_lite(pool, clusters, lambda i: y[i], 300, seed))


def test_hierarchical_lts_warm_start_labels_each_encoders_query_neighbours_first():
    embs, clusters = two_encoders(3_000)
    y = np.zeros(3_000, dtype=int)
    queries = [embs[0][7], embs[1][11]]
    idx = hierarchical_lts(embs, clusters, lambda i: y[i], 100, seed=0, warm_start=queries)
    assert set(idx[:25]) == set(np.argsort(-(embs[0] @ queries[0]))[:25])
    sims = embs[1] @ queries[1]
    sims[idx[:25]] = -np.inf                                   # rows encoder 0 already took
    assert set(idx[25:50]) == set(np.argsort(-sims)[:25])


@pytest.mark.parametrize("flat", [False, True])
def test_hierarchical_lts_exploits_a_cluster_planted_in_one_encoder(flat):
    # Positives all sit in cluster 0 of encoder 1; encoder 0's clusters are unrelated to the label.
    # 20 arms and 30 rounds, so there is time to explore: with many more arms than rounds it cannot.
    n, budget = 40_000, 750
    rng = np.random.default_rng(1)
    embs = [unit_rows(n, 16, seed=1), unit_rows(n, 16, seed=2)]
    clusters = [rng.integers(0, 10, n), rng.integers(0, 10, n)]
    y = np.zeros(n, dtype=int)
    y[np.flatnonzero(clusters[1] == 0)[:400]] = 1              # 1% overall, ~10% of that cluster
    found, rounds = [], np.zeros(2)
    for seed in range(5):
        trace = []
        idx = hierarchical_lts(embs, clusters, lambda i: y[i], budget, seed, share_labels=False, flat=flat,
                               trace=trace)
        found.append(y[idx].sum())
        rounds += np.bincount([t[0] for t in trace], minlength=2)
    assert np.mean(found) > 3 * 0.01 * budget                  # random would find ~7.5
    assert rounds[1] > rounds[0]                               # and it learned which encoder to use


def test_hierarchical_lts_rate_reward_fixes_binary_preferring_the_poorer_encoder():
    # Encoder 0's clusters ignore the label (15% positives everywhere), so its batches almost always
    # hold a positive. Encoder 1's clusters 0-4 are 30% positive and 5-9 hold none, so exploring them
    # costs it binary losses: the binary reward ends up preferring encoder 0 (~12% of rounds to
    # encoder 1). Counting positives per batch must shift rounds to encoder 1 and find more positives.
    n, budget = 20_000, 1_000
    rng = np.random.default_rng(3)
    embs = [unit_rows(n, 16, seed=1), unit_rows(n, 16, seed=2)]
    clusters = [rng.integers(0, 10, n), rng.integers(0, 10, n)]
    y = ((clusters[1] < 5) & (rng.random(n) < 0.3)).astype(int)
    share, found = {}, {}
    for reward in ("binary", "rate"):
        rounds, pos = np.zeros(2), []
        for seed in range(5):
            trace = []
            idx = hierarchical_lts(embs, clusters, lambda i: y[i], budget, seed, share_labels=False,
                                   encoder_reward=reward, trace=trace)
            rounds += np.bincount([t[0] for t in trace], minlength=2)
            pos.append(y[idx].sum())
        share[reward], found[reward] = rounds[1] / rounds.sum(), np.mean(pos)
    assert share["binary"] < 0.3
    assert share["rate"] > share["binary"] + 0.25
    assert found["rate"] > 1.15 * found["binary"]


def test_hierarchical_lts_rejects_an_unknown_encoder_reward():
    embs, clusters = two_encoders(500)
    with pytest.raises(ValueError):
        hierarchical_lts(embs, clusters, lambda i: np.zeros(len(i), dtype=int), 50, 0, encoder_reward="count")


def test_multi_query_takes_turns_and_matches_the_bandits_warm_start():
    embs, clusters = two_encoders(3_000)
    queries = [embs[0][7], embs[1][11]]
    idx = multi_query(embs, queries, 130)
    assert len(idx) == 130 and len(np.unique(idx)) == 130
    assert set(idx[:25]) == set(np.argsort(-(embs[0] @ queries[0]))[:25])
    y = np.zeros(3_000, dtype=int)
    warm = hierarchical_lts(embs, clusters, lambda i: y[i], 100, seed=0, warm_start=queries)
    assert np.array_equal(warm[:50], idx[:50])
    # One encoder is plain query-seeded search.
    assert np.array_equal(multi_query(embs[:1], queries[:1], 100), query_seeded(embs[0], queries[0], 100))


def test_hierarchical_lts_warm_batch_sets_the_number_of_query_rows():
    embs, clusters = two_encoders(3_000)
    y = np.zeros(3_000, dtype=int)
    idx = hierarchical_lts(embs[:1], clusters[:1], lambda i: y[i], 100, seed=0, warm_start=[embs[0][7]],
                           warm_batch=75)
    assert np.array_equal(idx[:75], query_seeded(embs[0], embs[0][7], 75))
