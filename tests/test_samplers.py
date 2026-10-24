"""Samplers: right size, no duplicates, correct ranking, and no peeking at labels."""
import numpy as np
import pytest

from embedseek.samplers import lts_lite, query_seeded, random_sample


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
