"""Proxy: failed samples handled, learns when it should, threshold never uses test labels; bootstrap CIs."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from embedseek.proxy import (_Scorer, bootstrap_ci, bootstrap_samples, mean_diff_ci, paired_bootstrap,
                             train_and_evaluate)


def blobs(n, pos_frac, sep, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_frac).astype(int)
    X = rng.normal(size=(n, 8)) + sep * y[:, None]
    return X, y


def test_no_positives_is_recorded_as_failed_run():
    X, _ = blobs(200, 0.1, 3, 0)
    Xv, yv = blobs(500, 0.1, 3, 1)
    res = train_and_evaluate(X, np.zeros(200, int), Xv, yv, Xv, yv)
    assert not res.trained and res.f1 == 0.0 and res.n_pos == 0


def test_separable_data_gives_high_f1():
    X, y = blobs(500, 0.2, 4, 0)
    Xv, yv = blobs(2000, 0.02, 4, 1)
    Xt, yt = blobs(2000, 0.02, 4, 2)
    res = train_and_evaluate(X, y, Xv, yv, Xt, yt)
    assert res.trained and res.f1 > 0.9


def test_threshold_depends_on_val_not_test_labels():
    X, y = blobs(500, 0.2, 1.5, 0)
    Xv, yv = blobs(2000, 0.05, 1.5, 1)
    Xt, yt = blobs(2000, 0.05, 1.5, 2)
    a = train_and_evaluate(X, y, Xv, yv, Xt, yt)
    b = train_and_evaluate(X, y, Xv, yv, Xt, np.random.default_rng(9).permutation(yt))
    assert a.threshold == b.threshold   # shuffling test labels must not move the threshold
    assert a.f1 != b.f1                 # ...but must change the (honest) test score


@pytest.fixture(scope="module")
def fitted():
    """A rare-positive proxy (~2% test prevalence) with imperfect separation, like threat."""
    X, y = blobs(600, 0.2, 1.2, 0)
    Xv, yv = blobs(3000, 0.02, 1.2, 1)
    Xt, yt = blobs(5000, 0.02, 1.2, 2)
    return train_and_evaluate(X, y, Xv, yv, Xt, yt), yt


def test_result_dict_has_no_scores_but_result_does(fitted):
    res, yt = fitted
    assert "test_scores" not in res.as_dict()     # CSV rows stay scalar
    assert res.test_scores.shape == yt.shape


def test_unit_weights_reproduce_sklearn_metrics(fitted):
    res, yt = fitted
    s, ones = _Scorer(yt, res.test_scores, res.threshold), np.ones(len(yt), dtype=int)
    assert s.f1(ones) == pytest.approx(f1_score(yt, res.test_scores >= res.threshold))
    assert s.ap(ones) == pytest.approx(average_precision_score(yt, res.test_scores))
    # ...and integer weights equal sklearn on the explicitly resampled rows (ties included)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(yt), len(yt))
    w = np.bincount(idx, minlength=len(yt))
    rounded = np.round(res.test_scores, 1)        # force many tied scores
    t = _Scorer(yt, rounded, res.threshold)
    assert t.ap(w) == pytest.approx(average_precision_score(yt[idx], rounded[idx]))
    assert t.f1(w) == pytest.approx(f1_score(yt[idx], rounded[idx] >= res.threshold))


def test_ci_contains_point_estimate(fitted):
    res, yt = fitted
    ci = bootstrap_ci(yt, res.test_scores, res.threshold, n=300)
    assert ci["f1"][0] <= res.f1 <= ci["f1"][1]
    assert ci["ap"][0] <= res.avg_precision <= ci["ap"][1]
    assert ci["f1"][1] - ci["f1"][0] > 0          # a real interval, not a point


def test_bootstrap_is_deterministic_for_a_seed(fitted):
    res, yt = fitted
    a = bootstrap_ci(yt, res.test_scores, res.threshold, n=200, seed=5)
    assert a == bootstrap_ci(yt, res.test_scores, res.threshold, n=200, seed=5)
    assert a != bootstrap_ci(yt, res.test_scores, res.threshold, n=200, seed=6)


def test_failed_run_has_degenerate_ci():
    yt = (np.random.default_rng(0).random(2000) < 0.05).astype(int)
    res = train_and_evaluate(np.zeros((50, 8)), np.zeros(50, int), np.zeros((10, 8)), np.zeros(10, int),
                             np.zeros((2000, 8)), yt)
    ci = bootstrap_ci(yt, res.test_scores, res.threshold, n=100)
    assert ci["f1"] == (0.0, 0.0)                 # never predicts a positive
    assert ci["ap"][0] <= res.avg_precision <= ci["ap"][1]


def test_paired_a_vs_a_is_exactly_zero(fitted):
    res, yt = fitted
    d = paired_bootstrap(yt, res.test_scores, res.threshold, res.test_scores, res.threshold, n=200)
    assert d["f1_diff"] == 0 and d["f1_diff_ci"] == (0.0, 0.0)
    assert d["ap_diff"] == 0 and d["ap_diff_ci"] == (0.0, 0.0)


def test_paired_detects_a_real_gap(fitted):
    res, yt = fitted
    noisy = res.test_scores + np.random.default_rng(1).normal(scale=3, size=len(yt))
    d = paired_bootstrap(yt, res.test_scores, res.threshold, noisy, res.threshold, n=300)
    assert d["f1_diff_ci"][0] > 0 and d["ap_diff_ci"][0] > 0   # A clearly better: CI excludes 0


def test_mean_diff_ci_of_single_runs_is_the_paired_bootstrap(fitted):
    res, yt = fitted
    noisy = res.test_scores + np.random.default_rng(0).normal(0, 3.0, len(yt))
    a = bootstrap_samples(yt, res.test_scores, res.threshold, n=300)[0]
    b = bootstrap_samples(yt, noisy, res.threshold, n=300)[0]
    paired = paired_bootstrap(yt, res.test_scores, res.threshold, noisy, res.threshold, n=300)
    assert mean_diff_ci(a, b) == pytest.approx(paired["f1_diff_ci"])
    assert mean_diff_ci(a, a) == (0.0, 0.0)


def test_mean_diff_ci_widens_with_seed_noise():
    # Same test-set noise in both; B's runs also disagree with each other, so its CI must be wider.
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 0.01, 500)
    a = 0.5 + np.tile(shared, (10, 1))
    steady = 0.4 + np.tile(shared, (10, 1))
    spread = steady + rng.normal(0, 0.1, (10, 1))
    lo_s, hi_s = mean_diff_ci(a, steady)
    lo_w, hi_w = mean_diff_ci(a, spread)
    assert lo_s == pytest.approx(0.1) and hi_s == pytest.approx(0.1)   # paired noise cancels exactly
    assert hi_w - lo_w > 0.05
