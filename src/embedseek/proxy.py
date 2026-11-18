"""Shared downstream proxy: identical for every sampler (proposal §3.5).

Class-weighted logistic regression on a fixed reference embedding, threshold chosen
to maximize F1 on validation, metrics reported on the held-out human-labeled test set.

Uncertainty from the finite test set: `bootstrap_ci` (one model) and `paired_bootstrap`
(model A vs B on the same resamples). Both hold predictions fixed and refit nothing.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support


@dataclass
class ProxyResult:
    n_labeled: int
    n_pos: int
    trained: bool
    threshold: float
    f1: float
    precision: float
    recall: float
    avg_precision: float
    # Raw decision scores on test, for bootstrap CIs. Kept out of as_dict() so CSV rows stay scalar.
    test_scores: np.ndarray | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict:
        d = asdict(self)
        del d["test_scores"]
        return d


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # thr has one fewer entry than prec/rec; the last PR point has no threshold.
    return float(thr[np.argmax(f1[:-1])])


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test) -> ProxyResult:
    n_pos = int(y_train.sum())
    # A sample with no positives (or no negatives) cannot train a classifier:
    # record it as a failed run (predict all negative) instead of dropping it.
    # Its test scores are constant and its threshold NaN (scores >= NaN is all False), so bootstrap
    # resamples of a failed run give F1 = 0 and AP = prevalence, matching the point estimate.
    if n_pos == 0 or n_pos == len(y_train):
        return ProxyResult(len(y_train), n_pos, False, float("nan"), 0.0, 0.0, 0.0, float(y_test.mean()),
                           np.zeros(len(y_test)))

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    thr = best_f1_threshold(y_val, clf.decision_function(X_val))

    test_scores = clf.decision_function(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, test_scores >= thr, average="binary", zero_division=0)
    return ProxyResult(len(y_train), n_pos, True, thr, float(f1), float(p), float(r),
                       float(average_precision_score(y_test, test_scores)), test_scores)


# ---- Test-set bootstrap -------------------------------------------------------------------------
# A resample of the N test rows (with replacement) is represented by per-row counts w (sum N),
# so F1 and AP are weighted sums over the original rows: no N x n index matrix, no re-sorting.

def _resample_weights(n_rows: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.bincount(rng.integers(0, n_rows, n_rows), minlength=n_rows)


class _Scorer:
    """F1 at a fixed threshold and AP for one model's fixed test scores, under row weights."""

    def __init__(self, y, scores, thr):
        y = np.asarray(y).astype(bool)
        pred = np.asarray(scores) >= thr
        self.tp, self.fp, self.fn = y & pred, ~y & pred, y & ~pred
        # AP = sum over distinct thresholds (high to low) of (R_k - R_k-1) * P_k, as in sklearn.
        self.order = np.argsort(-np.asarray(scores), kind="stable")
        s_sorted = np.asarray(scores)[self.order]
        self.y_sorted = y[self.order]
        self.group_end = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]  # last row of each tie

    def f1(self, w):
        tp, fp, fn = w @ self.tp, w @ self.fp, w @ self.fn
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    def ap(self, w):
        w = w[self.order]
        tps = np.cumsum(w * self.y_sorted)[self.group_end]
        fps = np.cumsum(w * ~self.y_sorted)[self.group_end]
        if tps[-1] == 0:
            return np.nan                               # no positive drawn: AP undefined
        precision = tps / np.maximum(tps + fps, 1)
        recall = tps / tps[-1]
        return float(np.sum(np.diff(recall, prepend=0) * precision))


def _ci(values, level=0.95):
    lo, hi = np.nanpercentile(values, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def bootstrap_samples(y_test, scores, thr, n=1000, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """F1 (at the fixed threshold) and AP on each of `n` test-set resamples: two arrays of length n.

    Resample j is the same test rows for every call with the same seed and test set, so arrays
    from different models are paired column by column.
    """
    s = _Scorer(y_test, scores, thr)
    f1s, aps = zip(*[(s.f1(w), s.ap(w)) for w in _resample_weights(len(y_test), n, seed)])
    return np.array(f1s), np.array(aps)


def bootstrap_ci(y_test, scores, thr, n=1000, seed=0) -> dict:
    """95% percentile CIs for test F1 (at the fixed threshold) and AP, resampling test rows.

    Returns {"f1": (lo, hi), "ap": (lo, hi)}. Uses the same resamples as any other call with the
    same seed and test set, so CIs from one run script are computed on identical draws.
    """
    f1s, aps = bootstrap_samples(y_test, scores, thr, n, seed)
    return {"f1": _ci(f1s), "ap": _ci(aps)}


def ci_columns(y_test, res: ProxyResult, n=1000, seed=0) -> dict:
    """bootstrap_ci of one ProxyResult as flat CSV columns: f1_lo, f1_hi, ap_lo, ap_hi."""
    ci = bootstrap_ci(y_test, res.test_scores, res.threshold, n, seed)
    return {"f1_lo": ci["f1"][0], "f1_hi": ci["f1"][1], "ap_lo": ci["ap"][0], "ap_hi": ci["ap"][1]}


def paired_bootstrap(y_test, scores_a, thr_a, scores_b, thr_b, n=1000, seed=0) -> dict:
    """CI of F1_A - F1_B and AP_A - AP_B, with A and B scored on the same resamples each time.

    Returns {"f1_diff": point, "f1_diff_ci": (lo, hi), "ap_diff": point, "ap_diff_ci": (lo, hi)}.
    A gap is significant at 95% if its CI excludes 0.
    """
    a, b = _Scorer(y_test, scores_a, thr_a), _Scorer(y_test, scores_b, thr_b)
    ones = np.ones(len(y_test), dtype=int)
    d_f1, d_ap = zip(*[(a.f1(w) - b.f1(w), a.ap(w) - b.ap(w)) for w in _resample_weights(len(y_test), n, seed)])
    return {"f1_diff": a.f1(ones) - b.f1(ones), "f1_diff_ci": _ci(d_f1),
            "ap_diff": a.ap(ones) - b.ap(ones), "ap_diff_ci": _ci(d_ap)}


def mean_diff_ci(boot_a, boot_b, seed=0) -> tuple[float, float]:
    """95% CI of mean(A) - mean(B) for two methods run with several sampler seeds.

    `boot_a` and `boot_b` are (runs, n) arrays: one row of `bootstrap_samples` per sampler seed
    (one row for a deterministic sampler). Each of the n iterations uses one test-set resample
    (column j, shared by A and B) and redraws each method's runs with replacement, so the CI
    covers both test-set noise and sampler-seed noise. A gap is significant if it excludes 0.
    """
    a, b = np.atleast_2d(boot_a), np.atleast_2d(boot_b)
    rng = np.random.default_rng(seed)
    cols = np.arange(a.shape[1])
    mean_a = a[rng.integers(0, len(a), (len(a), len(cols))), cols].mean(axis=0)
    mean_b = b[rng.integers(0, len(b), (len(b), len(cols))), cols].mean(axis=0)
    return _ci(mean_a - mean_b)
