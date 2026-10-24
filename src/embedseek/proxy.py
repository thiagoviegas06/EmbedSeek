"""Shared downstream proxy: identical for every sampler (proposal §3.5).

Class-weighted logistic regression on a fixed reference embedding, threshold chosen
to maximize F1 on validation, metrics reported on the held-out human-labeled test set.
"""
from dataclasses import asdict, dataclass

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

    def as_dict(self) -> dict:
        return asdict(self)


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # thr has one fewer entry than prec/rec; the last PR point has no threshold.
    return float(thr[np.argmax(f1[:-1])])


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test) -> ProxyResult:
    n_pos = int(y_train.sum())
    # A sample with no positives (or no negatives) cannot train a classifier:
    # record it as a failed run (predict all negative) instead of dropping it.
    if n_pos == 0 or n_pos == len(y_train):
        return ProxyResult(len(y_train), n_pos, False, float("nan"), 0.0, 0.0, 0.0, float(y_test.mean()))

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    thr = best_f1_threshold(y_val, clf.decision_function(X_val))

    test_scores = clf.decision_function(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, test_scores >= thr, average="binary", zero_division=0)
    return ProxyResult(len(y_train), n_pos, True, thr, float(f1), float(p), float(r),
                       float(average_precision_score(y_test, test_scores)))
