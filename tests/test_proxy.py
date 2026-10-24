"""Proxy: failed samples handled, learns when it should, threshold never uses test labels."""
import numpy as np

from embedseek.proxy import train_and_evaluate


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
