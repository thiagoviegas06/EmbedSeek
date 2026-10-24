"""Frozen splits and cached embeddings: disjoint, leak-free, aligned. Skipped if data isn't built."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from embedseek.data import LABELS

DATA = Path(__file__).resolve().parents[1] / "data" / "jigsaw"
SPLITS = ("pool", "val", "test")
pytestmark = pytest.mark.skipif(not (DATA / "splits" / "pool.parquet").exists(), reason="splits not built")


@pytest.fixture(scope="module")
def splits():
    return {s: pd.read_parquet(DATA / "splits" / f"{s}.parquet") for s in SPLITS}


def norm(s):
    return set(s.str.lower().str.split().str.join(" "))


def test_ids_disjoint_across_splits(splits):
    ids = [set(splits[s].id) for s in SPLITS]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_no_test_text_in_pool_or_val(splits):
    test_text = norm(splits["test"].comment_text)
    assert not (norm(splits["pool"].comment_text) & test_text)
    assert not (norm(splits["val"].comment_text) & test_text)


def test_labels_binary_and_counts_match_stats(splits):
    stats = json.loads((DATA / "splits" / "split_stats.json").read_text())
    for s in SPLITS:
        assert splits[s][LABELS].isin([0, 1]).all().all()  # no -1 "unscored" rows leaked into test
        assert len(splits[s]) == stats[s]["rows"]
        for l in LABELS:
            assert int(splits[s][l].sum()) == stats[s][l]


def test_val_prevalence_matches_pool(splits):
    for l in ("threat", "identity_hate"):
        assert splits["val"][l].mean() == pytest.approx(splits["pool"][l].mean(), rel=0.15)


@pytest.mark.parametrize("model", [p.name for p in (DATA / "emb").glob("*") if p.is_dir()] or ["none"])
def test_embeddings_aligned_and_normalized(splits, model):
    if model == "none":
        pytest.skip("no embeddings cached")
    for s in SPLITS:
        emb = np.load(DATA / "emb" / model / f"{s}.npy").astype(np.float32)
        assert len(emb) == len(splits[s])
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-2)


def test_cached_embeddings_match_fresh_encoding(splits):
    """Row i of the cache must be the embedding of row i's text (catches any misalignment)."""
    from sentence_transformers import SentenceTransformer

    model_dir = DATA / "emb" / "all-MiniLM-L6-v2"
    if not model_dir.exists():
        pytest.skip("MiniLM cache missing")
    rows = np.random.default_rng(0).choice(len(splits["pool"]), 20, replace=False)
    fresh = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(
        splits["pool"].comment_text.iloc[rows].tolist(), normalize_embeddings=True)
    cached = np.load(model_dir / "pool.npy")[rows].astype(np.float32)
    assert (np.sum(fresh * cached, axis=1) > 0.99).all()
