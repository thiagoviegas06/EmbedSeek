"""Load frozen splits and cached embeddings."""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@dataclass
class Split:
    df: pd.DataFrame
    emb: np.ndarray  # float32, L2-normalized, row-aligned with df

    def y(self, label: str) -> np.ndarray:
        return self.df[label].to_numpy(dtype=np.int8)


def load_split(data_dir: Path, name: str, model_slug: str) -> Split:
    df = pd.read_parquet(data_dir / "splits" / f"{name}.parquet")
    emb = np.load(data_dir / "emb" / model_slug / f"{name}.npy").astype(np.float32)
    assert len(df) == len(emb), f"{name}: {len(df)} rows vs {len(emb)} embeddings"
    return Split(df, emb)


def load_clusters(data_dir: Path, model_slug: str, k: int = 50) -> np.ndarray:
    """k-means cluster id of every pool row in one encoder's space, cached next to the embeddings.

    Computed once (n_init=1, random_state=0) and saved as emb/<model>/clusters_k<k>.npy, because
    k-means does not give the same clusters on every machine and every script must share them.
    Prints a short hash so logs show which clustering a run used.
    """
    path = data_dir / "emb" / model_slug / f"clusters_k{k}.npy"
    if not path.exists():
        emb = np.load(data_dir / "emb" / model_slug / "pool.npy").astype(np.float32)
        ids = KMeans(n_clusters=k, n_init=1, random_state=0).fit_predict(emb).astype(np.int16)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, ids)
        os.replace(tmp, path)   # atomic, so a concurrent job never reads a half-written file
    ids = np.load(path).astype(np.int64)
    print(f"[{model_slug}] clusters k={k}: {path.name} md5 {hashlib.md5(ids.tobytes()).hexdigest()[:8]}", flush=True)
    return ids
