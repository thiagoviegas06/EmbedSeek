"""Load frozen splits and cached embeddings."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
