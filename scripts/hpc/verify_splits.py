"""Check that rebuilt splits are identical to the frozen ones every result was computed on.

Compares row counts, positive counts and a fingerprint of the ordered `id` column (row order
matters: embeddings are row-aligned). Parquet bytes can differ across library versions, so
this compares content rather than file checksums.

Usage: python -I scripts/hpc/verify_splits.py data/jigsaw/splits
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

EXPECTED = {  # split: (rows, threat, identity_hate, sha256(ids)[:16]) from the original build, 2026-10-24
    "pool": (143354, 427, 1257, "dc5b1f1d0001bdac"),
    "val": (15929, 47, 140, "180c17b43d45f523"),
    "test": (63939, 211, 712, "35e05c83f7ffa5b5"),
}

splits = Path(sys.argv[1])
ok = True
for name, expected in EXPECTED.items():
    df = pd.read_parquet(splits / f"{name}.parquet")
    got = (len(df), int(df.threat.sum()), int(df.identity_hate.sum()),
           hashlib.sha256("\n".join(df["id"].astype(str)).encode()).hexdigest()[:16])
    match = got == expected
    ok &= match
    print(f"{name:5s} {'OK      ' if match else 'MISMATCH'} got {got}" + ("" if match else f" expected {expected}"))
sys.exit(0 if ok else 1)
