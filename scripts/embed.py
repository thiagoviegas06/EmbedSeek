"""Embed every split with one sentence-transformers model and cache the vectors.

Writes <out>/<model_slug>/{pool,val,test}.npy (float16, L2-normalized) and meta.json
with wall-clock time per split (C_embed) and the share of texts truncated by max_seq_length.

Usage: python -I scripts/embed.py data/jigsaw/splits data/jigsaw/emb sentence-transformers/all-MiniLM-L6-v2
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

splits, out_root, model_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 256

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)
out = out_root / model_name.split("/")[-1]
out.mkdir(parents=True, exist_ok=True)

meta = {"model": model_name, "device": device, "dim": model.get_embedding_dimension(),
        "max_seq_length": model.max_seq_length, "splits": {}}

for name in ["pool", "val", "test"]:
    df = pd.read_parquet(splits / f"{name}.parquet")
    texts = df.comment_text.tolist()

    n_tokens = np.array([len(ids) for ids in model.tokenizer(texts, add_special_tokens=True)["input_ids"]])
    truncated = n_tokens > model.max_seq_length

    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    secs = time.perf_counter() - t0

    np.save(out / f"{name}.npy", emb.astype(np.float16))
    meta["splits"][name] = {
        "rows": len(texts), "seconds": round(secs, 1),
        "truncated_frac": round(float(truncated.mean()), 4),
        "truncated_frac_threat": round(float(truncated[df.threat.values == 1].mean()), 4),
        "truncated_frac_identity_hate": round(float(truncated[df.identity_hate.values == 1].mean()), 4),
    }
    print(name, meta["splits"][name], flush=True)

(out / "meta.json").write_text(json.dumps(meta, indent=2))
