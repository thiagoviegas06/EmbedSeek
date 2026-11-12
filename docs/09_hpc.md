# Running on the NYU HPC cluster

Embedding with larger encoders on a laptop is too slow: mpnet took ~16 h for the pool on an M3 Pro, against 2.5 min for MiniLM. Run embedding (and later, LLM labeling) as GPU jobs on the cluster. Everything downstream (samplers, proxy, plots, tests) runs fine on a laptop once the `.npy` files are copied back.

| Work | Where | Why |
|---|---|---|
| Embedding the pool/val/test with each encoder | **HPC (GPU job)** | Minutes on a GPU vs hours locally |
| Sampler experiments, proxy training, figures | Laptop or HPC | LR on cached vectors is cheap |
| Real-LLM labeling run (later) | HPC or API | Depends on the model chosen |

Placeholders below: `<netid>` = your NetID, `<login>` = the cluster login host, `<account>` = the Slurm account for the course/project. Get these from the NYU HPC documentation or the course staff. You may need the NYU VPN off campus.

## 1. Package the project (on your laptop)

```bash
bash scripts/hpc/make_bundle.sh      # → ../embedseek_bundle_<date>.tar.gz (~270 MB)
```

The bundle is the **entire project**, so it doesn't depend on what's pushed to GitHub:
- **Code, docs and results:** everything in the repo, including uncommitted work.
- **Git history:** `.git`, so you can commit and push from the cluster.
- **Local notes:** `FUTURE_DIRECTIONS.md` and the papers.
- **Data:** raw CSVs, frozen splits and finished embeddings, plus `data/MANIFEST.sha256` (checksums for every data file).

It leaves out:
- **`.venv`:** macOS binaries won't run on Linux, so it's rebuilt on the cluster.
- **Caches.**
- **Unfinished embedding folders:** any folder without `meta.json` (currently the partial mpnet run).

## 2. Copy it up and unpack (on the cluster)

```bash
scp ../embedseek_bundle_<date>.tar.gz <netid>@<login>:/scratch/<netid>/
ssh <netid>@<login>
cd $SCRATCH                     # code + data on scratch: the home quota is small
tar -xzf embedseek_bundle_<date>.tar.gz && cd EmbedSeek
```

## 3. Set up the environment (on the cluster, once)

```bash
module avail python             # pick a Python >= 3.11 module
module load python/<version>
bash scripts/hpc/setup_env.sh   # builds .venv, verifies data checksums, checks torch, runs the tests
```

`setup_env.sh` stops at the first failure. "data checksums OK" means the copy is intact.

On Linux, the pinned `torch` wheel from PyPI includes CUDA support. A login node usually has no GPU, so "GPU visible: False" there is expected; the job script prints the GPU it gets. If NYU recommends a container or conda overlay instead of a plain venv, follow that and run the `pip install` lines from `setup_env.sh` inside it.

## 4. Submit the embedding job

Edit `#SBATCH --account=` (and add `--partition` if required) in [scripts/hpc/embed.sbatch](../scripts/hpc/embed.sbatch), then:

```bash
mkdir -p logs                   # Slurm won't create the log directory
sbatch scripts/hpc/embed.sbatch # array: 0 = mpnet, 1 = bge, 2 = MiniLM
squeue -u $USER                 # watch progress
tail -f logs/embedseek-embed-*_0.out
```

To add an encoder, append it to `MODELS` and raise `--array=0-N`.

## 5. Bring results back (optional)

If you keep working on the cluster, there's nothing to copy. Commit code from there; `.git` came with the bundle. To also work on the laptop, pull the new embeddings down:

```bash
# on your laptop
rsync -av <netid>@<login>:/scratch/<netid>/EmbedSeek/data/jigsaw/emb/ data/jigsaw/emb/
.venv/bin/python -m pytest tests -q   # alignment + normalization checks run on every cached model
```

Each `meta.json` records the GPU name, batch size, max sequence length and seconds per split. These feed the cost term `C_embed`. Note that GPU timings aren't comparable to MiniLM's laptop timing; for the cost analysis, re-time MiniLM on the same GPU (it takes about a minute).

## Notes

- **Scratch is purged** after a period of inactivity on most clusters. Push code to GitHub regularly and keep the bundle (or the data) somewhere safe; don't treat scratch as storage.
- **Sequence length:** encoders use their own limit by default (MiniLM 256, mpnet 384, bge 512 tokens). Pass a 5th argument to `embed.py` to cap it, e.g. `256` for all models. Decide this as a team: capping makes encoders more comparable but truncates more long comments (~7% exceed 256 tokens).
- **Laptop fallback:** if you must embed locally, use a small batch (`... <model> 32`). The ~100x slowdown was most likely batch 256 x long sequences exhausting unified memory.
