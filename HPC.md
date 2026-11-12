# HPC.md: picking up EmbedSeek on the NYU HPC cluster

**For an AI agent (or teammate) continuing this project on the cluster.** Read this file top to bottom before doing anything. It covers what the project is, how to bootstrap the environment and data, the rules you must not break, and the ordered task queue. When you finish a session, update [§7 Status log](#7-status-log).

---

## 1. The project in brief

EmbedSeek is an NYU Data Engineering research project (Oct 11 – Dec 10, 2026; team: Thiago Viegas, Deep Paghdal, Prashanth Sreenivasan).

- **Setting:** an LLM labels a small sample of rows; a cheap **proxy** classifier (logistic regression on text embeddings) is trained on those labels and run over the whole table.
- **Problem:** when the predicate is rare (Jigsaw `threat` = 0.30%), a random sample contains almost no positives, so the proxy can't learn.
- **Our change:** only the **sampling step**. Smarter samplers search embedding spaces to find rare positives. The proposed method is a **two-level Thompson-sampling bandit**: pick an embedding space (encoder), then a k-means cluster inside it.
- **Fixed for every method:** the proxy (LR, `class_weight="balanced"`, one fixed embedding, threshold chosen on val, metrics on test). Better F1 must come from better training rows.

**Read these before coding:**

| Doc | Why |
|---|---|
| [README.md](README.md) | Research questions, current results, repo layout, sampler contract (§10) |
| [docs/07_updated_direction.md](docs/07_updated_direction.md) | **The current plan:** two-level bandit, screening, bootstrap CIs, task order. Includes the method figure |
| [docs/08_metrics_reference.md](docs/08_metrics_reference.md) | Exactly how every metric is computed; what "training the proxy" means |
| [docs/06_baseline_results.md](docs/06_baseline_results.md) | Baseline numbers you will compare against |
| [docs/09_hpc.md](docs/09_hpc.md) | Human-oriented cluster guide (the commands below are the same) |

## 2. Rules you must not break

**Experimental integrity**
- **Never modify the frozen splits.** After building data, `scripts/hpc/verify_splits.py` must print OK for all three splits. Embeddings are row-aligned with the splits, so row order matters.
- **Test labels are for final metrics only.** Never use them for sampling, thresholds or design decisions. The threshold comes from val.
- **Samplers see labels only through `oracle(idx)`.** Never pass the full label array to a sampler. Every new sampler gets a spy-oracle test (see `test_lts_lite_budget_uniqueness_and_no_peeking` in [tests/test_samplers.py](tests/test_samplers.py)).
- **The proxy stays identical across methods.** Don't tune it per sampler.
- **Seeds:** 10 seeds (0–9) for every stochastic sampler; budgets B ∈ {100, 250, 500, 1000}.
- **Diagnostics that read true labels** (e.g. `scripts/filter_recall.py`) are analysis only. Never feed them into a sampler.

**Cluster etiquette**
- **No heavy compute on login nodes.** Embedding, k-means on full pools and multi-seed sweeps go through `sbatch` (or `srun` for short interactive work). Downloading data, `pip install` and running the test suite are fine on a login node.
- **Work under `$SCRATCH`**; set `HF_HOME=$SCRATCH/hf_cache`, since model downloads overflow the home quota.
- **Scratch gets purged.** Anything worth keeping goes into git (code, docs, result CSVs, figures) and gets pushed.

**Git**
- `data/` is gitignored. Never commit data, embeddings or `.venv`.
- Work on a branch (`git checkout -b <topic>`), commit with clear messages, push, and let the humans merge.
- Match the existing code style: small modules in `src/embedseek/`, entry-point scripts in `scripts/` that take positional arguments and write CSVs to `results/`, a short docstring with a `Usage:` line, and comment density similar to the existing files.

**Ask the human, don't guess,** for:
- the Slurm account, partition, or which Python module to load;
- the open research decisions in §6.

## 3. Bootstrap (once per fresh clone)

```bash
cd $SCRATCH
git clone https://github.com/thiagoviegas06/EmbedSeek.git && cd EmbedSeek
module avail python                 # need Python >= 3.11; ask the human if unsure which module
module load python/<version>
bash scripts/hpc/setup_env.sh       # builds .venv, installs pinned deps, runs tests (data tests skip until data exists)
source .venv/bin/activate
export HF_HOME=$SCRATCH/hf_cache
```

**Rebuild the data** (light; fine on a login node, or use `srun` if login-node policy forbids it):

```bash
mkdir -p data/jigsaw/raw
for f in train.csv test.csv test_labels.csv; do
  curl -L -o data/jigsaw/raw/$f \
    "https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/resolve/main/$f"
done
python -I scripts/make_splits.py data/jigsaw/raw data/jigsaw/splits
python -I scripts/hpc/verify_splits.py data/jigsaw/splits   # MUST print OK x3; stop and report if not
```

If verification fails, **stop**. Results in `results/` were computed on the original splits. Ask the human to copy the originals up instead:

```bash
rsync -av data/jigsaw/splits/ <netid>@<login>:$SCRATCH/EmbedSeek/data/jigsaw/splits/
```

**Embed on GPU** (mpnet, bge and MiniLM as a 3-task array):

```bash
# first: set #SBATCH --account (and --partition if needed) in scripts/hpc/embed.sbatch; ask the human for values
mkdir -p logs
sbatch scripts/hpc/embed.sbatch
squeue -u $USER
```

Each task writes `data/jigsaw/emb/<model>/{pool,val,test}.npy` plus `meta.json`. When all three finish, `python -m pytest tests -q` must pass. It checks row alignment and normalization for every cached model.

## 4. Current state (2026-11-12)

**Done** (committed):
- Harness: `src/embedseek/` with `data.py`, `proxy.py`, `samplers.py` (`random_sample`, `query_seeded`, `lts_lite`).
- Frozen Jigsaw splits; MiniLM embeddings (laptop).
- Baselines: ceiling, random, query-seeded, LTS-lite on `threat` and `identity_hate`. CSVs and figures are in `results/`.
- Screening diagnostic (`scripts/filter_recall.py`): MiniLM top-10% by query similarity holds 85% of pool threats (random: 10%).
- Docs 07–09, the method figure, and the HPC scripts.

**Not done:**
- mpnet and bge embeddings. The laptop run was ~100x too slow, which is why we moved here.
- No error bars yet: every comparison so far is a point estimate.

**Key baseline numbers** (test F1 at B = 1000, MiniLM for search and proxy):

| Sampler | threat | identity_hate |
|---|---|---|
| Random | 0.21 | 0.38 |
| Query: comment-like | 0.38 | 0.36 |
| LTS-lite | 0.30 | 0.42 |
| Ceiling: all pool labels | 0.43 | 0.46 |

## 5. Task queue (in order)

Each task has a definition of done. Don't start the bandit (task 5) before task 3's go/no-go.

**1. Embeddings on GPU.** Covered in §3.
*Done when:* three `meta.json` files exist and the tests pass.

**2. Bootstrap CIs** in [src/embedseek/proxy.py](src/embedseek/proxy.py). This can run in parallel with task 1 and needs no GPU.
- `train_and_evaluate` also returns the raw test scores and the threshold, without breaking existing callers.
- `bootstrap_ci(y_test, scores, thr, n=1000, seed=0)` returns 95% CIs for F1 and AP by resampling test rows with replacement. Predictions stay fixed; nothing is refit.
- `paired_bootstrap(y_test, scores_a, thr_a, scores_b, thr_b, n=1000, seed=0)` returns the CI of F1_A − F1_B using the same resamples for both.
- Tests: the CI contains the point estimate; A vs A gives a difference of exactly 0; results are deterministic for a fixed seed.

*Done when:* the tests pass and the baselines report CIs. See docs/08 §8.

**3. Encoder diagnostics: the go/no-go** (new `scripts/encoder_diagnostics.py`, run via `sbatch`). For each encoder (MiniLM, mpnet, bge) and predicate:
- Ceiling F1/AP, i.e. the proxy trained on all pool labels.
- k-means (k = 50, `random_state=0`) and the share of positives held by the top-5 clusters. MiniLM reference: threat 72%, identity_hate 68%.
- Overlap of *which* positives each encoder's top-5 clusters hold: pairwise Jaccard, plus a count of positives concentrated by only one encoder.
- Overlap of the comment-like query's top-1000 results between encoders. bge needs its query instruction prefix on the *query* only; check the model card.

Write results to `results/encoder_diagnostics.csv` and a short write-up in `docs/10_encoder_diagnostics.md`.
*Done when:* the write-up states go or no-go. **Go** = encoders concentrate meaningfully different positives, or different encoders win on different predicates. **No-go** = high overlap everywhere. Report either way and stop for the humans to decide.

**4. Choose the fixed proxy embedding** from task 3's ceilings. Propose it in the write-up and let the humans confirm (see §6).

**5. Two-level bandit** in `src/embedseek/samplers.py`, only after a go. The spec is in docs/07, "Idea 1".
- Same contract as `lts_lite`: `oracle`, `budget`, `seed`; returns exactly `budget` unique indices.
- Options: `share_labels` (a labeled row updates its cluster in every encoder) and `warm_start` (one query-seeded batch per encoder).
- Also a flat variant, with every (encoder, cluster) pair as one arm, for the ablation.
- A spy-oracle test, plus a test where a planted positive-rich cluster in one encoder gets exploited.
- Run script: `scripts/run_bandit.py`, writing `results/bandit_<proxy-embedding>.csv` with the existing CSV columns.

**6. Screening sweep** (docs/07, "Idea 2"): s ∈ {1.0, 0.5, 0.25, 0.1}, simulated by masking (no re-embedding). Cost is charged from `meta.json` timings.

## 6. Open decisions: ask the humans, don't decide alone

- **Sequence length:** each encoder's own limit (MiniLM 256, mpnet 384, bge 512) vs. capping all at 256. The current job uses each model's own limit.
- **Proxy embedding:** which one, and whether it may also be a search arm (it might favor that arm).
- **Bandit reward:** binary "batch found a positive" (LTS-like; the default) vs. positives per batch.
- **Main method:** the two-level bandit is the assumed core; query/anchor search within an encoder is a later extension.

## 7. Status log

Append a dated entry at the end of every working session: what you did, what's running (Slurm job IDs), what's blocked, and what's next.

- **2026-11-12 (laptop):** Wrote this handoff. Repo pushed with the HPC scripts. Next: bootstrap on the cluster (§3), then tasks 1 and 2 in parallel.
