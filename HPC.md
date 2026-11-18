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

### On NYU Torch (how it is actually set up)

- **Location:** the clone lives at `/scratch/tjv235/data-engineering/EmbedSeek`.
- **Environment:** interactive work runs inside a singularity container (`cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` + an ext3 overlay). There is no `module` command there.
  - `.venv` was built with `bash scripts/hpc/setup_env.sh /ext3/miniforge3/bin/python3` (Python 3.12), so it only runs where that overlay is mounted.
  - Batch jobs mount `/scratch/$USER/neuro3.ext3` read-only (override with `OVL=...`). Don't hold that overlay open read-write in an interactive session while jobs run: the job will fail to mount it.
- **Submitting:** `sbatch`/`squeue` don't exist inside the container. Submit from a login node:
  - GPU: `sbatch scripts/hpc/embed.sbatch` (account `torch_pr_60_tandon_priority`, partition `a100_tandon`).
  - CPU: `sbatch scripts/hpc/cpu.sbatch <script.py> [args...]` (partition `cpu_short`, 8 CPUs, 64 GB, 2 h).
  - `command.md` (untracked) lists the exact commands to run.
- **Caches:** set `PIP_CACHE_DIR=$SCRATCH/pip_cache` and `HF_HOME=$SCRATCH/hf_cache`. All three encoders are already downloaded there.
- **Git:** this session has no GitHub credentials. Push from a login node where you are authenticated.

## 4. Current state (2026-11-14)

**Done:**
- Harness: `src/embedseek/` with `data.py`, `proxy.py`, `samplers.py` (`random_sample`, `query_seeded`, `lts_lite`).
- Frozen Jigsaw splits: rebuilt on Torch, `verify_splits.py` OK x3.
- Embeddings for MiniLM, mpnet and bge on an A100 (job 22865730). See the timings below.
- Baselines: ceiling, random, query-seeded, LTS-lite on `threat` and `identity_hate`. CSVs and figures are in `results/`.
- **Bootstrap CIs (task 2):** `bootstrap_ci`, `paired_bootstrap` and `ci_columns` in `proxy.py`, plus 7 new tests (25 tests total). Both baseline runners now write `f1_lo, f1_hi, ap_lo, ap_hi`.
- Screening diagnostic (`scripts/filter_recall.py`): MiniLM top-10% by query similarity holds 85% of pool threats (random: 10%).
- Docs 07–09, the method figure, and the HPC scripts (`embed.sbatch`, new `cpu.sbatch`).
- **Encoder diagnostics (task 3):** full run done (job 23156866), `results/encoder_diagnostics.csv`, write-up in [docs/10_encoder_diagnostics.md](docs/10_encoder_diagnostics.md). Verdict: a **qualified go**, awaiting the humans' confirmation. bge is the best single encoder nearly everywhere; complementarity is clearer in query retrieval than in clusters.

**Not done:**
- The bandit re-run with controls (task 5): code is ready, submit `command.md` §5. It adds random, per-encoder query, multi-encoder query, warm start + random, and LTS-lite with a warm start, plus pairwise difference CIs.
- Re-running `run_lts_lite.py` and `encoder_diagnostics.py` on the cached clusters, so every result shares one clustering (docs/10's cluster numbers came from an uncached k-means).
- The humans' confirmation of the proxy embedding (proposed: bge). `run_bandit.py` takes it as an argument, so both bge and MiniLM can be run.
- Equal-budget versions of the union comparisons in docs/10 (bge top-15 clusters; bge top-3000 query results).
- Paired comparisons between baselines (the function exists; no method pairs have been run yet).
- `docs/06` and README still show point estimates only and say "16 tests".

**Embedding cost** (A100-80GB, batch 128, each model's own max length; seconds):

| Model | dim | max tokens | pool | val | test | pool rows truncated |
|---|---|---|---|---|---|---|
| MiniLM | 384 | 256 | 47 | 5 | 25 | 6.9% |
| mpnet | 768 | 384 | 225 | 26 | 131 | 3.4% |
| bge | 768 | 512 | 220 | 25 | 131 | 2.2% |

**Key baseline numbers** (test F1 at B = 1000 with 95% test-set bootstrap CI; MiniLM for search and proxy):

| Sampler | threat | identity_hate |
|---|---|---|
| Random (10-seed mean) | 0.21 [0.16, 0.25] | 0.38 [0.35, 0.41] |
| Query: comment-like | 0.38 [0.33, 0.43] | 0.36 [0.34, 0.39] |
| LTS-lite (10-seed mean, re-run) | 0.29 [0.24, 0.33] | 0.42 [0.39, 0.45] |
| Ceiling: all pool labels | 0.43 [0.37, 0.49] | 0.46 [0.43, 0.49] |

For seeded samplers, the bracket is the mean of the per-seed test-set CIs. Seed spread is a separate error source (docs/08 §8). The CIs are wide (±0.05 F1 for threat), so most gaps between methods need a paired bootstrap before calling them real.

**LTS-lite re-run caveat:**
- Re-running `run_lts_lite.py` on the GPU MiniLM vectors gave different k-means clusters, which changed the trajectories.
- At B = 1000 the numbers are about the same: threat F1 0.30 → 0.29 with 26 → 36 positives found; identity_hate 0.42 → 0.42.
- At small budgets they moved a lot: identity_hate B = 250 F1 0.26 → 0.13.
- So LTS-lite at low budgets is sensitive to the clustering. The new CSV is in the working tree, uncommitted, pending the humans' choice (see §6).
- `run_baselines.py` reproduced every committed point estimate exactly.

## 5. Task queue (in order)

Each task has a definition of done. Don't start the bandit (task 5) before task 3's go/no-go.

**1. Embeddings on GPU.** Covered in §3. ✅ **Done 2026-11-13** (job 22865730).
*Done when:* three `meta.json` files exist and the tests pass.

**2. Bootstrap CIs** ✅ **Done 2026-11-13** (code, tests, baseline CSV columns; uncommitted). In [src/embedseek/proxy.py](src/embedseek/proxy.py). This can run in parallel with task 1 and needs no GPU.
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

✅ **Run and written up 2026-11-14** (job 23156866); the verdict is a qualified go, pending the humans.
- The script also reports ceiling CIs, the share of pool rows the top-5 clusters cover, and positives found by any encoder vs. the best single encoder.
- bge's query runs both with the instruction prefix and without it (`_noprefix`), since the model card says no instruction for symmetric similarity.
- Re-run with `sbatch scripts/hpc/cpu.sbatch scripts/encoder_diagnostics.py data/jigsaw` (3 min).
*Done when:* the write-up states go or no-go. **Go** = encoders concentrate meaningfully different positives, or different encoders win on different predicates. **No-go** = high overlap everywhere. Report either way and stop for the humans to decide.

**4. Choose the fixed proxy embedding** from task 3's ceilings. Proposed in docs/10: **bge** (ceiling F1 0.48 threat, 0.52 identity_hate, against 0.43 and 0.46 for MiniLM and mpnet). Waiting for the humans to confirm (see §6).

**5. Two-level bandit** in `src/embedseek/samplers.py`, only after a go. The spec is in docs/07, "Idea 1". 🔄 **Go given 2026-11-14; first full run done (jobs 23168988 bge proxy, 23169002 MiniLM proxy); controls still missing, see §7.**
- `hierarchical_lts` (with `flat=True` for the ablation) and `scripts/run_bandit.py`. With one encoder it is exactly `lts_lite` (tested).
- `share_labels` rule (the agent's interpretation of docs/07, for the humans to check): a cluster in another encoder holding a fraction f of the batch gets f of a win if any of its rows is positive, else f of a loss.
- Reward is binary; the within-cluster LR uses the chosen encoder's embedding; warm start uses the comment-like query with no bge prefix.
- The run also includes LTS-lite on each encoder alone with the same proxy. Not included yet: LTS-lite on concatenated embeddings.
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
- **LTS-lite results CSV:** replace the committed laptop-run CSV with the Torch re-run (it has CIs and is consistent with the cluster embeddings), or keep the original? See §4 for how they differ.

## 7. Status log

Append a dated entry at the end of every working session: what you did, what's running (Slurm job IDs), what's blocked, and what's next.

- **2026-11-12 (laptop):** Wrote this handoff. Repo pushed with the HPC scripts. Next: bootstrap on the cluster (§3), then tasks 1 and 2 in parallel.
- **2026-11-12 (Torch):** Bootstrapped in an interactive singularity session (no `module` command there): `.venv` built from the overlay's `/ext3/miniforge3` Python 3.12 (torch 2.14.1+cu130). Raw data downloaded, splits rebuilt, `verify_splits.py` OK x3, tests 14 passed / 2 skipped (no embeddings yet). Models pre-downloaded to `$SCRATCH/hf_cache`. `embed.sbatch` now targets `torch_pr_60_tandon_priority` / `a100_tandon` and runs inside `singularity exec --nv --overlay <ext3>:ro`, since the venv's Python lives in the overlay. Gotchas: `sbatch` only exists outside the container (submit from a login node), and a job can't mount an overlay that an interactive session holds read-write. Next: submit the embedding array (task 1), start bootstrap CIs (task 2).
- **2026-11-13 (Torch):** Task 1 done. Job 22865730 (A100-80GB, driver 610.43, batch 128, each model's own max length) embedded mpnet, bge and MiniLM; tests 18 passed, 0 skipped. Pool seconds: MiniLM 47, mpnet 225, bge 220. Next: task 2 (bootstrap CIs), then task 3 (encoder diagnostics go/no-go).
- **2026-11-13 (Torch, cont.):**
  - Task 2 done: bootstrap CIs in `proxy.py` (with 7 new tests, 25/25 pass), and both baseline runners re-run with CI columns. Baselines reproduced exactly; LTS-lite shifted (§4).
  - Task 3: `scripts/encoder_diagnostics.py` and `scripts/hpc/cpu.sbatch` written and smoke-tested.
  - Running: nothing.
  - Blocked: the push of branch `hpc-torch-setup` (no GitHub credentials in the session; commit 82561cc is local). Today's task 2/3 work is uncommitted pending the LTS-lite CSV decision.
  - Next: submit the diagnostics job, write `docs/10_encoder_diagnostics.md` with go/no-go, propose the proxy embedding (task 4).
- **2026-11-14 (Torch):**
  - Task 3 done: diagnostics job 23156866 ran in 3 min; results in `results/encoder_diagnostics.csv`, write-up in `docs/10_encoder_diagnostics.md`.
  - Verdict: qualified go. Cluster-level overlap is fairly high (Jaccard 0.7–0.78; union of top-5 clusters reaches 91% of positives against 78–79% for bge alone). Query-level overlap is lower, especially for identity_hate (positives Jaccard ≈ 0.4). bge wins both ceilings.
  - Task 4 proposed: bge as the fixed proxy embedding.
  - Running: nothing.
  - Blocked: the humans' go/no-go and proxy-embedding confirmation; the push of `hpc-torch-setup`; the LTS-lite CSV decision. All work since 82561cc is still uncommitted.
  - Next: once confirmed, the equal-budget union checks, then the two-level bandit (task 5).
- **2026-11-14 (Torch, cont.):**
  - The human gave a go on the two-level bandit ("for now, let's see how it does").
  - Task 5 code: `hierarchical_lts` in `samplers.py` (hierarchical and flat, `share_labels`, `warm_start`, optional `trace`), 19 new tests, `scripts/run_bandit.py`. `lts_lite`'s within-cluster pick moved into a shared helper, unchanged in behavior.
  - Smoke test on a 20K-row pool subset (one seed, identity_hate, B = 1000) ran end to end; not evidence about performance.
  - Running: nothing. Next: submit `command.md` §5, then compare against LTS-lite per encoder with a paired bootstrap.
- **2026-11-14 (Torch, bandit results):**
  - Jobs 23168988 (bge proxy) and 23169002 (MiniLM proxy) finished in 34 min each: `results/bandit_<proxy>.csv`, 720 runs each.
  - Warm start is what matters. Threat F1 at B = 100 is 0.40–0.42 with warm start against 0.00–0.03 cold; at B = 1000 it is 0.43–0.45 against 0.17–0.38. Warm variants also have much lower seed spread (sd ≈ 0.02–0.03 against 0.12–0.18 on threat).
  - No evidence yet for the encoder level: rounds are split close to evenly across encoders (22–47%), and hierarchical and flat variants are within noise of each other.
  - Cold start: hierarchical with shared labels matches the best single-encoder LTS-lite and does not beat it.
  - `share_labels` finds more positives in all 8 hierarchical comparisons (e.g. warm, B = 1000, bge run: threat 127 vs 106, identity_hate 176 vs 117), with no clear F1 gain.
  - Caveats: LTS-lite got no warm start, so warm bandit vs LTS-lite is not a fair comparison; the same LTS-lite configuration found different positives in the two jobs (threat, MiniLM search, B = 1000: 19.7 vs 39.9), probably from k-means differing across nodes; all intervals are unpaired.
  - Running: nothing. Next: the controls and fixes listed under "Not done" in §4.
- **2026-11-14 (Torch, controls):**
  - Cached clusters: `load_clusters` in `data.py` computes k-means once and saves `data/jigsaw/emb/<model>/clusters_k50.npy`; `run_lts_lite.py`, `encoder_diagnostics.py` and `run_bandit.py` all load it and print its hash. The cache does not exist yet: the next job creates it.
  - New controls in `run_bandit.py`: `random`, `query/<encoder>`, `multi_query` (new sampler), `warm_random`, `lts_lite/<encoder>+warm` (75 query rows, the same number the bandit's warm start uses; via the new `warm_batch` option).
  - Method comparisons: `bootstrap_samples` and `mean_diff_ci` in `proxy.py` give a CI on the difference of seed-mean F1/AP that covers both test-set and sampler-seed noise; written to `results/bandit_pairs_<proxy>.csv`.
  - `run_bandit.py` now takes several proxy embeddings and samples once for all of them.
  - Tests: 48 pass. Smoke run on a 15K-row copy of the data ran end to end.
  - Running: nothing. Next: submit `command.md` §5 and read the pairs CSV.
- **2026-11-18 (Torch, controls results + encoder reward):**
  - Decisions from the humans: proxy embeddings bge and MiniLM (run both); bandit reward binary to start; `share_labels` keeps proportional credit.
  - Controls job 23547044 (85 min) done; cached clusters now exist (md5 MiniLM b1c36c14, mpnet 8cd206fc, bge 97fbd62e).
  - Result: the hierarchical bandit (share+warm) ties `lts_lite/bge+warm` at every budget on both predicates (paired CIs include 0); hierarchical vs flat and share vs noshare also tie. It beats `warm_random` at B = 1000 (+0.05 threat, +0.045 identity_hate F1) and single-encoder query search, so cluster search after the warm start helps.
  - Encoder choice matters but the bandit misses it: with warm start, LTS-lite on bge wins threat (F1 0.48) and on MiniLM wins identity_hate (0.47 F1, 0.50 AP; significantly above the bandit at B >= 500). The bandit gives identity_hate rounds mostly to mpnet.
  - Diagnosis: the binary encoder reward saturates once every encoder finds some positives, and it can actively favour an encoder with uniformly spread positives (shown in a synthetic test: ~12% of rounds to the richer encoder).
  - New: `hierarchical_lts(..., encoder_reward="rate")`: the encoder level gets r = positives / batch as a fractional win and 1 - r as a loss; clusters stay binary. `run_bandit.py` adds `hier_lts / share+{cold,warm}+rate`. Fixed a fixture-order bug in `test_mean_diff_ci_of_single_runs_is_the_paired_bootstrap`. Tests: 56 pass.
  - Caveat: in the synthetic test the rate reward moves rounds to the richer encoder (12% -> 54%) and finds 26% more positives, but it is still a weak signal (one fractional observation per round).
  - Running: nothing. Next: submit `command.md` §6; check whether `share+warm+rate` shifts identity_hate rounds to MiniLM and closes the gap to `lts_lite/MiniLM+warm`.
- **2026-11-18 (Torch, rate-reward results):**
  - Job 23607041 (2 h 44 min) done. All 1,152 runs shared with job 23547044 reproduce exactly (same F1 and positives): the pipeline is deterministic on the cached clusters.
  - `share+warm+rate` ties `share+warm` on F1 everywhere (bge proxy); AP is slightly lower on threat at B >= 500 (-0.03, significant). It still loses to `lts_lite/MiniLM+warm` on identity_hate at B = 1000 (-0.04 F1 bge proxy, -0.055 MiniLM proxy, both significant). Cold rate variants tie or slightly trail cold binary.
  - Encoder rounds stay near even (25-40% each). On identity_hate the rate reward stops favouring mpnet (48% -> 33% at B = 1000) but moves rounds to bge, not MiniLM.
  - Why it can't work: MiniLM's identity_hate advantage is not in positives found. The bandit finds more positives (167.5) than `lts_lite/MiniLM+warm` (152.6) yet gets lower F1. Rewarding positives cannot recover a gap that is about which positives make good proxy training data.
  - Running: nothing. Next: the humans' call on whether to stop iterating on the encoder level (see the recommendation in the session) and move to the screening sweep and write-up.
- **2026-11-18 (direction change):** The humans dropped the encoder-selecting bandit as the main method (kept as a negative result) and chose the **varied warm start** (several example sentences per predicate) as the next focus. Results and plan written up in `docs/11_bandit_results_and_next_directions.md`; README §7 updated. Next: implement the varied warm start with hand-written sentences, run against the single-sentence warm start at equal size.
