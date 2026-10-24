# Step-by-step plan

Today is in Week 2 (Oct 11–24 = Weeks 1–2). Status marks: [x] done, [ ] todo.

## Weeks 1–2 (Oct 11–24): Understand problem + datasets
- [x] Read the proposal PDF
- [x] Download Jigsaw Toxic Comments, profile it → [04_jigsaw_exploration.md](04_jigsaw_exploration.md)
- [x] Read the 5 cited papers; notes in [papers/00_index.md](papers/00_index.md)
- [ ] Team discussion of the notes, especially how we differ from Barbosa's LTS bandit
- [x] Write corrected reference list → [proposal_reference_fixes.md](proposal_reference_fixes.md)
- [ ] Apply it in the proposal source (one in-text change: AnchorAL [6] → [4])
- [ ] Download + profile MTEB classification candidates (e.g. toxic_conversations, tweet_sentiment, emotion) and BEIR SciFact/FiQA → [03_datasets.md](03_datasets.md)
- [ ] Feasibility audit: pick the final 3 datasets and 2–3 predicates each. Need enough positives in the *test* set to measure F1 (rule of thumb: ≥50).
- [ ] Pick encoder checkpoints (Week-1 feasibility check in proposal): check speed on this laptop (Apple silicon, `mps`) for embedding ~160k comments.
- [ ] Agree on environment: Python venv with pandas, scikit-learn, sentence-transformers, umap-learn, faiss-cpu, matplotlib.

## Done early (Oct 24): harness + first baselines
- [x] venv + `requirements.txt`; `embedseek` package (`src/`) with data / proxy / samplers
- [x] Frozen splits with dedup + test-leakage removal → `data/jigsaw/splits/`
- [x] MiniLM embeddings cached; ceiling / random / query baselines → [06_baseline_results.md](06_baseline_results.md)
- [x] **Competing baseline LTS-lite** (Barbosa et al.: Thompson sampling over k-means clusters, one embedding) → same doc
- [x] Results figure → `results/figures/baselines_minilm.png`

## Weeks 3–4 (Oct 25–Nov 7): Embedding exploration (RQ1)
- [ ] Create disjoint train/pool, validation, test splits **before** any prevalence manipulation; save split ids to disk.
- [ ] Embed the pool with each encoder; cache to `.npy` (float16 ok); time each (cost term `C_embed`).
- [ ] Build FAISS index per encoder.
- [ ] Diagnostics per encoder, per predicate: top-k positive enrichment, neighborhood purity (kNN label agreement), overlap (Jaccard) of top-k between encoders, positives per cluster.
- [ ] PCA/UMAP side-by-side plots (exploratory only; sampling uses full-dim vectors).
- [ ] Draft the experiment harness design: `Oracle`, `Sampler` interface, `run(sampler, budget, seed)`.

## Week 5 (Nov 8–14): Go / no-go on the bandit
- [ ] Do embeddings disagree enough (low overlap, different yields)? If all return the same neighbors the bandit has nothing to choose between.
- [ ] Prototype reward (positives only vs positives+diversity) on one dataset.
- [ ] Decide: proceed, refine, or pivot. Write the decision in `docs/decision_week5.md`.

## Weeks 6–7 (Nov 15–28): Implement and run
- [ ] Implement samplers (a)–(d) and the bandit; fixed seeds (10) and budgets.
- [ ] Rerun LTS-lite in every embedding and dataset; key comparison: bandit over clusters (LTS-lite) vs bandit over embeddings (ours)
- [ ] Build controlled pools at 20%, 5%, 1%, 0.5%, 0.1% positives from MTEB data.
- [ ] Train the shared logistic-regression proxy, tune threshold on validation, score on test.
- [ ] Bounded real-LLM labeling run on a subset; measure agreement with human labels.
- [ ] Ablate reward variants; record failures.

## Week 8 (Nov 29–Dec 5): Results
- [ ] Figures: (1) PCA/UMAP + neighborhood stats, (2) positives & subtype coverage vs labels, (3) arm allocation vs rounds, (4) test F1 vs budget, (5) performance vs cost.
- [ ] Cost table using `C_adaptive` formula; cold-start vs cached.
- [ ] Confidence intervals across seeds.

## Dec 6–10: Report, slides, submit (Dec 10)
- [ ] Report, presentation, reproducible repo (splits, scripts, configs).

## Suggested split of work (adjust as you like)
- Thiago: data/splits/oracle harness, Jigsaw
- Deep: embeddings, FAISS, diagnostics/visualization
- Prashanth: samplers + bandit, proxy training/eval
