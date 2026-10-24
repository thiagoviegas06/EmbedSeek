# [4] Lesci & Vlachos 2024: AnchorAL (Computationally Efficient Active Learning for Large and Imbalanced Datasets)

**Citation:** P. Lesci, A. Vlachos (Cambridge). NAACL 2024, pp. 8445–8464. https://aclanthology.org/2024.naacl-long.467/
**PDF:** [papers/lesci2024_anchoral.pdf](../../papers/lesci2024_anchoral.pdf). **Code:** github.com/pietrolesci/anchoral. **Data:** huggingface.co/collections/pietrolesci/anchoral
**Role in our project:** the second inspiration for **sampler (d) positive-anchor search**. Its fix for SEALS's "trapping" problem gives us the design of our exploration fraction.

> Originally cited twice in the proposal ([4] and [6]). Merged into [4] with the author name fixed; see [../proposal_reference_fixes.md](../proposal_reference_fixes.md).

## One-line summary
At each active learning round, pick a few **diverse labeled "anchors" per class** (k-means++ seeding). Retrieve each anchor's K nearest unlabeled neighbors and keep the top-M by average similarity as a **small, fixed-size subpool**. Run any AL strategy on that subpool only. It's faster than SEALS and RandSub, finds more minority instances, and gives better minority F1.

## Two problems it targets (§2.2–2.3)
1. **Path dependence / "missed cluster effect":** with an imbalanced seed set, uncertainty sampling keeps refining the *initial* decision boundary and never explores. If seed positives come from one cluster, the budget can run out before other positive clusters are found. **This is the same concern as our "positive-subtype diversity" metric.**
2. **SEALS's subpool grows** (k per labeled item) and gets redundant. After a few rounds it barely changes, so exploration stalls.

## Algorithm (Alg. 1)
```
for t in 1..T:
    f_t = train(D_t)
    A_t = ∪_c kmeans++_seed({x in D_t with y=c}, A)   # A anchors per class, diversity-seeded
    N_t = index.kNN(A_t, U_t, K)                        # K neighbors per anchor
    score(x) = mean similarity to the anchors that retrieved x
    Ũ_t = top-M by score                                # fixed-size subpool
    B_t = Φ(f_t, Ũ_t, b)                                # AL strategy on the subpool only
    label B_t; D_{t+1} = D_t ∪ B_t
```
- **Defaults:** A = 10 anchors per class, K = 50, M ≤ 1k–10k, b = 25 per round, budget 5k, seed = 100 (5 per minority class).
- **Encoder for the index:** `all-mpnet-base-v2` (sentence-transformers), HNSW index (hnswlib), cosine similarity. It's kept fixed while the *classifier* is fine-tuned (BERT, ALBERT, DeBERTa, GPT-2, T5, BERT-tiny).
- Anchors come from *both* classes. Majority anchors also pull in hard negatives near minority regions.

## Ablations (Table 3, Amazon-Agri, minority F1-AUC)
| Change | Minority | Time |
|---|---|---|
| default (k-means++ anchors, A = 10, K = 50) | **73.5** | 4.1 |
| entropy-selected anchors | 71.4–71.5 | 12.7 |
| A = 50 / 100 | 72.6 / 72.2 | 14.7 / 24.8 |
| K = 500 / 5k | 70.7 / 68.3 | 30.1 / 39.0 |
| **no anchoring** | **33.6** | 3.4 |
- **More anchors or more neighbors hurt.** A bigger subpool fills with majority-class items. Model-agnostic (k-means++) anchor choice beats model-driven choice because it avoids propagating model bias.
- Anchoring is the whole effect: removing it more than halves minority F1.

## Results (Table 2, BERT-base + Entropy)
- AnchorAL beats SEALS and RandSub on minority F1 on 3 of 4 tasks, with selection time of **minutes vs hours** (e.g. WikiToxic 2.6 vs SEALS 60.9 vs RandSub 20.4).
- It **finds more minority instances at every iteration** (Fig. 3) and its subpools contain more minority items (Fig. 4).
- The exception is Agnews-Bus, where SEALS is slightly better. The authors suspect their artificial downsampling put minority items near the initial boundary, so exploration *hurt* there. → **Caution for our controlled-prevalence pools:** downsampling can create unrealistic geometry.

## Datasets (App. B) and the Jigsaw connection
| Task | Train pool | Minority % | Note |
|---|---|---|---|
| Amazon-Agri | AmazonCat-13k | 0.09% | natural rarity |
| Amazon-Multi | AmazonCat-13k | 0.09–0.78% | multiclass |
| **WikiToxic** | 115,881 | **1% (downsampled)** | **= Kaggle Jigsaw Toxic Comments**, binary toxic. HF: `OxAISH-AL-LLM/wiki_toxic` |
| Agnews-Bus | 90,910 | 1% (downsampled) | |
- **Their WikiToxic test set is 55% toxic.** They downsampled only the training pool, not the test set. → F1 on a balanced test set is not comparable to F1 at natural prevalence. **We should evaluate at natural prevalence**, which our proposal implies. We should state this explicitly, and possibly also report their setting for comparison.
- They use the binary `toxic` label (9.6% natural). We target `threat` (0.30%) and `identity_hate` (0.88%), which are rarer and naturally rare. That's a harder, more realistic setting than theirs.

## How EmbedSeek relates
| | AnchorAL | EmbedSeek |
|---|---|---|
| Goal | Classifier quality (active learning) | Positive *discovery* + diversity (active search), then proxy F1 |
| Embedding | **One fixed encoder** (all-mpnet-base-v2) | Several encoders; the sampler picks |
| Exploration | Diverse anchors via k-means++ | Exploration fraction + bandit over spaces (+ diverse anchors) |
| Oracle | Perfect (gold labels) | Gold labels first, then a real LLM |
| Evaluation | F1-AUC over iterations, 8 runs, median + IQR | Positives per 100 labels, diversity, proxy F1, cost; 10 seeds |

## Things to borrow
- **Sampler (d) design:** k-means++ to pick A ≈ 5–20 anchors from found positives (plus a few negatives), K ≈ 20–50 neighbors each, ranked by mean similarity. Use their hyperparameter tips directly as defaults.
- `all-mpnet-base-v2` is a natural **second encoder** next to `all-MiniLM-L6-v2`: same family but larger (768-d vs 384-d). For real diversity, add a retrieval-tuned family too (e.g. bge or e5).
- Their **F1 learning-curve AUC** (trapezoidal) is a single-number summary of "F1 vs budget" across budgets.
- Reproducibility: separate RNGs for each source of randomness (init, ordering, seed selection, subsampling).

## Caveats
- They assume a perfect oracle (acknowledged in Limitations). Our LLM-labeling phase deals with noise.
- Their full-model fine-tuning makes RandSub stronger than SEALS reported. Our fixed LR proxy is closer to SEALS's setup (LR on embeddings).
