# [2] Barbosa et al. 2025: LTS (Learn to Sample) for Wildlife Trafficking Ads

**Citation:** J. Barbosa, U. Gondhali, G. Petrossian, K. Sharma, S. Chakraborty, J. Jacquet, **J. Freire** (NYU VIDA). *Proc. ACM Manag. Data* 3(3), Art. 119, SIGMOD 2025. doi:10.1145/3725256. arXiv:2504.21211.
**PDF:** [papers/barbosa2025_wildlife.pdf](../../papers/barbosa2025_wildlife.pdf). **Code:** https://github.com/VIDA-NYU/LTS
**Role in our project:** the closest prior method. It is a **multi-armed bandit for finding rare positives**, which is what we propose too. We must clearly separate our contribution from it.

> Note: this is an NYU VIDA lab paper (Juliana Freire's group). Expect reviewers or instructors to know it well.

## One-line summary
Wildlife ads are a needle in a haystack (sometimes ≤5% relevant). LTS clusters the ads once, then uses **Thompson sampling over clusters** to decide where to draw the next batch. It labels the batch with GPT-4 (few-shot), fine-tunes BERT, and rewards the cluster if validation F1 improved. The final BERT classifier matches or beats GPT-4 zero-shot at ~1% of the cost.

## Problem definition (§3.2)
Collection `D` with positive ratio `k = N_pos/N_neg`. Choose a sample `S` with `S_pos/S_neg ≫ k` that is also **diverse and representative**. Diversity matters for *both* classes. Shark positives include teeth, jaws, and taxidermy; shark negatives include plush toys, stamps, and books.

## The LTS algorithm (§3.4, Alg. 1)
1. **Cluster** all ads once, using topic modeling.
2. **Thompson sampling over clusters:** Beta(α + wins, β + losses) per cluster; pick the argmax draw. Wins and losses are multiplied by a **decay δ = 0.99** each round (non-stationary rewards).
3. **Sample N = 200** from the chosen cluster: random in round 1, then use the current model to **prioritize predicted positives** up to a cap, filling the rest with predicted negatives.
4. **LLM pseudo-label** with GPT-4, using a few-shot prompt with rationales (Fig. 3).
5. **Fine-tune** `bert-base-uncased` (multilingual for AP). **Reward = 1 if validation F1 beats the previous best** (initial baseline 0.5), else 0, and revert to the previous model.
6. Repeat until the budget is exhausted (~2,000–4,000 labels).

## Experiments
| Task | # ads | Gold eval |
|---|---|---|
| Shark products (SP) | 14.5K | ~500 expert-labeled |
| Leather products (LP) | 152K | ~500 |
| Animal products (AP) | 700K | ~500 |

**Baselines:** Random (RS), Knowledge-based keyword sampling (KBS), Active learning with greedy coreset (ACTL, small-text + BERT).

**Results (text model F1, Table 2):**
| | RS | KBS | ACTL | LTS |
|---|---|---|---|---|
| Animal | 0.745 | 0.685 | 0.663* | **0.816** |
| Leather | 0.465 | 0.776 | 0.822 | **0.855** |
| Shark | 0.774 | 0.780 | **0.924** | 0.872 |
\*ACTL timed out after 72h with only 3K labels.

- **KBS (keyword sampling) gets high precision but low recall.** Keywords bias the sample toward a subset of positives. → This is a direct argument for *diversity*: a query-seeded sampler (our sampler b) risks the same failure.
- ACTL's coreset method needs pairwise distances, so it's O(n²). It didn't scale to 700K. LTS clusters once and runs inference only inside the chosen cluster.
- LTS-text (110M params) vs GPT-4 zero-shot: 0.855 vs 0.850 (leather), 0.873 vs 0.946 (shark).
- **Cost:** ~2,000 GPT-4 calls ≈ $40 per task, vs $320–$14,000 to label everything with GPT-4.
- **Labeler ablation (Table 6):** a weaker LLM gives much worse labels and much worse classifiers (GPT-4 0.82–0.87 vs Llama-8B 0.40–0.62). Label quality is critical.
- **Runtime (Table 7):** clustering is <4% of the time. **Model inference over the selected cluster is the bottleneck** (AP: ~107K s for 10 iterations).

## How EmbedSeek differs (important for our writeup)
| | LTS (Barbosa) | EmbedSeek |
|---|---|---|
| Bandit arms | **Clusters** within one representation | **Embedding spaces** (whole representations) |
| Reward | Binary: did validation F1 improve? Needs a gold validation set and retraining every round | Positives found (+ λ·novel types) per label. No retraining needed per round |
| Representation | One topic model, fixed | 2–3 frozen pretrained encoders, compared |
| Within-arm selection | Model-predicted positives (needs inference over the cluster) | Query similarity / positive-anchor kNN (cheap, index lookups) |
| Classifier | Fine-tuned BERT | LR on a fixed reference embedding |
| Evaluation of diversity | Implicit (P/R balance) | Explicit, measured in a shared independent space |

**Proposal framing:** our motivation states that Barbosa's "repeated sampling and retraining may impose substantial overhead". Table 7 supports this: inference over clusters dominates runtime.

**Possible combo (stretch idea):** arms = (embedding × cluster) pairs, or a hierarchical bandit that picks an embedding first and then a cluster inside it.

## Implemented as a competing baseline: LTS-lite
`embedseek.samplers.lts_lite` / `scripts/run_lts_lite.py`. Deviations: reward = batch found a positive (not val-F1 gain), LR (not BERT), batch 25, k-means k = 50. First results (MiniLM, Jigsaw): random-like until it hits a positive cluster, then strong; best identity_hate F1 at B = 1000 (0.42) but high variance. See [../06_baseline_results.md](../06_baseline_results.md).

## Things to borrow
- **Thompson sampling with Beta priors and decay** is a ready-made bandit template for our §3.3. The decay handles rewards that drop as an arm's neighborhood gets exhausted, which will happen with anchor search.
- Their future work explicitly says: *"Using similarity measures for selecting instances as well as core-sets for attaining diversity are directions we plan to pursue"* and *"explore if alternative embedding-based clustering strategies can improve the target classification models."* **Our project is close to that stated future work.** Cite this.
- The comparison of random vs keyword vs AL vs bandit is a good template for our baseline table.

## Questions / caveats
- The reward uses a *gold validation set* for every round. That's realistic for them, but it means LTS isn't purely LLM-driven. Our reward avoids this.
- They report no seeds or CIs, and the ~500-ad gold sets are small.
- Is the topic-model clustering itself a strong single representation? It could serve as a "cluster-coverage" baseline (our sampler c).
