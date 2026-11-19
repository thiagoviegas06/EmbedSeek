# Varied warm start: several example sentences per predicate (2026-11-19)

**Summary:** replacing the single example sentence with five varied ones helps identity_hate and hurts threat.
- **Variety goes up, count goes down.** Five sentences find more varied positives but fewer of them in the 75 warm rows.
- **identity_hate:** the variety pays off, most at small budgets: up to +0.08 F1 at B = 100 with the bge proxy, and lower seed spread.
- **threat:** the lost positives cost 0.02–0.05 F1.
- **The best method at B = 1000 is unchanged** on both predicates.
- **docs/11's "variety beats count" needs a qualifier:** among warm start + follower methods, positives found still tracks F1 at B ≥ 500.

**Run:** Slurm job 23664198 (62 min on 8 CPUs), plan in [docs/11](11_bandit_results_and_next_directions.md) §3.1.
- Settings: cached k-means clusters (k = 50, same hashes as the bandit runs), 10 seeds, B ∈ {100, 250, 500, 1000}, proxy embeddings bge and MiniLM.
- Results: `results/warm_start_<proxy>.csv` (one row per run) and `results/warm_start_pairs_<proxy>.csv` (paired differences with 95% CIs over test rows and sampler seeds).
- Numbers below use the bge proxy unless noted. The MiniLM proxy agrees in direction and favours five sentences more on identity_hate.

**Reproduce:** `sbatch --time=04:00:00 scripts/hpc/cpu.sbatch scripts/run_warm_start.py data/jigsaw bge-base-en-v1.5 all-MiniLM-L6-v2`.

## 1. Setup

Every warm start labels 75 rows before a follower takes over. The (sentence, encoder) pairs take turns adding their most similar unlabeled rows (`multi_query`).

| Source | Sentences | Encoders | Rows per (sentence, encoder) | Same as |
|---|---|---|---|---|
| 1s-1enc | the original | the follower's own | 75 | `lts_lite/<encoder>+warm` in docs/11 |
| 1s-3enc | the original | MiniLM, mpnet, bge | 25 | the bandit's warm start |
| 5s-1enc | five | the follower's own | 15 | new |
| 5s-3enc | five | all three | 5 | new |

**Followers:**
- LTS-lite in bge (`lts_bge`).
- LTS-lite in MiniLM (`lts_minilm`).
- Random rows (`random`); its 1enc warm start uses bge.
- Query-only controls (`query/1s-3enc`, `query/5s-3enc`) fill the whole budget by query. They are deterministic, so there is one run each.

**The five sentences per predicate** are hand-written and comment-like; the text is in `scripts/run_warm_start.py`. The first is the original from docs/11.
- **threat:** direct (the original), veiled, against property, conditional, assault.
- **identity_hate:** race and religion (the original), sexual orientation, nationality, religion (Muslims), gender.

**New columns:**
- `warm_pos`: positives among the 75 warm rows.
- `pos_spread`: mean pairwise cosine distance between all positives found, in bge space. Higher means more varied positives.

**Reproduction check:** every run of `lts_bge/1s-1enc`, `lts_minilm/1s-1enc` and `random/1s-3enc` (2 predicates × 4 budgets × 10 seeds) matches `lts_lite/<encoder>+warm` and `warm_random` in `results/bandit_*.csv` exactly. That holds for F1 and positives found, with both proxies.

## 2. Results

### Five sentences trade count for variety

| | Warm positives, 1s-3enc | Warm positives, 5s-3enc | Spread at B = 100, 1s-3enc | Spread at B = 100, 5s-3enc |
|---|---|---|---|---|
| threat | 55 | 33 | 0.333 | 0.347 |
| identity_hate | 24 | 17 | 0.358 | 0.400 |

- Spread is the mean over the LTS-lite and random followers.
- With one encoder the drop is similar: threat 53–56 → 28–32, identity_hate 18–26 → 17–22.
- The four new threat sentences retrieve far fewer positives than the original: the warm start loses 40% of its threats.

### The effect depends on the predicate

The comparison is five sentences against one, everything else equal: same follower, same number of encoders. That gives 28 comparisons per predicate (7 follower × encoder settings × 4 budgets).

| | bge proxy, F1 | bge proxy, AP | MiniLM proxy, F1 | MiniLM proxy, AP |
|---|---|---|---|---|
| identity_hate: 5s better / tie / worse | **8** / 16 / 4 | 5 / 19 / 4 | **11** / 16 / 1 | 7 / 19 / 2 |
| threat: 5s better / tie / worse | 2 / 18 / **8** | 1 / 18 / **9** | 2 / 20 / **6** | 2 / 16 / **10** |

"Better" and "worse" mean the paired 95% CI excludes 0. By chance alone, fewer than one per side would.

**identity_hate**, test F1 (bge proxy):

| Follower | B | 1s-1enc | 1s-3enc | 5s-1enc | 5s-3enc |
|---|---|---|---|---|---|
| LTS-lite, bge | 100 | 0.334 | 0.307 | 0.327 | 0.349 |
| LTS-lite, MiniLM | 100 | 0.309 | 0.317 | 0.271 | **0.355** |
| Random | 100 | 0.307 | 0.306 | 0.332 | 0.354 |
| Query only | 100 | – | 0.258 | – | 0.338 |
| LTS-lite, bge | 1000 | 0.415 | 0.413 | 0.436 | 0.445 |
| LTS-lite, MiniLM | 1000 | **0.475** | 0.441 | 0.423 | 0.447 |
| Random | 1000 | 0.387 | 0.383 | 0.384 | 0.381 |
| Query only | 1000 | – | 0.397 | – | 0.445 |

- **At B = 100, 5s-3enc beats 1s-3enc for every follower:** +0.04 to +0.08 F1, all significant. With the MiniLM proxy the gains reach +0.11.
- **It also steadies the result.** The seed sd of F1 at B = 100 is 0.007–0.017 with 5s-3enc, against 0.015–0.046 with the 1s sources.
- **At B = 1000 the gain shrinks.** It survives for LTS-lite on bge (+0.03, significant against 1s-3enc) and for query-only search (+0.05).
- **Exception: five sentences in MiniLM alone hurt.** `lts_minilm/5s-1enc` is significantly worse than `lts_minilm/1s-1enc` at every budget (−0.04 to −0.07). Spreading the five sentences over three encoders rescues it (see "Sentences × encoders" below).

**threat**, test F1 (bge proxy):

| Follower | B | 1s-1enc | 1s-3enc | 5s-1enc | 5s-3enc |
|---|---|---|---|---|---|
| LTS-lite, bge | 100 | 0.406 | 0.433 | 0.394 | 0.384 |
| LTS-lite, MiniLM | 100 | 0.352 | 0.394 | 0.378 | 0.394 |
| Random | 100 | **0.440** | 0.436 | 0.409 | 0.403 |
| Query only | 100 | – | 0.339 | – | 0.419 |
| LTS-lite, bge | 1000 | **0.477** | 0.466 | 0.452 | 0.459 |
| LTS-lite, MiniLM | 1000 | 0.419 | 0.430 | 0.411 | 0.411 |
| Random | 1000 | 0.407 | 0.408 | 0.384 | 0.377 |
| Query only | 1000 | – | 0.390 | – | 0.367 |

- **Five sentences are worse or tied everywhere,** except query-only search at B ≤ 250 (+0.08 at B = 100, +0.04 at B = 250).
- **The losses are 0.02–0.05 F1.** They are significant mostly for the random follower (B ≥ 250 with three encoders) and for LTS-lite on MiniLM with three encoders (B = 250–500).
- **The random follower suffers most.** It adds almost no positives after the warm start (56 warm positives, 58 at B = 1000), so its proxy lives off the warm positives. Five sentences give it 33 instead of 55.

### The best method does not change

Best five-sentence configuration against the best one-sentence configuration at each budget (bge proxy):

| | B | Best 1s | F1 | Best 5s | F1 | 5s − 1s [95% CI] |
|---|---|---|---|---|---|---|
| threat | 100 | random/1s-1enc | 0.440 | query/5s-3enc | 0.419 | −0.021 [−0.064, +0.019] |
| threat | 250 | random/1s-3enc | 0.420 | query/5s-3enc | 0.419 | −0.001 [−0.046, +0.045] |
| threat | 500 | lts_bge/1s-3enc | 0.438 | query/5s-3enc | 0.432 | −0.005 [−0.040, +0.031] |
| threat | 1000 | lts_bge/1s-1enc | 0.477 | lts_bge/5s-3enc | 0.459 | −0.018 [−0.048, +0.013] |
| identity_hate | 100 | lts_bge/1s-1enc | 0.334 | lts_minilm/5s-3enc | 0.355 | +0.022 [−0.010, +0.057] |
| identity_hate | 250 | lts_minilm/1s-3enc | 0.370 | lts_bge/5s-3enc | 0.367 | −0.003 [−0.031, +0.027] |
| identity_hate | 500 | lts_minilm/1s-1enc | 0.422 | lts_minilm/5s-3enc | 0.400 | −0.022 [−0.049, +0.008] |
| identity_hate | 1000 | lts_minilm/1s-1enc | 0.475 | lts_minilm/5s-3enc | 0.447 | −0.027 [−0.057, +0.005] |

- **Every row ties.** Both winners are picked after the fact, so both sides are optimistic.
- **The one significant best-vs-best gap** is identity_hate at B = 100 with the MiniLM proxy: `lts_minilm/5s-3enc` 0.315 against `lts_bge/1s-1enc` 0.269, +0.047 [+0.010, +0.085].

### Sentences × encoders

docs/11 asked whether any gain comes from more sentences, more encoders, or both. This compares three encoders against one, with the follower and number of sentences fixed.

- **One sentence:** mostly ties. LTS-lite on MiniLM gains from three encoders on threat at B ≤ 250 (+0.03 to +0.04) but loses on identity_hate at B ≥ 500 (−0.03).
- **Five sentences:** three encoders matter for LTS-lite on MiniLM on identity_hate: +0.08, +0.05, +0.04 and +0.02 at B = 100 to 1000, the first three significant. Elsewhere it is mostly ties.
- **On identity_hate the gain needs both:** five sentences across three encoders. This fits docs/10, which found the encoders disagree most at the query level on identity_hate (the union finds 63% more positives than the best single encoder).

## 3. What this says about "variety beats count"

docs/11 hypothesised that the variety of the labeled positives matters more than how many there are. This run gives mixed support.

- **For:**
  - identity_hate at small budgets.
  - Query-only search on both predicates at B = 100. On threat, `query/5s-3enc` finds fewer positives (46 against 65) yet scores 0.42 against 0.34.
- **Against:** threat with any follower. Five sentences give more varied but fewer positives, and lose.
- **Across the 12 warm start + follower configurations** (rough, with 12 points per budget):
  - Positives found tracks F1 at B ≥ 500 on both predicates (Spearman ρ 0.6–0.8).
  - Spread tracks F1 only on identity_hate at B = 100 (ρ 0.5), and runs against it on threat at B ≤ 250 (ρ ≈ −0.5).

A better reading of both experiments: **variety helps when it does not cost many positives.**
- The docs/11 cases compared different method families: query search against LTS-lite.
- Within one family (warm start + follower), count still matters.

**Likely reason for the predicate split** (not yet tested):
- The original identity_hate sentence covers race and religion only. The label also covers sexual orientation, nationality, gender and more, so one sentence missed whole kinds of positives.
- The original threat sentence already describes the typical threat: 55 of its 75 warm rows are positive. The variants pull in fewer threats.
- This run cannot tell whether threat doesn't benefit from variety or the four extra threat sentences are simply weak.

## 4. Caveats

- **One hand-written set of sentences.** Per-sentence hit counts were not computed.
- **Equal split across sentences** (15 rows each with one encoder). An unequal split that keeps more rows for the original was not tried.
- **Query-only controls are deterministic** (one run each), so their CIs cover test-set noise only.
- **Best-vs-best picks both winners after the fact.**
- **Threat is close to its ceiling** (0.48), which leaves little room to separate methods, as in docs/11.
- **`pos_spread` is measured in bge space only.**

## 5. Next

1. **Per-sentence hit counts:** the positives each (sentence, encoder) pair adds to the warm start. This is analysis only (it reads labels) and cheap. It separates weak sentences from a real predicate effect.
2. **Mixed warm start:** keep most of the original sentence's rows and add the variants (for example 45 + 30), so threat keeps its positives while identity_hate gets its variety.
3. **Varied picks inside clusters** (docs/11 §3.2): unchanged in priority.
4. **LLM-generated sentences:** after 1 and 2.
5. **Third predicate, label efficiency and the screening sweep** (docs/11 §3.3–3.4), as planned.

**Open decision for the humans: which warm start do the next experiments use?**
- **5s-3enc:** more robust to one phrasing, best at small budgets on identity_hate, with lower seed spread. Ties at B = 1000; costs up to 0.05 F1 on threat.
- **1s:** best at B = 1000 on both predicates.
- The mixed warm start (item 2) may make the choice moot.
