# Bandit results and next directions (2026-11-18)

**Summary:** the two-level bandit (encoder → cluster) does not beat single-encoder LTS-lite once both get the same query warm start. The warm start is the method's biggest lever. The results point to a different bottleneck: the *variety* of the labeled rows, not how many positives they contain. The next phase targets that, starting with a more varied warm start.

**Runs:** Slurm jobs 23547044 (controls) and 23607041 (adds the positives-per-batch encoder reward). Both use the cached k-means clusters (k = 50), 10 seeds, B ∈ {100, 250, 500, 1000}, and two proxy embeddings (bge, MiniLM). Every run shared by the two jobs reproduces exactly. Results: `results/bandit_<proxy>.csv` (one row per run) and `results/bandit_pairs_<proxy>.csv` (paired differences with 95% CIs over test rows and sampler seeds). All numbers below use the bge proxy; the MiniLM proxy gives the same picture.

## 1. What the warm start is

Before any bandit round, label 75 rows chosen by similarity to **one hand-written example sentence** per predicate:

| Predicate | Example sentence |
|---|---|
| threat | "I am going to find you and kill you." |
| identity_hate | "people of your race and religion are disgusting and should all be gone" |

Each encoder (MiniLM, mpnet, bge) embeds the sentence, and the encoders take turns adding their 25 most similar unlabeled rows (`multi_query` in `src/embedseek/samplers.py`). The 75 labels count against the budget and update the bandit's statistics like any other round. The matched controls get the same 75 rows: `warm_random` (then random rows) and `lts_lite/<encoder>+warm` (75 rows from that encoder alone, then LTS-lite).

| | Positives in the first 75 labels |
|---|---|
| Random rows | ≈ 0.2 threats |
| Warm start | ≈ 55 threats, ≈ 25 identity_hate |

The LTS paper [Barbosa et al. 2025] has no warm start: its first round samples a cluster at random. That works at its prevalence (≈ 5% or more) but not at ours (0.3% threat, ≈ 0.9% identity_hate). Its keyword-sampling baseline is the closest analogue, and it showed the same weakness as our pure query search: high precision, low recall.

## 2. Results

### What works

**The warm start.** At B = 100, threat F1 goes from ≈ 0.02 (cold) to ≈ 0.41 (warm), against a ceiling of 0.48 with all 143K pool labels. Warm variants also have much lower seed spread.

**Cluster search after the warm start.** The bandit with shared labels and warm start beats:
- warm start + random rows at B = 1000 (+0.05 F1 threat, +0.045 identity_hate; both significant);
- query search on bge alone at every budget on threat (+0.09 to +0.29) and at B ≤ 500 on identity_hate (+0.05 to +0.07).

### What does not work

**The encoder level.** Paired CIs include 0 at every budget and on both predicates for:
- the bandit vs LTS-lite on bge with the same warm start;
- hierarchical vs flat (every encoder-cluster pair as one arm);
- shared vs unshared labels (sharing finds more positives, but F1 doesn't improve).

**The encoder choice matters, but the bandit can't learn it.** With a warm start, LTS-lite on **bge** is best for threat (F1 0.48) and LTS-lite on **MiniLM** is best for identity_hate (F1 0.47, AP 0.50; significantly above the bandit at B ≥ 500). The bandit splits rounds roughly evenly across encoders, and on identity_hate it favours mpnet, the weakest encoder there.

**Rewarding positives per batch doesn't fix it.** `encoder_reward="rate"` ties the binary reward on F1, has slightly lower threat AP at B ≥ 500 (−0.03), and still loses to LTS-lite on MiniLM for identity_hate at B = 1000 (−0.04 F1). It moved identity_hate rounds away from mpnet but toward bge, not MiniLM.

### Why: variety beats count

| identity_hate, B = 1000 | Positives found | F1 |
|---|---|---|
| Bandit, shared labels, warm start, rate reward | **167.5** | 0.44 |
| LTS-lite on MiniLM, warm start | 152.6 | **0.47** |

| threat, B = 1000 | Positives found | F1 |
|---|---|---|
| Query search on bge alone | **218** | 0.37 |
| LTS-lite on bge, warm start | 114 | **0.48** |

More positives do not mean a better proxy. Two likely reasons, not yet tested directly:
1. **Redundant positives.** Search keeps labeling from the same rich pocket. Similar positives teach the proxy less than varied ones, so it misses the kinds it never saw.
2. **Unrepresentative negatives.** The proxy also learns from the rows labeled negative. If every negative comes from near the positives, it never sees typical comments and misjudges them on the full table. Warm start + random rows doing well on threat supports this.

Any reward that counts positives (binary, rate, or per row) optimises the wrong target, so it can't pick the encoder whose labels make the better proxy. LTS's own reward ("did validation F1 improve after retraining?") measures the right thing, but it needs a labeled validation set and a retrain every round.

**Caveats:**
- The gaps are small (≈ 0.03–0.05 F1), though significant under paired CIs.
- Threat is close to its ceiling: most warm methods score 0.41–0.48 against 0.48, which leaves little room to separate methods.
- One example sentence per predicate; docs/06 found results depend heavily on the wording.

## 3. Next directions

### 3.1 A more varied warm start (done)

**Results: [docs/12](12_varied_warm_start.md).** Five sentences help identity_hate (most at small budgets) and hurt threat; the best method at B = 1000 is unchanged.

Replace the single example sentence with **several different example sentences per predicate**. For threat, for example: direct threats, veiled threats, threats against property or family, conditional threats. Each sentence gets a share of the warm-start rows.

Why this first:
- It builds on the part that works.
- It targets the diagnosed weakness: one sentence returns near-copies of one kind of positive, the same high-precision, low-recall failure as keyword sampling in the LTS paper.
- It keeps the multi-encoder idea where it helps. Encoders disagree most at the **query** level: the union of the three encoders' query results finds 63% more identity_hate positives than the best single encoder (docs/10). That complementarity was lost inside the bandit but can be used in the warm start.
- It also tests robustness to one lucky phrasing.

Open choice: where the example sentences come from.
- **Hand-written** by the team: quick and reproducible. Start here.
- **LLM-generated** from the predicate description: closer to a real deployment, needs an LLM call. A natural follow-up.

Questions for the experiment:
- Does a varied warm start beat the single-sentence warm start at equal size (75 rows)?
- Does it help most at small budgets, where the warm start is a large share of all labels?
- Is the gain from more sentences, more encoders, or both? (Grid: 1 vs several sentences × 1 vs 3 encoders.)

### 3.2 Pick varied rows inside each cluster

Inside a chosen cluster, the sampler labels the rows the current proxy is **most confident** are positive (`_pick_from_cluster` in `samplers.py`). Those tend to repeat what is already labeled. Alternatives:
- **Uncertainty:** label rows the proxy is least sure about (classic active learning).
- **Distance:** label rows far from those already labeled.
- **Random mix:** reserve a few rows of each batch for random rows from the whole table, for representative negatives.

A small code change in one function, aimed directly at the diagnosed problem.

### 3.3 Show where gains can appear

- **Add a predicate with more headroom.** Jigsaw also has `severe_toxic`, `insult` and `obscene`, at different prevalences.
- **Report label efficiency:** the budget needed to reach 90% of the ceiling F1. Methods can differ clearly there even when final F1 ties.

### 3.4 Kept from the plan

- **Screening sweep** (task 6, docs/07 "Idea 2"): embed only a fraction s of rows with the extra encoders; cost from `meta.json`. Independent of the bandit outcome.
- **Proxy-improvement reward** (optional): reward an encoder by how much its batch improves the proxy, as LTS does. The one remaining way to rescue the encoder level, at the cost of a validation set and retraining per round.

### 3.5 Dropped

The encoder-selecting bandit as the main method. It is reported as a negative result: the best encoder depends on the predicate, and a reward that counts positives can't detect which one makes the better proxy.

## 4. Plan to Dec 10

| When | Work |
|---|---|
| Week of Nov 18 | Varied warm start (3.1); varied in-cluster picks (3.2). One CPU job, same controls and paired CIs |
| Week of Nov 25 | Third predicate (3.3); screening sweep (3.4) on the best method |
| Week of Dec 2 | Write-up |
