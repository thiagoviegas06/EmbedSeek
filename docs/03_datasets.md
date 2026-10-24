# Datasets

| Dataset | Role | Predicates | Access | Status |
|---|---|---|---|---|
| Jigsaw Toxic Comments | Naturally rare classes, human labels | `threat` (0.30%), `identity_hate` (0.88%), `severe_toxic` (1.0%) | HF mirror `thesofakillers/jigsaw-toxic-comment-classification-challenge` (ungated; original is Kaggle) | **Downloaded** to `data/jigsaw/raw/` |
| MTEB classification | Controlled pools at 20/5/1/0.5/0.1% prevalence, reproducible baseline | TBD (binary-izable task) | HF `mteb/*` datasets | TODO |
| BEIR (SciFact or FiQA) | Query–document relevance, extremely sparse positives | Query = predicate | HF `BeIR/scifact`, `BeIR/fiqa` + qrels | TODO |

## Rules from the proposal
- Splits (pool / validation / test) are fixed **before** changing class prevalence.
- Same candidate pool and label oracle for all samplers; 10 seeds.
- BEIR: use official judgments; unjudged docs are **not** automatic negatives. If negatives aren't defensible, report as acquisition-only.
- Report exact positive counts and uncertainty, especially at 0.1%.

## Next dataset to pull
BEIR SciFact is small (~5k docs, 300 test queries) so it's a quick second download. For MTEB, shortlist 2–3 candidate binary tasks and check positive counts.
