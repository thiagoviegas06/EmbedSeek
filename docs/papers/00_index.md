# Paper notes: index and synthesis

The proposal originally cited 6 references, but [4] and [6] were the same paper (AnchorAL). They're now merged into [4]; see [../proposal_reference_fixes.md](../proposal_reference_fixes.md). PDFs and extracted text are in [papers/](../../papers/).

| # | Notes | Paper | What we take from it |
|---|---|---|---|
| [1] | [01_chung2026_proxy_models.md](01_chung2026_proxy_models.md) | Chung et al., SIGMOD '26: proxy models for AI.IF | The **baseline pipeline** (embed → sample → LLM-label → LR); the needle failure we target; the cost breakdown |
| [2] | [02_barbosa2025_lts_wildlife.md](02_barbosa2025_lts_wildlife.md) | Barbosa et al., SIGMOD '25: LTS | **Closest prior method**: Thompson-sampling bandit over *clusters*. We must separate ourselves from it |
| [5] | [03_coleman2022_seals.md](03_coleman2022_seals.md) | Coleman et al., AAAI '22: SEALS | kNN-of-labeled candidate pools; **graph diagnostics** that predict success; text failure with generic SBERT |
| [4] | [04_lesci2024_anchoral.md](04_lesci2024_anchoral.md) | Lesci & Vlachos, NAACL '24: AnchorAL | Diverse anchors to avoid getting stuck; default hyperparameters for sampler (d); used **Jigsaw (WikiToxic)** |
| [3] | [05_minilm_model_card.md](05_minilm_model_card.md) | all-MiniLM-L6-v2 model card | Default encoder; 128-token training / 256 truncation; symmetric (not query-tuned) |

## How the papers fit together
```
Chung [1]:   embed ─► RANDOM sample ─► LLM label ─► LR proxy      ← fails when positives are rare (§5.4, §6.2)
                          │
     replace the sampler with something that finds rare positives:
                          ▼
Barbosa [2]: bandit over CLUSTERS (one representation), reward = val-F1 gain, retrains BERT each round
SEALS [5]:   kNN of labeled set in ONE embedding, grows pool          ← traps; text + generic SBERT failed
AnchorAL[4]: kNN of DIVERSE ANCHORS in ONE embedding, fixed subpool   ← fixes trapping; still one encoder
                          ▼
EmbedSeek:   bandit over EMBEDDING SPACES; inside each arm: query / anchor / diversity sampling;
             reward = new positives (+ diversity) per label; same LR proxy as [1]; human-gold test
```

## Recurring findings that shape our design
1. **Random fails at <1% prevalence.** All four papers show this. It's our baseline (a) and should lose clearly.
2. **Pure coverage/diversity sampling fails** at extreme rarity: SEALS found k-centers never beat random, and AnchorAL notes diversity strategies fail when classes overlap. → Sampler (c) should be *diversity among likely positives*, not global coverage.
3. **Pure similarity sampling is biased:** Chung found Top-K raises imbalance, and Barbosa found keyword sampling gives low recall. → Query-seeded (b) alone will probably find *one kind* of positive. This is why we measure diversity.
4. **Neighbor search works when positives form connected, tight components** in the embedding (SEALS). That's a property of the *representation*, which is our RQ1 hypothesis.
5. **The embedding matters:** Chung shows different proxy F1 across encoders, and SEALS's only text experiment failed with generic SBERT. Nobody varies the *sampler's* embedding adaptively. **That's our gap.**
6. **Labeler quality matters:** in Barbosa, GPT-4 vs Llama-8B labels give 0.82 vs 0.40 F1, and Chung's proxy is capped by LLM accuracy. → Our real-LLM phase should report label agreement with human gold.
7. **Overhead is real:** Chung measured Top-K / AL sampling at 4–5 orders of magnitude slower than random (serial), and Barbosa found inference over clusters dominates runtime. → RQ4 cost accounting is necessary, not optional.

## Proposal fixes found while reading
- ~~Merge refs [4] and [6]~~ Done: see [../proposal_reference_fixes.md](../proposal_reference_fixes.md) (still needs applying in the proposal source).
- Consider citing Barbosa's future-work sentence ("Using similarity measures for selecting instances … are directions we plan to pursue") as direct motivation.
- Consider stating evaluation at **natural prevalence** (AnchorAL's WikiToxic test set was 55% positive, so its numbers aren't comparable).

## Possible additional reading (cited by these papers)
- **UQE** (Dai et al., NeurIPS 2024): the original embedding-proxy + active-learning query engine that Chung builds on.
- **LOTUS** (Patel et al., 2024): semantic operators baseline; Chung compares to it.
- **Active search** (Garnett et al., ICML 2012; Jiang et al., 2017–2019): the formal "maximize positives found" objective. That's what our acquisition metric really is.
- **Chapelle & Li 2011**, *An empirical evaluation of Thompson sampling*: Barbosa's justification for Thompson sampling over UCB.
