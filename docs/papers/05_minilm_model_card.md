# [3] Sentence Transformers: `all-MiniLM-L6-v2` model card

**Source:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2. Local copy: [papers/minilm_model_card.md](../../papers/minilm_model_card.md)
**Role in our project:** the likely **default / reference encoder**. It's cheap, fast, and well known, and a candidate for the fixed proxy-feature embedding (§3.5). It's the "general-purpose Sentence Transformer" family in §3.1.

## Facts
| | |
|---|---|
| Output | **384-d**, mean pooling, **L2-normalized** (use cosine = dot product) |
| Base | `nreimers/MiniLM-L6-H384-uncased`: 6 layers, hidden 384 (~22M params) |
| Training | Contrastive (in-batch negatives, cross-entropy over cosine sims) on **1.17B sentence pairs**, 100k steps, batch 1024, TPU v3-8 |
| Train sequence length | **128 tokens** |
| Inference truncation | **256 word pieces** (default) |
| License | Apache-2.0 |
| Intended use | Sentence / short-paragraph encoding for retrieval, clustering, similarity |

**Training data mix (top sources):** Reddit comments 2015–18 (**726M pairs, about 62%**), S2ORC citations (~210M), WikiAnswers (77M), PAQ (64M), Stack Exchange (~70M), MS MARCO (9M), plus NLI, QA, captions, and others.

## Implications for EmbedSeek
1. **Truncation vs Jigsaw lengths.** The median Jigsaw comment is 205 chars (~50 tokens), fine. But p90 is 889 chars (~200+ tokens) and p99 is 3,444 chars, past 256 word pieces. **The model was trained on only 128 tokens**, so quality on long inputs is likely lower even before truncation. A threat buried at the end of a long rant gets cut off. Options: (a) accept and report, (b) chunk and max-pool, (c) prefer an encoder with longer context for the comparison. Log what fraction of positives exceed 256 tokens.
2. **Reddit-heavy training** (informal, conversational, sometimes toxic text) probably suits Wikipedia talk-page comments better than encoders trained mostly on QA or web search. That's a hypothesis to test in RQ1, not an assumption.
3. **It's a symmetric similarity model** (pairs of similar texts), not an asymmetric query→document retriever. So **query-seeded retrieval (sampler b)** with a short predicate like *"contains a credible threat"* may match poorly, because a predicate isn't phrased like a comment. Retrieval-tuned encoders (e.g. e5/bge with `query:` / `passage:` prefixes) are built for this asymmetry. **This is one concrete reason the two families could rank items differently**, which is the complementarity our hypothesis needs.
   - Cheap mitigation to test: write the query as a *pseudo-example* ("I will kill you") instead of a description. Treat it as an ablation.
4. **Speed and cost:** 384-d × 160K comments × float32 ≈ 245 MB (float16 ≈ 123 MB). It's one of the fastest sentence encoders, so cold-start `C_embed` will be small. That makes it a good baseline for the cost-vs-benefit question (RQ4).
5. **Comparison with what the papers used:**
   - Chung et al.: Gecko `text-embedding-005` (768-d, proprietary API). They found the *cheapest* open model (EmbeddingGemma-300m) was consistently worst. MiniLM could be similarly weak for the *proxy*, so check before fixing it as the reference.
   - AnchorAL: `all-mpnet-base-v2` (768-d), the bigger sibling with the same training recipe.
   - SEALS (text appendix): SBERT-NLI-base. Generic NLI training left positives scattered (mean cosine 0.004).

## Candidate encoder set for the Week 3 feasibility check (suggestion)
| Family | Candidate | Dim | Why |
|---|---|---|---|
| General sentence (symmetric) | `all-MiniLM-L6-v2` | 384 | cheap baseline, Reddit-heavy |
| General sentence, larger | `all-mpnet-base-v2` | 768 | used by AnchorAL; same recipe, more capacity |
| Retrieval-tuned (asymmetric) | `BAAI/bge-base-en-v1.5` or `intfloat/e5-base-v2` | 768 | query/passage asymmetry, for sampler (b) |
| (optional) Domain-adapted | a toxicity-tuned encoder | — | §3.1 mentions "adapted to a domain"; but check it wasn't trained on Jigsaw (label leakage) |

Check exact model names, licenses, and max lengths on Hugging Face before committing. These are starting points, not verified choices.
