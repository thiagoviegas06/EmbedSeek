# Proposal reference fixes

The proposal PDF has no source file in this repo. Apply these edits in the source (Overleaf / Docs) and re-export.

## What changed
- **[4] and [6] were the same paper (AnchorAL).** I kept [4], corrected it, and deleted [6].
- Fixed [4]: first author is **Pietro** Lesci (old [6] said "G."), and added the missing closing quote after the title.
- Renumbering is minimal: [1], [2], [3], [5] stay the same, so only **one in-text citation** changes.

## In-text edit (§1, Motivation and Research Gap, 2nd paragraph)
```diff
- Related approaches such as SEALS [5] and AnchorAL [6] exploit neighbors around informative labeled examples.
+ Related approaches such as SEALS [5] and AnchorAL [4] exploit neighbors around informative labeled examples.
```
No other passage cites [4] or [6].

> Side note: [3] (MiniLM model card) is listed but never cited in the text. Either cite it in §3.1, where you mention "a general-purpose Sentence Transformer", or drop it.

## Corrected reference list
[1] Y. Chung et al. "100x Cost & Latency Reduction: Performance Analysis of AI Query Approximation using Lightweight Proxy Models." *Proceedings of the ACM on Management of Data*, 2026. doi:10.1145/3802002.

[2] J. Barbosa et al. "A Cost-Effective LLM-based Approach to Identify Wildlife Trafficking in Online Marketplaces." *Proceedings of the ACM on Management of Data*, 2025. doi:10.1145/3725256.

[3] Sentence Transformers. "all-MiniLM-L6-v2" model card. Hugging Face model documentation.

[4] P. Lesci and A. Vlachos. "AnchorAL: Computationally Efficient Active Learning for Large and Imbalanced Datasets." *NAACL*, 2024. https://aclanthology.org/2024.naacl-long.467/.

[5] C. Coleman et al. "Similarity Search for Efficient Active Learning and Search of Rare Concepts (SEALS)." *AAAI*, 2022. https://arxiv.org/abs/2007.00077.

> Also: [2]'s first author is **Juliana** Barbosa. The proposal says "J. S. Barbosa", but the paper lists no middle initial. Worth double-checking.

## BibTeX (for the final report)
```bibtex
@article{chung2026proxy,
  title   = {100x Cost \& Latency Reduction: Performance Analysis of AI Query Approximation using Lightweight Proxy Models},
  author  = {Chung, Yeounoh and Desai, Rushabh and He, Jian and Xiao, Yu and Hottelier, Thibaud and Kom Samo, Yves-Laurent and Khadilkar, Pushkar and Chen, Xianshun and Idicula, Sam and {\"O}zcan, Fatma and Halevy, Alon and Papakonstantinou, Yannis},
  journal = {Proceedings of the ACM on Management of Data},
  volume  = {4}, number = {3}, year = {2026},
  doi     = {10.1145/3802002}
}

@article{barbosa2025lts,
  title   = {A Cost-Effective {LLM}-based Approach to Identify Wildlife Trafficking in Online Marketplaces},
  author  = {Barbosa, Juliana and Gondhali, Ulhas and Petrossian, Gohar and Sharma, Kinshuk and Chakraborty, Sunandan and Jacquet, Jennifer and Freire, Juliana},
  journal = {Proceedings of the ACM on Management of Data},
  volume  = {3}, number = {3}, articleno = {119}, year = {2025},
  doi     = {10.1145/3725256}
}

@misc{minilm,
  title        = {all-MiniLM-L6-v2},
  author       = {{Sentence Transformers}},
  howpublished = {\url{https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2}},
  note         = {Hugging Face model card}
}

@inproceedings{lesci2024anchoral,
  title     = {{AnchorAL}: Computationally Efficient Active Learning for Large and Imbalanced Datasets},
  author    = {Lesci, Pietro and Vlachos, Andreas},
  booktitle = {Proceedings of NAACL-HLT (Volume 1: Long Papers)},
  pages     = {8445--8464}, year = {2024},
  url       = {https://aclanthology.org/2024.naacl-long.467/}
}

@inproceedings{coleman2022seals,
  title     = {Similarity Search for Efficient Active Learning and Search of Rare Concepts},
  author    = {Coleman, Cody and Chou, Edward and Katz-Samuels, Julian and Culatana, Sean and Bailis, Peter and Berg, Alexander C. and Nowak, Robert and Sumbaly, Roshan and Zaharia, Matei and Yalniz, I. Zeki},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {36}, pages = {6402--6410}, year = {2022}
}
```
