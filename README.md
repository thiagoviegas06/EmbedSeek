# EmbedSeek: Proxy Models for Needle-in-a-Haystack Semantic Filters

NYU Data Engineering project, Oct 11 – Dec 10, 2026.
Team: Thiago Viegas (tjv235), Deep Paghdal (drp9757), Prashanth Sreenivasan (ps5761).

## Docs
| File | Purpose |
|---|---|
| [docs/01_project_overview.md](docs/01_project_overview.md) | The problem, research questions, method, in plain language |
| [docs/02_step_by_step_plan.md](docs/02_step_by_step_plan.md) | Week-by-week actionable plan with checklists |
| [docs/03_datasets.md](docs/03_datasets.md) | The three datasets, how to get them, status |
| [docs/04_jigsaw_exploration.md](docs/04_jigsaw_exploration.md) | First dataset downloaded + findings |
| [docs/05_evaluation_and_risks.md](docs/05_evaluation_and_risks.md) | Metrics, cost model, pitfalls to avoid |
| [docs/06_baseline_results.md](docs/06_baseline_results.md) | First baselines (random / query / ceiling) and findings |
| [docs/papers/00_index.md](docs/papers/00_index.md) | Notes on the 5 cited papers + synthesis |

## Setup
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -I scripts/make_splits.py data/jigsaw/raw data/jigsaw/splits
.venv/bin/python -I scripts/embed.py data/jigsaw/splits data/jigsaw/emb sentence-transformers/all-MiniLM-L6-v2
.venv/bin/python -I scripts/run_baselines.py data/jigsaw all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L6-v2
.venv/bin/python -m pytest tests -q          # leakage / integrity / sampler tests
```

## Layout
```
Project_Data_Engineering (1).pdf   proposal
data/<dataset>/raw/                untouched downloads (gitignore if repo is created)
data/<dataset>/profile_output.txt  profiling results
scripts/                           code (kept separate from data)
papers/                            cited papers (PDF + extracted text)
docs/                              these notes
```
