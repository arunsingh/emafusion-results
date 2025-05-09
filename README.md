# EMAFusion-results 🚀

**Open-source toolkit (MIT) for reproducing and extending “EMAFusion:  
A Self-Optimizing System for Seamless LLM Selection and Integration” (arXiv 2504.10681).**

Main goals
- ⚡ Re-run the original taxonomy/learned routing + cascade experiments
- 💸 Log *cost vs. quality* curves for new model mixes (Llama 3, Gemma 27B, etc.)
- 📊 Publish result CSVs so the community can compare apples-to-apples

## Quick start

```bash
# 1. clone
git clone https://github.com/arunsingh/emafusion-results.git
cd emafusion-results

# 2. install (editable) + dev requirements
pip install -e ".[dev]"

# 3. set your API keys (or point to Ollama)
export OPENAI_API_KEY=...
export OLLAMA_HOST=http://localhost:11434

# 4. run the default cascade on sample prompts
emafusion run scripts/cascade_config.yaml examples/quick_prompt.json

```


## EmaFusion High Level architecture

```bash
           ┌─────────────┐
 Prompt →──► Taxonomy    │
           │  Router     │───► cheap-model-1 ┐
           └────┬────────┘                   │
                │  “Seen this before”        ▼
                │                    ┌────────────────┐
                ▼                    │ Multi-Judge    │
           ┌─────────────┐           │ Confidence     │
           │ Learned      │          │ Assessment     │
           │  Router      │──────────┤  (LLMs + RM)   │
           └────┬────────┘           └────────────────┘
                │ “Ambiguous / novel”         ▲
                └─────────────────────────────┘
                                 if confidence < τ
                                 escalate up the cascade
                                 (model-2 → … → GPT-4-class)

           When two or more drafts survive,
           ───────────────────────────────────
                         │
                         ▼
                 ┌─────────────┐
                 │  Fusion     │  (rank, vote, or blend)
                 └─────────────┘
                         │
                         ▼
                    Final answer

```

## Repo structure

| Path                 | What’s inside                                            |
| -------------------- | -------------------------------------------------------- |
| `src/emafusion/`     | Core library: router, cascade, fusion, judges, Typer CLI |
| `scripts/`           | Opinionated orchestration entry points                   |
| `examples/`          | Minimal configs & prompt sets to smoke-test your install |
| `notebooks/`         | Exploratory analysis of result logs                      |
| `tests/`             | Pytest unit tests (coverage ≥ 90 %)                      |
| `.github/workflows/` | CI (lint → type-check → test → docker-build)             |


## Contributing in the Project
- Fork → Feature branch → PR.
- Make sure pytest -q passes and pre-commit run --all-files is green.
- All code changes require a matching docstring + type hints.
