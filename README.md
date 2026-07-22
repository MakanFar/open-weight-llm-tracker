# Open-Weight & Open-Source LLM Tracker

A curated, machine-readable index of open-weight LLMs — parameter count, context window, an anchor benchmark, and license clarity. Currently tracking **16 models**.

Data lives in [`models.yaml`](models.yaml) (the source of truth). This table is generated — do not edit it by hand. See [SCHEMA.md](SCHEMA.md) for fields and [CONTRIBUTING.md](CONTRIBUTING.md) to add a model.

> **Benchmark caveat:** the benchmark column mixes vendor-reported and third-party numbers (see each row's `benchmark.source` in the YAML). Anchor to a single leaderboard before relying on it for comparisons.

<!-- MODELS_TABLE_START -->
| Model | Developer | Released | Params | Context | Modality | Benchmark | License | Commercial |
|---|---|---|---|---|---|---|---|---|
| [Qwen3 235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | Alibaba | 2025-04-28 | 235B (22B active) | 131K | text | MMLU 87.0 | `apache-2.0` | Yes |
| [Llama 4 Maverick](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) | Meta | 2025-04-05 | 400B (17B active) | 1M | multimodal | MMLU 85.5 | `llama-4-community` | Conditional |
| [Llama 4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | Meta | 2025-04-05 | 109B (17B active) | 10M | multimodal | MMLU 79.6 | `llama-4-community` | Conditional |
| [Gemma 3 27B](https://huggingface.co/google/gemma-3-27b-it) | Google | 2025-03-12 | 27B | 128K | vision-language | MMLU 78.6 | `gemma` | Conditional |
| [Mistral Small 3](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | Mistral AI | 2025-01-30 | 24B | 32K | text | MMLU 81.0 | `apache-2.0` | Yes |
| [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | DeepSeek | 2025-01-20 | 671B (37B active) | 128K | text | MMLU 90.8 | `mit` | Yes |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | DeepSeek | 2024-12-26 | 671B (37B active) | 128K | text | MMLU 88.5 | `deepseek` | Yes |
| [Phi-4](https://huggingface.co/microsoft/phi-4) | Microsoft | 2024-12-12 | 14B | 16K | text | MMLU 84.8 | `mit` | Yes |
| [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Meta | 2024-12-06 | 70B | 128K | text | MMLU 86.0 | `llama-3.3-community` | Conditional |
| [Qwen2.5 72B Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | Alibaba | 2024-09-19 | 72.7B | 131K | text | MMLU 86.1 | `qwen` | Conditional |
| [Qwen2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Alibaba | 2024-09-19 | 7.6B | 131K | text | MMLU 74.2 | `apache-2.0` | Yes |
| [Command R+ (08-2024)](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) | Cohere | 2024-08-30 | 104B | 128K | text | MMLU 75.7 | `cc-by-nc-4.0` | No |
| [Llama 3.1 405B Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) | Meta | 2024-07-23 | 405B | 128K | text | MMLU 88.6 | `llama-3.1-community` | Conditional |
| [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Meta | 2024-07-23 | 8B | 128K | text | MMLU 68.4 | `llama-3.1-community` | Conditional |
| [Gemma 2 27B](https://huggingface.co/google/gemma-2-27b-it) | Google | 2024-06-27 | 27B | 8K | text | MMLU 75.2 | `gemma` | Conditional |
| [Mixtral 8x22B Instruct](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) | Mistral AI | 2024-04-17 | 141B (39B active) | 65K | text | MMLU 77.8 | `apache-2.0` | Yes |
<!-- MODELS_TABLE_END -->

## Regenerate

```bash
pip install -r requirements.txt
python scripts/validate.py        # check the data
python scripts/pull_hf.py         # (optional) auto-fill fields from Hugging Face
python scripts/render_readme.py   # rebuild this table
```

## Staying current (automatic discovery)

Two scripts feed the review queue in [`candidates.yaml`](candidates.yaml). Neither ever edits `models.yaml` directly.

[`scripts/discover.py`](scripts/discover.py) sweeps an allowlist of organizations — one Hugging Face query per org — rather than scanning all of HF by recency. Sorting the whole Hub by upload date returns finetunes and quantizations, essentially never a frontier release. **Adding an org to `ORG_ALLOWLIST` in `scripts/discover.py` is how the tracker gains coverage.** It skips quantizations/adapters/merges, dedups against `models.yaml`, and writes new rows with fields pre-filled.

[`scripts/pull_arena.py`](scripts/pull_arena.py) scrapes the [arena.ai](https://arena.ai/leaderboard/agent) leaderboard and resolves each ranked model to a Hugging Face repo. Arena rank then orders the review queue, so the models people actually use are reviewed first.

**Open-weight status comes from whether weights actually resolve on Hugging Face** — not from a vendor's name and not from a leaderboard's license label. A model is open-weight if and only if a public weights repo was found for it. See [SCHEMA.md](SCHEMA.md) for the discovery-only fields, including `needs_hf_repo`, which flags an inexact name match for a human to confirm.

The `discover-models` GitHub Action runs it weekly and opens a **pull request** with the new candidates — review the PR, fill the `TODO` fields (active params, benchmark, commercial-use), move approved rows into `models.yaml`, and merge.
