# Open-Weight & Open-Source LLM Tracker

[![validate](https://github.com/MakanFar/open-weight-llm-tracker/actions/workflows/validate.yml/badge.svg)](https://github.com/MakanFar/open-weight-llm-tracker/actions/workflows/validate.yml)
[![models](https://img.shields.io/badge/models-53-1f6feb)](models.yaml)
[![last discovery run](https://img.shields.io/badge/last%20discovery%20run-2026--08--29-1f6feb)](https://github.com/MakanFar/open-weight-llm-tracker/actions/workflows/discover.yml)
[![code: MIT](https://img.shields.io/badge/code-MIT-3fb950)](LICENSE)
[![data: CC BY 4.0](https://img.shields.io/badge/data-CC--BY--4.0-3fb950)](LICENSE-DATA)

**What it is** — a curated, machine-readable index of **53 open-weight LLMs**: parameter count, context window, modality, licence, and an anchor benchmark. One row per model, newest first.

**Why it's different** — a model is listed as open-weight only if a public weights repo actually resolves on Hugging Face. Never from the vendor's name, never from a leaderboard's licence label. New releases are found automatically every week, but each one arrives as a **pull request**: nothing reaches the table without a human merging it.

**How to consume it** — fetch the data, don't scrape the table:

```bash
curl -sL https://raw.githubusercontent.com/MakanFar/open-weight-llm-tracker/main/models.json    # generated, with arena + AA joined in
curl -sL https://raw.githubusercontent.com/MakanFar/open-weight-llm-tracker/main/models.yaml    # the source of truth
```

[`models.yaml`](models.yaml) is the source of truth; [`models.json`](models.json) and this table are generated from it — do not edit either by hand. See [SCHEMA.md](SCHEMA.md) for fields and [CONTRIBUTING.md](CONTRIBUTING.md) to add a model.

> **Columns:** **AA Index** is the [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/leaderboards/models?weights=open) — a 0–100 composite of agentic, coding, scientific-reasoning and general evaluations. `—` means Artificial Analysis does not currently rate that model; it drops older models, so coverage skews to recent releases. The index is re-weighted between versions, so values are not comparable across time. **Arena** is the rank on arena.ai's text leaderboard among open-weight models (`—` = not currently ranked). A trailing `?` on **Commercial** marks a value inferred from the licence tag and not yet checked against the licence text; a trailing `†` marks one whose vendor publishes no licence file at all, so the tag is the only claim there has ever been and there is nothing to check. **Params** is the figure the vendor publishes where there is one (`params_total_stated_b`), otherwise the measured tensor count (`params_total_b`) — the two differ by a few percent because a checkpoint carries tensors a headline figure leaves out, and [`models.json`](models.json) carries both.

<!-- MODELS_TABLE_START -->
| Model | Developer | Released | Params | Context | Modality | Arena | AA Index | License | Commercial |
|---|---|---|---|---|---|---|---|---|---|
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | zai-org | 2026-08-25 | 321.3B (18B active) | 1M | text | 4 | — | `mit` | Yes |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Qwen | 2026-08-05 | 27.8B | 262K | multimodal | 21 | 52 | `apache-2.0` | Yes |
| [Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) | thinkingmachines | 2026-07-27 | 276B (12B active) | 1M | multimodal | 49 | 41 | `apache-2.0` | Yes |
| [Inkling](https://huggingface.co/thinkingmachines/Inkling) | thinkingmachines | 2026-07-14 | 975B (41B active) | 1M | multimodal | 18 | 42 | `apache-2.0` | Yes |
| [Hy3](https://huggingface.co/tencent/Hy3) | Tencent | 2026-07-02 | 298.8B (21B active) | 262K | text | 11 | 42 | `apache-2.0` | Yes |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | Moonshot AI | 2026-06-13 | 2779.9B (104B active) | 1M | text | 1 | 60 | `kimi-k3` | Conditional |
| [diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | google | 2026-06-09 | 25.8B (3.8B active) | 262K | multimodal | — | — | `apache-2.0` | Yes |
| [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | MiniMax | 2026-06-02 | 427B (23B active) | 1M | multimodal | 15 | 45 | `minimax-community` | Conditional |
| [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | google | 2026-05-23 | 12B | 262K | multimodal | — | — | `apache-2.0` | Yes |
| [MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) | Xiaomi | 2026-04-27 | 1023.2B (42B active) | 1M | text | 5 | 43 | `mit` | Yes |
| [MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) | XiaomiMiMo | 2026-04-27 | 310.8B (15B active) | 1M | text | 23 | 38 | `mit` | Yes |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | DeepSeek | 2026-04-22 | 1598.8B (49B active) | 1M | text | 10 | 45 | `mit` | Yes |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | deepseek-ai | 2026-04-22 | 284B (13B active) | 1M | text | 22 | 52 | `mit` | Yes |
| [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Qwen | 2026-04-21 | 27.8B | 262K | multimodal | — | 38 | `apache-2.0` | Yes |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Qwen | 2026-04-15 | 36B (3B active) | 262K | multimodal | — | 32 | `apache-2.0` | Yes |
| [granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) | ibm-granite | 2026-04-06 | 8.8B | 131K | text | 110 | — | `apache-2.0` | Yes |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | Google | 2026-03-11 | 31.3B | 262K | text | 13 | 30 | `apache-2.0` | Yes |
| [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | google | 2026-03-11 | 25.8B (3.8B active) | 262K | multimodal | 20 | 26 | `apache-2.0` | Yes |
| [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | google | 2026-03-02 | 8B | 131K | multimodal | — | — | `apache-2.0` | Yes |
| [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | google | 2026-03-02 | 5.1B | 131K | multimodal | — | — | `apache-2.0` | Yes |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | Qwen | 2026-02-24 | 122B (10B active) | 262K | multimodal | 38 | 33 | `apache-2.0` | Yes |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | Qwen | 2026-02-24 | 35B (3B active) | 262K | multimodal | 56 | 24 | `apache-2.0` | Yes |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | Qwen | 2026-02-24 | 27.8B | 262K | multimodal | 48 | — | `apache-2.0` | Yes |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | Qwen | 2026-02-16 | 397B (17B active) | 262K | multimodal | 17 | 34 | `apache-2.0` | Yes |
| [GLM-5](https://huggingface.co/zai-org/GLM-5) | zai-org | 2026-02-11 | 744B (40B active) | 202K | text | 9 | — | `mit` | Yes |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | Qwen | 2026-01-30 | 79.7B (3B active) | 262K | text | — | 21 | `apache-2.0` | Yes |
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | zai-org | 2026-01-19 | 30B (3B active) | 202K | text | 71 | — | `mit` | Yes |
| [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) | XiaomiMiMo | 2025-12-16 | 309.8B (15B active) | 262K | text | 62 | — | `mit` | Yes |
| [Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) | allenai | 2025-12-14 | 8.7B | 36K | multimodal | 93 | — | `apache-2.0` | Yes |
| [Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct) | allenai | 2025-12-10 | 32.2B | 65K | text | 92 | — | `apache-2.0` | Yes |
| [Olmo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think) | allenai | 2025-12-10 | 32.2B | 65K | text | 121 | — | `apache-2.0` | Yes |
| [Olmo-3-32B-Think](https://huggingface.co/allenai/Olmo-3-32B-Think) | allenai | 2025-11-19 | 32.2B | 65K | text | 108 | — | `apache-2.0` | Yes |
| [Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) | Qwen | 2025-09-22 | 235B (22B active) | 262K | multimodal | 45 | — | `apache-2.0` | Yes |
| [Qwen3-VL-235B-A22B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking) | Qwen | 2025-09-22 | 235B (22B active) | 262K | multimodal | 57 | — | `apache-2.0` | Yes |
| [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | Qwen | 2025-09-09 | 80B (3B active) | 262K | text | 53 | 17 | `apache-2.0` | Yes |
| [Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) | Qwen | 2025-09-09 | 80B (3B active) | 262K | text | 70 | — | `apache-2.0` | Yes |
| [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | openai | 2025-08-04 | 117B (5.1B active) | 131K | text | 79 | 24 | `apache-2.0` | Yes |
| [Qwen3 235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | Alibaba | 2025-04-28 | 235B (22B active) | 131K | text | 67 | — | `apache-2.0` | Yes |
| [Llama 4 Maverick](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) | Meta | 2025-04-05 | 400B (17B active) | 1M | multimodal | 96 | 14 | `llama-4-community` | Conditional |
| [Llama 4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | Meta | 2025-04-05 | 109B (17B active) | 10M | multimodal | 98 | 10 | `llama-4-community` | Conditional |
| [Gemma 3 27B](https://huggingface.co/google/gemma-3-27b-it) | Google | 2025-03-12 | 27B | 128K | vision-language | 72 | — | `gemma` | Conditional |
| [Mistral Small 3](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | Mistral AI | 2025-01-30 | 24B | 32K | text | 127 | — | `apache-2.0` | Yes |
| [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | DeepSeek | 2025-01-20 | 671B (37B active) | 128K | text | 54 | — | `mit` | Yes |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | DeepSeek | 2024-12-26 | 671B (37B active) | 128K | text | 75 | — | `deepseek` | Yes |
| [Phi-4](https://huggingface.co/microsoft/phi-4) | Microsoft | 2024-12-12 | 14B | 16K | text | 134 | — | `mit` | Yes |
| [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Meta | 2024-12-06 | 70B | 128K | text | 101 | — | `llama-3.3-community` | Conditional |
| [Qwen2.5 72B Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | Alibaba | 2024-09-19 | 72.7B | 131K | text | 114 | — | `qwen` | Conditional |
| [Qwen2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Alibaba | 2024-09-19 | 7.6B | 131K | text | — | — | `apache-2.0` | Yes |
| [Command R+ (08-2024)](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) | Cohere | 2024-08-30 | 104B | 128K | text | 142 | — | `cc-by-nc-4.0` | No |
| [Llama 3.1 405B Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) | Meta | 2024-07-23 | 405B | 128K | text | 91 | — | `llama-3.1-community` | Conditional |
| [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Meta | 2024-07-23 | 8B | 128K | text | 148 | — | `llama-3.1-community` | Conditional |
| [Gemma 2 27B](https://huggingface.co/google/gemma-2-27b-it) | Google | 2024-06-27 | 27B | 8K | text | 117 | — | `gemma` | Conditional |
| [Mixtral 8x22B Instruct](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) | Mistral AI | 2024-04-17 | 141B (39B active) | 65K | text | 141 | — | `apache-2.0` | Yes |
<!-- MODELS_TABLE_END -->

## Regenerate

```bash
pip install -r requirements.txt
python scripts/validate.py        # check the data
python scripts/pull_hf.py         # (optional) auto-fill fields from Hugging Face
python scripts/render_readme.py   # rebuild this table
python scripts/render_json.py     # rebuild models.json
```

## Staying current (automatic discovery)

Discovery runs weekly and classifies what it finds. A model that clears the notability bar (an Artificial Analysis score, an arena rank, or ≥500k Hugging Face downloads) **and** has no missing vitals is **appended** to `models.yaml` automatically — never edited, reordered, or deleted, only added to. A notable model missing something (most often a missing context window, or a family already tracked under a different repo name) waits in [`candidates.yaml`](candidates.yaml) with a `needs_review` list explaining what is missing. An unremarkable model is dropped before either file sees it.

[`scripts/discover.py`](scripts/discover.py) sweeps an allowlist of organizations — one Hugging Face query per org — rather than scanning all of HF by recency. Sorting the whole Hub by upload date returns finetunes and quantizations, essentially never a frontier release. **Adding an org to `ORG_ALLOWLIST` in `scripts/discover.py` is how the tracker gains coverage.** It skips quantizations/adapters/merges, dedups against `models.yaml`, and writes new rows with fields pre-filled.

[`scripts/pull_arena.py`](scripts/pull_arena.py) scrapes the [arena.ai](https://arena.ai/leaderboard/text/overall?license=open-source) leaderboard and resolves each ranked model to a Hugging Face repo. Arena rank then orders the review queue, so the models people actually use are reviewed first.

**Open-weight status comes from whether weights actually resolve on Hugging Face** — not from a vendor's name and not from a leaderboard's license label. A model is open-weight if and only if a public weights repo was found for it. See [SCHEMA.md](SCHEMA.md) for the discovery-only fields, including `needs_hf_repo`, which flags an inexact name match for a human to confirm.

The `discover-models` GitHub Action runs this weekly and opens a **pull request** — nothing it does ever commits straight to `main`, and that PR is the human-in-the-loop approval step. A row in the table above marked with a trailing `?` on **Commercial** was auto-promoted with `commercial_use` inferred from the licence tag: check it against the licence text before merging. For a row left in `candidates.yaml`, fill the gap named in its `needs_review` list and the next run promotes it automatically.

## License

Split by what the thing is. The code — `scripts/`, `tests/`, the workflows — is [MIT](LICENSE). The data — `models.yaml`, `models.json`, `candidates.yaml`, `aa_scores.yaml`, `arena_agent_rankings.yaml` and the table above — is [CC BY 4.0](LICENSE-DATA): reuse it freely, credit this repository, and say if you changed it.

That covers this compilation — the selection, verification and arrangement of the facts. **It grants nothing over the model weights themselves**, which stay under their vendors' own terms; the `license` column names those, and is the whole point of the index.
