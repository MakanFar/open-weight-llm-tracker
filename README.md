# Open-Weight & Open-Source LLM Tracker

A curated, machine-readable index of open-weight LLMs — parameter count, context window, an anchor benchmark, and license clarity. Currently tracking **25 models**.

Data lives in [`models.yaml`](models.yaml) (the source of truth). This table is generated — do not edit it by hand. See [SCHEMA.md](SCHEMA.md) for fields and [CONTRIBUTING.md](CONTRIBUTING.md) to add a model.

> **Columns:** **AA Index** is the [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/leaderboards/models?weights=open) — a 0–100 composite of agentic, coding, scientific-reasoning and general evaluations. `—` means Artificial Analysis does not currently rate that model; it drops older models, so coverage skews to recent releases. The index is re-weighted between versions, so values are not comparable across time. **Arena** is the Agent Arena rank (`—` = not currently ranked). A trailing `?` on **Commercial** marks a value inferred from the licence tag and not yet checked against the licence text.

<!-- MODELS_TABLE_START -->
| Model | Developer | Released | Params | Context | Modality | Arena | AA Index | License | Commercial |
|---|---|---|---|---|---|---|---|---|---|
| [Hy3](https://huggingface.co/tencent/Hy3) | Tencent | 2026-07-02 | 298.8B (21B active) | 262K | text | 30 | 42 | `apache-2.0` | Yes |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | Moonshot AI | 2026-06-13 | 2779.9B (104B active) | 1M | text | 5 | 60 | `kimi-k3` | Conditional |
| [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | MiniMax | 2026-06-02 | 427B (23B active) | 1M | multimodal | 33 | 45 | `minimax-community` | Conditional |
| [MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) | Xiaomi | 2026-04-27 | 1023.2B (42B active) | 1M | text | 32 | 43 | `mit` | Yes |
| [MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) | XiaomiMiMo | 2026-04-27 | 310.8B (15B active) | 1M | text | — | 38 | `mit` | Yes? |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | DeepSeek | 2026-04-22 | 1598.8B (49B active) | 1M | text | 26 | 45 | `mit` | Yes |
| [granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b) | ibm-granite | 2026-04-06 | 8.8B | 131K | text | — | — | `apache-2.0` | Yes? |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | Google | 2026-03-11 | 32.7B | 262K | text | 46 | 30 | `apache-2.0` | Yes |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | Qwen | 2026-01-30 | 79.7B (3B active) | 262K | text | — | 21 | `apache-2.0` | Yes? |
| [Qwen3 235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) | Alibaba | 2025-04-28 | 235B (22B active) | 131K | text | — | — | `apache-2.0` | Yes |
| [Llama 4 Maverick](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) | Meta | 2025-04-05 | 400B (17B active) | 1M | multimodal | — | 14 | `llama-4-community` | Conditional |
| [Llama 4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | Meta | 2025-04-05 | 109B (17B active) | 10M | multimodal | — | 10 | `llama-4-community` | Conditional |
| [Gemma 3 27B](https://huggingface.co/google/gemma-3-27b-it) | Google | 2025-03-12 | 27B | 128K | vision-language | — | — | `gemma` | Conditional |
| [Mistral Small 3](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | Mistral AI | 2025-01-30 | 24B | 32K | text | — | — | `apache-2.0` | Yes |
| [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | DeepSeek | 2025-01-20 | 671B (37B active) | 128K | text | — | — | `mit` | Yes |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) | DeepSeek | 2024-12-26 | 671B (37B active) | 128K | text | — | — | `deepseek` | Yes |
| [Phi-4](https://huggingface.co/microsoft/phi-4) | Microsoft | 2024-12-12 | 14B | 16K | text | — | — | `mit` | Yes |
| [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Meta | 2024-12-06 | 70B | 128K | text | — | 9 | `llama-3.3-community` | Conditional |
| [Qwen2.5 72B Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | Alibaba | 2024-09-19 | 72.7B | 131K | text | — | — | `qwen` | Conditional |
| [Qwen2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Alibaba | 2024-09-19 | 7.6B | 131K | text | — | — | `apache-2.0` | Yes |
| [Command R+ (08-2024)](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) | Cohere | 2024-08-30 | 104B | 128K | text | — | — | `cc-by-nc-4.0` | No |
| [Llama 3.1 405B Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) | Meta | 2024-07-23 | 405B | 128K | text | — | — | `llama-3.1-community` | Conditional |
| [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Meta | 2024-07-23 | 8B | 128K | text | — | — | `llama-3.1-community` | Conditional |
| [Gemma 2 27B](https://huggingface.co/google/gemma-2-27b-it) | Google | 2024-06-27 | 27B | 8K | text | — | — | `gemma` | Conditional |
| [Mixtral 8x22B Instruct](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) | Mistral AI | 2024-04-17 | 141B (39B active) | 65K | text | — | — | `apache-2.0` | Yes |
<!-- MODELS_TABLE_END -->

## Regenerate

```bash
pip install -r requirements.txt
python scripts/validate.py        # check the data
python scripts/pull_hf.py         # (optional) auto-fill fields from Hugging Face
python scripts/render_readme.py   # rebuild this table
```

## Staying current (automatic discovery)

Discovery runs weekly and classifies what it finds. A model that clears the notability bar (an Artificial Analysis score, an Agent Arena rank, or ≥500k Hugging Face downloads) **and** has no missing vitals is **appended** to `models.yaml` automatically — never edited, reordered, or deleted, only added to. A notable model missing something (most often a missing context window, or a family already tracked under a different repo name) waits in [`candidates.yaml`](candidates.yaml) with a `needs_review` list explaining what is missing. An unremarkable model is dropped before either file sees it.

[`scripts/discover.py`](scripts/discover.py) sweeps an allowlist of organizations — one Hugging Face query per org — rather than scanning all of HF by recency. Sorting the whole Hub by upload date returns finetunes and quantizations, essentially never a frontier release. **Adding an org to `ORG_ALLOWLIST` in `scripts/discover.py` is how the tracker gains coverage.** It skips quantizations/adapters/merges, dedups against `models.yaml`, and writes new rows with fields pre-filled.

[`scripts/pull_arena.py`](scripts/pull_arena.py) scrapes the [arena.ai](https://arena.ai/leaderboard/agent) leaderboard and resolves each ranked model to a Hugging Face repo. Arena rank then orders the review queue, so the models people actually use are reviewed first.

**Open-weight status comes from whether weights actually resolve on Hugging Face** — not from a vendor's name and not from a leaderboard's license label. A model is open-weight if and only if a public weights repo was found for it. See [SCHEMA.md](SCHEMA.md) for the discovery-only fields, including `needs_hf_repo`, which flags an inexact name match for a human to confirm.

The `discover-models` GitHub Action runs this weekly and opens a **pull request** — nothing it does ever commits straight to `main`, and that PR is the human-in-the-loop approval step. A row in the table above marked with a trailing `?` on **Commercial** was auto-promoted with `commercial_use` inferred from the licence tag: check it against the licence text before merging. For a row left in `candidates.yaml`, fill the gap named in its `needs_review` list and the next run promotes it automatically.
