# Data schema

`models.yaml` is the **single source of truth**. Everything else (the README table,
any HTML page) is generated from it. Edit the YAML, then run the render script — never
hand-edit the generated table.

Each list entry is one model. Fields:

| Field | Required | Type | Notes |
|-------|----------|------|-------|
| `name` | yes | string | Human-readable model name, e.g. `Llama 3.1 405B Instruct` |
| `hf_repo` | no | string | Hugging Face repo id, e.g. `meta-llama/Llama-3.1-405B-Instruct`. Used by the auto-puller. |
| `developer` | yes | string | Org that released it: Meta, Alibaba, DeepSeek, Mistral AI, Google, Microsoft, etc. |
| `release_date` | yes | date | `YYYY-MM-DD` (use the 1st if only month is known) |
| `params_total_b` | yes | number | Total parameters in **billions** |
| `params_active_b` | yes | number | Active params per token in billions. For dense models = total. For MoE, the smaller routed count. |
| `architecture` | yes | enum | `dense` or `moe` |
| `context_window` | yes | integer | Max context in tokens |
| `modality` | yes | enum | `text`, `vision-language`, `multimodal` |
| `license` | yes | string | Short license id (see allowlist in `scripts/validate.py`) |
| `commercial_use` | yes | enum | `true`, `false`, or `conditional` (e.g. Llama's 700M-MAU gate) |
| `license_notes` | no | string | Any restriction worth flagging |
| `benchmark.name` | yes | string | The **one canonical benchmark** you anchor on, e.g. `MMLU` |
| `benchmark.score` | yes | number | Score on that benchmark |
| `benchmark.source` | yes | url | Where the number came from — leaderboard or paper |
| `weights_url` | no | url | Direct link to the weights |
| `notes` | no | string | Free text (standout result, quantization tips, etc.) |

## Conventions

- **One canonical benchmark.** Pick MMLU (or MMLU-Pro) as the anchor column so rows are
  comparable. Put anything else in `notes`. A table where every row uses a different
  benchmark is not a comparison.
- **Prefer third-party benchmark numbers** (HF Open LLM Leaderboard, LMArena, Epoch AI)
  over lab-self-reported figures. Whatever you use, record it in `benchmark.source`.
- **Set `commercial_use` by reading the license**, not by trusting the word "open".
- **One row per model.** A family flagship, or list sizes separately — don't do both.
