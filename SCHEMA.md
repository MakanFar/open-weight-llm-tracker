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

## `candidates.yaml` (staging only)

`candidates.yaml` is written by `scripts/discover.py` and holds *unreviewed*
models. Rows use the `models.yaml` fields above plus the discovery-only fields
below. **Strip all of these fields when promoting a row into `models.yaml`** —
`validate.py` checks `models.yaml` only, so they would otherwise leak through.

| Field | Type | Notes |
|-------|------|-------|
| `discovered_via` | list | `org-sweep`, `arena`, or both — which source found it |
| `arena_rank` | integer | Agent Arena rank, present only if arena resolved it. Sorts the review queue. |
| `downloads` | integer | HF download count at discovery time; a rough popularity signal |
| `needs_hf_repo` | bool | Arena-only. `true` means either the leaderboard name matched the repo inexactly, or no repo resolved at all — **confirm the `hf_repo` really is that model (or find one) before promoting**. `false` means an exact match. |
| `resolution_confidence` | enum | Arena-only. `high` (exact name match) or `medium` (inexact, hence `needs_hf_repo: true`). Tells the reviewer *why* a row was flagged. |

Candidates are ordered arena-ranked first (ascending), then unranked by release
date descending — so the models people actually use lead the review queue.

### Open-weight status

`arena_agent_rankings.yaml` sets `open_weight: true` if and only if a public
weights repo resolved on Hugging Face. It is **not** arena's license label and
**not** an org guess. `needs_hf_repo: true` marks an inexact name match that a
human should verify before promotion.

## `leaderboard_scores.yaml` (generated, joined at render)

Written by `scripts/pull_leaderboard.py`. Maps `hf_repo` → an MMLU score from the
HF Open LLM Leaderboard:

```yaml
scores:
  meta-llama/Llama-3.1-405B-Instruct:
    mmlu: 88.6
    source: "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
```

`render_readme.py` reads this file and uses the leaderboard score for the MMLU
column when a repo is present, otherwise it falls back to that row's manual
`benchmark.score` in `models.yaml`. The file is committed so the render stays
offline; a missing/empty/malformed file just means every row uses its manual score.
