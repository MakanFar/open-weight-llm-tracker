# Data schema

`models.yaml` is the **single source of truth**. Everything else — the README table,
`models.json`, any HTML page — is generated from it. Edit the YAML, then run the render
scripts (`render_readme.py`, `render_json.py`) — never hand-edit a generated file; CI
re-renders both and fails on any diff.

`models.json` publishes the same index in the format consumers actually fetch, with
`arena_rank` and `aa_index` joined in as numbers or `null` (never the table's em dash).
Its envelope carries `source`, `license`, `generated` and `count`. Only the published
fields below are included — the candidates-only fields are never republished — and it
is licensed CC-BY-4.0 along with the rest of the data (see `LICENSE-DATA`).

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
| `weights_url` | no | url | Direct link to the weights |
| `notes` | no | string | Free text (standout result, quantization tips, etc.) |
| `commercial_use_verified` | no | bool | `false` (or absent) means the value was inferred from the licence tag, not read from the licence. The README marks these with `?`. |
| `params_active_source` | no | string | For auto-promoted MoE rows: the sentence in the model card the activation figure came from. |

## Conventions

- **Benchmark numbers are not stored here.** The anchor number is the Artificial
  Analysis Intelligence Index, fetched by `scripts/pull_aa.py` into
  `aa_scores.yaml` and joined on `hf_repo` (falling back to repo identity) at
  render time. Nothing hand-copies a score into `models.yaml` — a figure with
  no provenance is worse than no figure.
- **Set `commercial_use` by reading the license**, not by trusting the word "open".
- **One row per model.** A family flagship, or list sizes separately — don't do both.

## `candidates.yaml` (staging only)

`candidates.yaml` is written by `scripts/discover.py` and holds *unreviewed*
models. The file carries one top-level key besides `models:` — `generated`, the
ISO date of the last discovery run, which is what the README's "last discovery
run" badge reads (rendered output must never call `date.today()`, or CI's
re-render diff fails the day after). Rows use the `models.yaml` fields above
plus the discovery-only fields below. **Strip all of these fields when promoting a row into `models.yaml`** —
`validate.py` checks `models.yaml` only, so they would otherwise leak through.

| Field | Type | Notes |
|-------|------|-------|
| `discovered_via` | list | `org-sweep`, `arena`, or both — which source found it |
| `arena_rank` | integer | Agent Arena rank, present only if arena resolved it. Sorts the review queue. |
| `aa_index` | integer | Artificial Analysis Intelligence Index, present only if AA rates this exact `hf_repo` string. Lets you see the score *before* promoting. Refreshed every run, but keyed on an exact repo match — absent does NOT always mean AA does not rate the model: `render_readme.py` also falls back to a repo-identity join, so a model AA scores under a differently-spelled repo will still show a number once promoted. Check `aa_scores.yaml` directly if unsure. |
| `downloads` | integer | HF download count at discovery time; a rough popularity signal |
| `needs_hf_repo` | bool | Arena-only. `true` means either the leaderboard name matched the repo inexactly, or no repo resolved at all — **confirm the `hf_repo` really is that model (or find one) before promoting**. `false` means an exact match. |
| `resolution_confidence` | enum | Arena-only. `high` (exact name match) or `medium` (inexact, hence `needs_hf_repo: true`). Tells the reviewer *why* a row was flagged. |
| `needs_review` | list | Why the row was not auto-promoted, e.g. `moe-active-params-unknown`. Strip on promotion. |
| `gated_no_access` | bool | Set by `discover.py` when the repo's `config.json` answered **401/403** — the context window exists but is behind an HF access grant. Surfaces as `gated-repo-no-access` instead of `no-context-window`. Fix by accepting the licence terms on the model page with the account whose token CI uses. **Re-derived every run**, so it clears itself once access is granted. Strip on promotion. |
| `family_collision_reviewed` | bool | Set by a **human** to answer a `family-already-tracked` flag with *coexist*: this release collides with a family already in `models.yaml`, and a reviewer has confirmed both rows should stand. Must be a real YAML bool (`true`) — a quoted `"true"` or `no` is ignored on purpose, so a typo can never publish a row nobody cleared. Clears the collision reason **only**; every other gap still blocks promotion. To answer *supersede* instead, edit `models.yaml` by hand — `discover.py` only appends. Strip on promotion. |

Candidates are ordered arena-ranked first (ascending), then unranked by release
date descending — so the models people actually use lead the review queue.

### Open-weight status

`arena_agent_rankings.yaml` sets `open_weight: true` if and only if a public
weights repo resolved on Hugging Face. It is **not** arena's license label and
**not** an org guess. `needs_hf_repo: true` marks an inexact name match that a
human should verify before promotion.

## `aa_scores.yaml` (generated, joined at render)

Written by `scripts/pull_aa.py`. Maps `hf_repo` → the Artificial Analysis
Intelligence Index, a 0-100 composite (Agents 34%, Coding 24%, Scientific
Reasoning 24%, General 18%) that is re-weighted between versions, so values
are **not** comparable across time — only the current snapshot is stored:

```yaml
scores:
  meta-llama/Llama-3.3-70B-Instruct:
    aa_model: Llama 3.3 70B
    intelligence_index: 9
    source: https://artificialanalysis.ai/leaderboards/models
    variant: default
```

This replaced an HF-leaderboard-based `benchmark` field that could never be
filled automatically: HF Open LLM Leaderboard v2 publishes no plain MMLU and is
archived, and the HF model card API returns no structured eval data for any
tracked model. `render_readme.py` reads this file and prints `—` for any row
whose `hf_repo` isn't present, by exact match or by repo identity — AA covers
recent models only, so gaps on older rows are expected. The file is committed
so the render stays offline; a missing/empty/malformed file just means every
row renders `—`.
