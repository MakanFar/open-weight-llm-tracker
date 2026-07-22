# Contributing

Thanks for adding to the tracker. A few rules keep the data comparable and trustworthy.

## Adding or updating a model

1. Edit **`models.yaml`** only. Never edit the README table directly — it's generated.
2. Add one entry following the fields in [SCHEMA.md](SCHEMA.md).
3. Run the checks locally:
   ```bash
   pip install -r requirements.txt
   python scripts/validate.py
   python scripts/render_readme.py
   ```
4. Commit both `models.yaml` and the regenerated `README.md`.
5. Open a PR. CI runs `validate.py`; a red check means a schema problem.

## The rules that matter

- **One canonical benchmark.** We anchor on **MMLU** so rows compare. Put any other
  standout result in `notes`, not in the benchmark column.
- **Cite your benchmark source.** `benchmark.source` must be a URL (leaderboard or paper)
  or the literal string `vendor` if it's a self-reported number. Prefer third-party
  numbers (HF Open LLM Leaderboard, LMArena, Epoch AI) over vendor claims.
- **Read the license.** Set `commercial_use` to `true` / `false` / `conditional` by
  actually checking the license, not by the word "open". If the license isn't in the
  allowlist in `scripts/validate.py`, add it there in the same PR.
- **Open weights only.** The model's weights must be publicly downloadable. API-only
  models (GPT-4, Claude, Gemini) don't belong here.
- **One row per model.** Either the family flagship or distinct sizes — don't list both.
- **MoE params:** put total in `params_total_b` and routed/active in `params_active_b`.
  For dense models the two are equal.

## Reviewing auto-discovered candidates

`scripts/discover.py` (run weekly by the `discover-models` Action) stages new models
in `candidates.yaml`, never in `models.yaml`. Entries come from an org sweep, from the
arena leaderboard, or both — `discovered_via` says which. When a discovery PR shows up:

1. Open `candidates.yaml` and check each entry is a real base model worth tracking
   (not a fine-tune/merge the name filter missed).
2. If a row has **`needs_hf_repo: true`**, the leaderboard name matched that repo
   inexactly. Confirm the repo really is that model before promoting it — this is the
   one field you cannot take on trust.
3. Fix the `TODO` fields: `params_active_b` + `architecture` for MoE models, the
   `context_window` if it came through as `0`, the `benchmark` score + source, and
   confirm `commercial_use` by reading the actual license.
4. Move approved entries into `models.yaml`, delete them from `candidates.yaml`.
   Strip the discovery-only fields listed in [SCHEMA.md](SCHEMA.md) as you go.
5. Run `python scripts/render_readme.py`, commit, merge.

## Where the data comes from

- **Params / context / license tag:** the Hugging Face repo (`hf_repo`) — `pull_hf.py`
  can auto-fill these.
- **Authoritative params & MoE split:** the model's paper / technical report.
- **Benchmarks:** a single third-party leaderboard, run under one harness.
- **Commercial-use flag:** you, reading the actual license.
