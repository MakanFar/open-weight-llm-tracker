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
5. Open a PR. CI runs `validate.py`; a red check means a schema problem — including two rows whose `hf_repo` name the same weights (see "One row per model" below).

## The rules that matter

- **Don't add a benchmark field.** `models.yaml` stores no benchmark score — the anchor
  number is the Artificial Analysis Intelligence Index, fetched by `scripts/pull_aa.py`
  into `aa_scores.yaml` and joined on `hf_repo` (falling back to repo identity) at render
  time. A hand-copied score with no provenance is worse than no score; put any standout
  result in `notes` instead.
- **Read the license.** Set `commercial_use` to `true` / `false` / `conditional` by
  actually checking the license, not by the word "open". If the license isn't in the
  allowlist in `scripts/validate.py`, add it there in the same PR.
- **Open weights only.** The model's weights must be publicly downloadable. API-only
  models (GPT-4, Claude, Gemini) don't belong here.
- **One row per model.** Either the family flagship or distinct sizes — don't list both.
  `validate.py` enforces this: two rows whose `hf_repo` resolves to the same repo
  identity (same model, different repo spelling) fail CI as a duplicate.
- **MoE params:** put total in `params_total_b` and routed/active in `params_active_b`.
  For dense models the two are equal.

## Reviewing auto-discovered candidates

`scripts/discover.py` (run weekly by the `discover-models` Action) no longer only
stages — it **classifies**. Each discovered model is routed one of three ways: an
unremarkable one is dropped before either file sees it; a notable one with no missing
vitals and no schema problems is **appended straight into `models.yaml`**; everything
else notable waits in `candidates.yaml` with a `needs_review` list. `discover.py` never
edits, reorders, or deletes an existing row — appending to `models.yaml` is the only
write it makes there. Entries come from an org sweep, from the arena leaderboard, or
both — `discovered_via` says which.

The human gate is the weekly PR itself, not a hand-move step. When a discovery PR
shows up:

1. **In `models.yaml`'s diff** — the rows `discover.py` auto-promoted. Check each is a
   real base model (not a fine-tune/merge the name filter missed), and check
   `commercial_use` on any row the rendered README marks with a trailing `?` on
   **Commercial** — that means `commercial_use_verified` is `false`, i.e. the value was
   *inferred* from the licence tag, not read from the licence text. Fix it by hand if
   the guess is wrong.
2. **In `candidates.yaml`** — rows still waiting on a human. Each carries a
   `needs_review` list saying exactly what is missing: a vitals gap (e.g.
   `no-context-window`, `moe-active-params-unknown`) or a `schema-invalid: ...` entry
   quoting the specific `validate.py` complaint (most often a hand-edited field, like a
   `release_date` that isn't a real date). If a row has **`needs_hf_repo: true`**, the
   leaderboard name matched that repo inexactly — confirm the repo really is that model
   before promoting it, the one field you cannot take on trust. Fix the gap in place;
   the next run promotes the row automatically once nothing is left to flag. To
   promote a row yourself instead of waiting, move it into `models.yaml` and strip the
   discovery-only fields listed in [SCHEMA.md](SCHEMA.md).
3. Run `python scripts/render_readme.py`, commit, merge.

## Where the data comes from

- **Params / context / license tag:** the Hugging Face repo (`hf_repo`) — `pull_hf.py`
  can auto-fill these.
- **Authoritative params & MoE split:** the model's paper / technical report.
- **Benchmarks:** not stored in `models.yaml`. The AA Index column comes from
  `scripts/pull_aa.py`, which scrapes the Artificial Analysis Intelligence
  Index into `aa_scores.yaml` and is joined on `hf_repo` (falling back to repo
  identity) at render time.
- **Commercial-use flag:** you, reading the actual license.
