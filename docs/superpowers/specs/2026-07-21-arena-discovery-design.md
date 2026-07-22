# Arena-assisted model discovery

Date: 2026-07-21
Status: approved, ready for implementation planning

## Problem

`models.yaml` stops at 2025-04-28 — roughly 15 months stale. The cause is not a
missing data source. The automation that was supposed to keep it current has
never successfully run.

Four independent blockers, found by inspection on 2026-07-21:

1. **`models.yaml` was never pushed.** It lived only in unpushed local commit
   `dbd85fb`. Every `validate` run on the remote died with
   `FileNotFoundError: models.yaml`. Both CI runs to date were red.
2. **`discover-models` had never executed.** Cron is `0 13 * * 1` (Mondays
   13:00 UTC); the workflow landed on the remote at Mon 2026-07-20 20:14 UTC,
   seven hours past that week's slot. `candidates.yaml` was empty because the
   job had not run once — not because of a filtering bug.
3. **GitHub Actions cannot open PRs on this repo.** The first manual run
   (`workflow_dispatch`, run 29878915163) completed discovery and then failed:
   `GitHub Actions is not permitted to create or approve pull requests`.
4. **The discovery query is mis-shaped.** That same run scanned 300 models and
   produced zero candidates:

   ```
   Skipped: {'derivative': 110, 'not_org': 189, 'blocked': 1,
             'known': 0, 'small': 0, 'license': 0, 'no_params': 0}
   ```

   110 + 189 + 1 = 300. Every model was filtered before reaching the
   params/license checks.

Blockers 1 and 2 are resolved (see Completed below). Blocker 3 is a repo
setting only the owner can change. Blocker 4 is what this design addresses.

## Why the query fails

`discover.py` sorts *all of Hugging Face* by `created_at` descending and takes
the newest N. That population is dominated by finetunes, quantizations, and
merges. Major labs ship a flagship every few months, so a frontier release will
essentially never appear in the newest-300 window. Raising `--limit` buys more
derivatives, not more flagships.

An org-scoped query finds the same models immediately. One
`list_models(author=org)` call per org returned:

| HF repo | Size | Arena rank |
|---|---|---|
| `zai-org/GLM-5.2` | 753B | #10 |
| `moonshotai/Kimi-K2.7-Code` | 1058B | #22 |
| `MiniMaxAI/MiniMax-M3` | 427B | #27 |
| `moonshotai/Kimi-K2.6` | 1058B | #26 |
| `MiniMaxAI/MiniMax-M2.7` | 228B | #35 |
| `zai-org/GLM-5.1` | 753B | #17 |
| `google/gemma-4-12B-it` | 13B | Gemma 4 family, #37 |

Recency alone is still not sufficient. `deepseek-ai`'s five newest uploads are
all research artifacts (`eagle3_*`, `dflash_*`); DeepSeek V4 Pro and V4 Flash do
not appear despite ranking #24 and #29 on arena. Something must say *which*
models matter. That is arena's job.

## Why arena is a discovery source, not a benchmark

Arena and `models.yaml` have **zero** model overlap. `models.yaml` tops out at
Qwen3 / Llama 4 (April 2025); arena ranks Kimi K3, GLM 5.2, DeepSeek V4,
Qwen3.7, Gemma 4, Nemotron 3 Ultra, Minimax M3. Different generations entirely.
Attaching arena scores to existing rows would attach nothing to nothing.

The existing `pull_arena.py` also asserts open-weight status it cannot support.
It infers from org/product line via `KEYWORD_MAP` and deliberately overrides
arena's own per-model license label, so any disagreement resolves toward
"open". That yields false positives:

- `Meta Muse Spark 1.1 Meta · Proprietary` → `open_weight: true` (matched `meta`)
- `Kimi K3 Moonshot · Proprietary` → `open_weight: true` (matched `kimi`)

For a tracker whose value proposition is license clarity, that is the sharpest
edge in the repo. Meanwhile three rows carrying open licenses in their own
labels — Tencent Hy3 (Apache 2.0), Mimo V2.5 Pro (MIT), Thinking Machines
Inkling (Apache 2.0) — are discarded as `org: null`.

Resolving each arena name to a Hugging Face repo fixes both problems at once.
Public weights on HF are ground truth for open-weight status, strictly better
than an org keyword guess.

## Design

### Architecture

Three flat scripts, matching the repo's existing style. No new package layout.

```
scripts/hf_meta.py     (new)     shared: repo -> candidate dict, filter predicates
scripts/discover.py    (rewrite) org sweep + merge + write candidates.yaml
scripts/pull_arena.py  (refocus) scrape -> resolve -> arena_agent_rankings.yaml
```

`pull_arena.py` runs standalone and writes its YAML. `discover.py` reads that
file if present and works correctly without it. Arena being unreachable never
blocks HF discovery. Each script is independently runnable and testable.

### Org sweep (`discover.py`)

Replace the global firehose with a loop over `ORG_ALLOWLIST`, issuing one
`list_models(author=org, pipeline_tag="text-generation", sort="created_at")`
per org.

Consequences:

- `is_organization()`, `get_org_overview()`, `_ORG_CACHE`, the
  `--orgs-only` flag, and `AUTHOR_BLOCKLIST` all become dead code. They existed
  to police an unbounded query. Delete them. The allowlist is the query now.
- Filtering still matters. Extend `EXCLUDE_PATTERNS` with
  `mxfp8|nvfp4|w4a16|w8a8|-qat-|eagle3|dflash|-embed|reranker` to drop
  quantizations, speculative-decoding drafts, and embedding models that the
  current pattern misses.
- Zero-param research artifacts (`tabfm`, `NV-JEPA`) already fall out via the
  existing `no_params` and `min-params` checks. No new handling needed.

`hf_meta.py` owns `candidate_from_repo(info) -> dict` and the exclude
predicates, shared by both sources so a candidate row has one definition.

### Arena resolution (`pull_arena.py`)

`KEYWORD_MAP` stops being a truth claim about licensing and becomes a search
hint. It maps a keyword to an org name only; the `open_weight` boolean is
removed from it entirely.

Resolution, per arena row:

1. Normalize the display name. `GLM 5.2 (Max) Z.ai · MIT · SiliconFlow`
   becomes `GLM 5.2`: strip the trailing `Org · License · Provider` segment and
   parenthetical effort tags such as `(Max)`, `(High)`, `(xHigh)`.
2. Query `list_models(search=<normalized>, author=<hinted org>)`, falling back
   to an unhinted search when the keyword map has no entry.
3. Score the match against the repo id's normalized final segment:
   - exact match -> `resolution_confidence: high`
   - candidate is a prefix of the repo name, or vice versa -> `medium`
   - otherwise -> `low`, and set `needs_hf_repo: true`
4. No resolution means not open-weight. The row is written with
   `open_weight: false` and no candidate is emitted for it.

`open_weight` is therefore derived from whether public weights were found, not
from `KEYWORD_MAP`. Arena's own license label is retained on the row as
`arena_license_label` for human comparison.

`pull_arena.py` writes only leaderboard rows — rank, model, org, score,
`resolved_repo`, `resolution_confidence`. It does **not** build candidate
entries. Constructing candidates from resolved repos is `discover.py`'s job via
`hf_meta.candidate_from_repo`, so candidate rows have exactly one construction
path regardless of which source found the model.

Rows whose resolved org is absent from `ORG_ALLOWLIST` are collected into a
`new_orgs` list, surfaced in the PR body as candidate allowlist additions for
human approval. As of 2026-07-21 that is Tencent, Xiaomi, and Thinking
Machines.

### Merge

`discover.py` reads `arena_agent_rankings.yaml` if present, takes each row with
a `resolved_repo`, fetches its HF metadata, and builds a candidate through the
same `hf_meta.candidate_from_repo` path the org sweep uses. It then merges both
sets, deduplicating on lowercased `hf_repo`. Each candidate carries:

- `discovered_via`: list containing `org-sweep`, `arena`, or both
- `arena_rank`: integer, present only when arena resolved it
- `resolution_confidence` and `needs_hf_repo`: present only for arena rows

Sort by `arena_rank` ascending (rows without a rank last), then `release_date`
descending. The review queue leads with models people actually use.

**`models.yaml`'s schema is unchanged.** Arena stays discovery-only. Whether to
add an arena benchmark column is deferred until the tracker has caught up and
there are real rows to populate — designing that schema against zero
overlapping data would be guesswork. `SCHEMA.md` gains documentation for the
new `candidates.yaml`-only fields above.

### Error handling

- Arena unreachable, or its markup changed such that zero rows parse: log a
  warning, skip the arena source, continue with the org sweep. Never fatal.
  `discover.py` must produce candidates when arena.ai is down.
- Per-org HF failure (rate limit, 5xx): log that org, continue the remaining
  orgs. A single bad org does not abort the sweep.
- HF search returning nothing for an arena row is an expected outcome
  (the model is closed-weight), not an error.
- `discover.py` continues to exit 0 always; the Action decides whether the
  `candidates.yaml` diff is non-empty.

### Testing

The repo has no tests today. Add a minimal pytest setup covering pure
functions only, with no network:

- Arena HTML fixture (saved page) driving `parse_leaderboard` via the existing
  `--html` flag, asserting row count and field extraction.
- Display-name normalization: the `(Max)` / `· MIT · SiliconFlow` cases.
- Match scoring: exact, prefix, and no-match producing high/medium/low.
- Exclude predicates: `GLM-5.2-FP8`, `MiniMax-M3-MXFP8`, `gemma-4-12B-it-qat-w4a16-ct`,
  `eagle3_qwen3_8b_ttt7`, and `Nemotron-3-Embed-8B-BF16` are all excluded;
  `GLM-5.2`, `MiniMax-M3`, `Kimi-K2.6` are all kept.

Add `pytest` to `requirements.txt` and a test step to the `validate` workflow.

## Manual step required

`discover-models` cannot open its PR until the repo owner sets
**Settings -> Actions -> General -> Workflow permissions -> "Allow GitHub
Actions to create and approve pull requests."** Without this the pipeline
cannot deliver regardless of what is built.

## Completed during investigation

Pushed to `main` on 2026-07-21:

- `models.yaml` pushed; `validate` is green for the first time.
- `.gitignore` added covering `.DS_Store`, `.venv/`, `__pycache__/`, `*.pyc`.
- `.DS_Store` untracked via `git rm --cached`.

Not committed, pending this design: `pull_arena.py` and
`arena_agent_rankings.yaml` remain in the working tree.

## Out of scope

- Any change to `models.yaml`'s benchmark schema.
- Replacing or supplementing the MMLU anchor column.
- Backfilling the 15 months of missing models by hand. This design makes them
  *discoverable*; promoting candidates into `models.yaml` stays a human review
  step through the existing PR flow.
- A generated HTML page or any output surface beyond the README table.
