# Auto-populated MMLU and Agent Arena columns + context_window fix

**Date:** 2026-07-28
**Status:** Design approved, ready for implementation plan

## Problem

Three gaps in how the tracker's data is populated:

1. **MMLU is entirely hand-entered.** Every `benchmark.score` in `models.yaml`
   was typed in from a vendor report or a blog round-up (13 of 16 rows are
   `source: vendor`, 3 are blog aggregators). None is auto-refreshed, and the
   column mixes harnesses, so it is not a clean comparison.
2. **Agent Arena rank is not surfaced in the published table.** The rank already
   exists per resolved model in `arena_agent_rankings.yaml` and flows into
   `candidates.yaml` as `arena_rank`, but never reaches `models.yaml` or the
   README, so a reader cannot see real-world usage ranking.
3. **`discover.py` writes `context_window: 0`.** It reads the context length
   from the HF API `expand`, which is usually empty, so every discovered
   candidate lands with `0` and a manual TODO — even though `pull_hf.py`
   already knows how to get it from `config.json`.

## Goals

- A README **MMLU column** auto-populated from the HF Open LLM Leaderboard,
  falling back to the existing manual score when the leaderboard has no entry.
- A README **Arena column** auto-populated from the arena rank already present
  in `arena_agent_rankings.yaml`.
- **`discover.py`** fills a real `context_window` for new candidates by fetching
  `config.json`, instead of emitting `0`.

## Non-goals

- Writing any auto-fetched value into `models.yaml`. It stays 100% human-curated
  — the auto values live in committed sidecar files.
- Backfilling context windows for the 16 already-published models (they already
  have correct values). The fix is discovery-side only.
- Enriching `candidates.yaml` with MMLU/arena scores. This feature targets the
  published README columns plus the context_window discovery fix.
- Replacing the manual `benchmark.score`. It remains required in `validate.py`
  and is the guaranteed fallback.

## Key constraints (why the architecture is shaped this way)

- **Render must stay deterministic and offline.** `validate.yml` re-renders the
  README and fails on any diff (`git diff --exit-code README.md`). Therefore
  `render_readme.py` must never fetch at render time; it reads only committed
  files. All network access lives in separate fetch scripts whose output is
  committed, exactly like `discover.py` / `pull_arena.py` / `pull_hf.py`.
- **`models.yaml` is the human-only source of truth.** Nothing automated writes
  to it. Auto values are joined in at render from sidecars, never merged into
  `models.yaml`.
- **Graceful degradation.** A missing, empty, or malformed sidecar must never
  break the render or CI — render falls back to manual scores / blank arena
  cells, mirroring the existing `load_arena()` shape-validation pattern.

## Architecture

Three independent, separately testable pieces.

### A. New fetcher: `scripts/pull_leaderboard.py`

Queries the HF Open LLM Leaderboard for each `hf_repo` in `models.yaml` and
writes a committed sidecar `leaderboard_scores.yaml`:

```yaml
# AUTO-FETCHED by scripts/pull_leaderboard.py — do not hand-edit
scores:
  meta-llama/Llama-3.1-405B-Instruct:
    mmlu: 88.6
    source: "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
```

- Keyed by `hf_repo` (exact repo id; join is case-insensitive at render).
- Network only here, never at render.
- A repo absent from the leaderboard is simply omitted from the file.
- Resilient like the other pullers: per-repo failures are logged and skipped;
  the script writes whatever it resolved and exits 0.

### B. Arena rank — no new fetcher

The rank already lives in `arena_agent_rankings.yaml` under each row
(`resolved_repo` → `rank`). Render reads it directly. No new fetch or storage.

### C. `render_readme.py` join

Reads `models.yaml` (source of truth) plus the two sidecars
(`leaderboard_scores.yaml`, `arena_agent_rankings.yaml`), joins by lowercased
`hf_repo`, and emits two columns:

- **MMLU**: leaderboard score if present, else the row's `benchmark.score`.
  The cell marks provenance so the mixed-source caveat stays honest — leaderboard
  values render plain (`88.6`), manual-fallback values render marked (`88.6*`),
  with a one-line legend under the table.
- **Arena**: the arena rank when the repo resolved on the leaderboard, else `—`.

Table sort order is unchanged (newest-first, then size). Arena is an added
column, not a re-sort.

### D. Shared context fetch in `hf_meta.py`

Add a best-effort `config.json` fetch (reads `max_position_embeddings`,
`max_sequence_length`, `n_positions`, including nested `text_config` /
`llm_config`) into `hf_meta.py`, so both discovery paths (org sweep and
arena-resolved) and `pull_hf.py` share one code path. `discover.py` calls it
after `should_track()` passes; on any failure the value stays `0` and the
existing review TODO applies. The fetch is wrapped so one repo's failure never
aborts the sweep.

## Data flow

```
pull_leaderboard.py --(HF Open LLM Leaderboard)--> leaderboard_scores.yaml
pull_arena.py       --(arena.ai + HF)-----------> arena_agent_rankings.yaml
                                                          |
models.yaml (human) --------------------------------------+--> render_readme.py --> README.md
                                                (join by lowercased hf_repo, offline)

discover.py --(HfApi + config.json via hf_meta)--> candidates.yaml (real context_window)
```

## Error handling

- **Leaderboard fetch fails / repo absent / network blocked** → log and skip the
  repo; write only what resolved. Missing/empty/malformed `leaderboard_scores.yaml`
  → render falls back to manual scores everywhere.
- **Malformed sidecar shape** → render validates after parse (top-level mapping,
  `scores` a dict, each entry a dict with numeric `mmlu`); bad entries skipped,
  never fatal. Same defense as `load_arena()`.
- **Arena file missing/empty** → Arena column all `—`. Never fatal.
- **context_window unresolvable** → stays `0`, existing "fill during review"
  TODO applies; best-effort fetch wrapped so a failure never aborts the sweep.

## Testing (pytest, fully offline — no live network in any test)

- `pull_leaderboard.py`: inject a fake fetch fn (as `fake_search` does in the
  arena tests). Assert correct repo→score mapping, missing repos skipped,
  malformed leaderboard rows tolerated, exits 0 on total failure.
- `render_readme.py`: unit-test the join — leaderboard hit uses leaderboard
  score; miss falls back to manual and marks it; arena hit shows rank; miss
  shows `—`; malformed/missing sidecars degrade to manual/blank.
- `hf_meta.py` context fetch: fake `config.json` fetch → each of the three keys
  read, nested `text_config`/`llm_config` handled, failure returns `0` without
  raising.
- Update existing `render_readme` / discovery / `hf_meta` tests for the new
  column and context path.

## Files touched

- **New:** `scripts/pull_leaderboard.py`, `leaderboard_scores.yaml` (generated),
  `tests/test_pull_leaderboard.py`, `tests/test_render_join.py` (or extend the
  existing render test).
- **Changed:** `scripts/render_readme.py` (join + two columns + legend),
  `scripts/hf_meta.py` (shared context fetch), `scripts/discover.py` (call the
  fetch), `scripts/pull_hf.py` (reuse the shared fetch — optional consolidation),
  `SCHEMA.md` (document `leaderboard_scores.yaml` and the render join),
  `.gitignore` (nothing — the sidecar is committed).
- **Docs:** README prose in `render_readme.py`'s `body` string describes the two
  new auto columns and the legend.

## Open follow-up (not in this spec)

- Wiring `pull_leaderboard.py` into the weekly `discover.yml` workflow so the
  scores refresh automatically and land in the same review PR. Default for now
  is a manual `python scripts/pull_leaderboard.py` run. Decide during/after
  implementation.
