# Replace HF-sourced MMLU with the Artificial Analysis Intelligence Index

**Date:** 2026-08-03
**Status:** approved design, not yet implemented

## Problem

The `benchmark` column is the least trustworthy thing in the tracker.

`scripts/pull_leaderboard.py` was built to fill it automatically from the HF Open
LLM Leaderboard. It never worked: leaderboard v2 publishes `MMLU-PRO` and
`MMLU-PRO Raw`, never a plain `MMLU` column, so the exact-match on `"mmlu"`
scored nothing across all 4576 rows and wrote an empty sidecar. It failed
silently because the renderer falls back to `models.yaml`.

Fixing the key exposed the deeper problem: leaderboard v2 is archived. Only
**8 of 16** tracked models appear in it at all, and none of the 2026 frontier
models the discovery pipeline exists to surface. Meanwhile the HF model card API
returns **no** structured eval data for any model we track — 0 of 42 repos
checked carry a `model-index`, `metrics`, or `eval_results` field. The API can
tell us parameters, context window and license; it cannot tell us how good a
model is.

That leaves every `benchmark.score` a hand-copied, vendor-self-reported MMLU
figure of unknown vintage, in a column that reads as if the numbers were
comparable.

## Decision

Drop MMLU entirely. Source the anchor number from the **Artificial Analysis
Intelligence Index**, joined to `models.yaml` the way arena ranks already are.

Arena rank and AA Index become the tracker's only two comparative signals, both
third-party, both current, both re-fetched on a schedule.

### Decisions taken, with rationale

| Decision | Choice | Why |
|---|---|---|
| What to store | AA Intelligence Index score (0–100) | A real comparable value. AA publishes no rank column — position implies order, so a derived rank would churn whenever any model is added. |
| Fate of `benchmark:` | Removed entirely | Keeping it as a fallback would mix scales: AA Index ~10–57 next to MMLU ~86 in one column invites a false comparison, and it would affect 13 of 16 rows. |
| Join key | Normalised name match, no HF calls | AA publishes display names, not repo ids, so matching is unavoidable. Doing it locally avoids the HF search dependency and its rate-limit failure modes. |
| Multiple AA rows per model | Highest index wins, variant recorded | AA lists reasoning-effort variants with a wide spread (Kimi K3: 57 max vs 47 low). "Base row only" would drop Kimi K3 entirely — no bare row exists. |
| Vals Index | Deferred to a separate spec | Not sweepable the way AA is; see Out of scope. |

## Evidence

Gathered against the live sites on 2026-08-03:

- AA leaderboard is fully server-rendered: one `requests` call returns a single
  `<table>` with 129 scored rows. The `open_weights=open_source` query parameter
  filters client-side only, so the server returns everything.
- AA covers **3 of 16** currently tracked models: Llama 4 Scout (10), Llama 4
  Maverick (14), Llama 3.3 70B (9). AA drops older models — no Mixtral, no
  DeepSeek V3/R1, no Qwen2.5, no Gemma 2/3, no Phi-4, no Command R+.
- AA covers the **candidate** set well: Kimi K3 (57), GLM-5.2 (51), DeepSeek V4
  Pro (44), Gemma 4 31B (29), plus 18 Qwen3.5+ rows.
- AA names models `Llama 3.3 70B` where `models.yaml` says
  `Llama 3.3 70B Instruct`; suffix-tolerant matching is required, not optional.

Coverage is therefore poor today and improves precisely as the discovery
pipeline promotes 2026 models. This is accepted knowingly.

## Removed

- `scripts/pull_leaderboard.py`
- `leaderboard_scores.yaml`
- `tests/test_pull_leaderboard.py`
- `load_leaderboard`, `mmlu_cell`, `mmlu_pro_cell` in `scripts/render_readme.py`
  and their tests
- The `benchmark:` block from all 16 `models.yaml` rows
- `"benchmark"` from `REQUIRED` and the benchmark validation block in
  `scripts/validate.py`
- The `benchmark.*` rows and the "one canonical benchmark" convention in
  `SCHEMA.md`; the equivalent line in `CLAUDE.md`
- MMLU wording in the README prose (the `body` string in `render_readme.py:main()`)

## Added

### `scripts/names.py`

`slug()` and the variant-suffix stripping currently private to `pull_arena.py`
move here, so `pull_aa.py` shares one implementation rather than duplicating it
or importing a scraper from a scraper. `pull_arena.py` is updated to import from
it; its behaviour is unchanged.

### `scripts/pull_aa.py`

Mirrors the shape of `pull_arena.py`.

1. Fetch `https://artificialanalysis.ai/leaderboards/models`.
2. Parse the single `<table>`; skip its two header rows (a grouped header and a
   column header).
3. Read `(display name, creator, intelligence index)` from cell indices 0, 2 and
   3 — the header row reads
   `Model | Context Window | Creator | Artificial Analysis Intelligence Index | …`.
   Rows whose index cell is not an integer are skipped; that check is what
   discards the two header rows and any footer, so no positional row-slicing is
   relied on.
4. Slug the display name: strip the parenthetical variant, lowercase, drop
   non-alphanumerics. The variant label is the parenthetical text, or `default`.
5. Group rows by slug; keep the highest index; record the winning variant and
   the original AA display name.
6. Match slugs against both `name` and the `hf_repo` tail from `models.yaml`,
   with `instruct` / `it` / `chat` / `base` stripped from both sides.
7. Write `aa_scores.yaml` keyed by `hf_repo`.

**Openness is never read from AA.** Proprietary rows simply fail to match
`models.yaml`, so the project rule — open-weight iff a public weights repo
resolves on HF — remains the only authority on that question.

**Failure semantics.** A failed fetch leaves the existing sidecar untouched
rather than overwriting it with an empty one. This is the lesson from
`pull_leaderboard.py`, where a 429 destroyed freshly fetched scores, and from
`discover.py`, where staged candidates were wiped by a wholesale rewrite. An
empty parse (page fetched, zero rows recognised) is also treated as failure,
since it means the markup changed.

`--html <file>` parses a saved page offline. There is deliberately no
`--no-fetch` flag: its only effect would be to write an empty sidecar, which is
exactly the destructive behaviour this design exists to prevent.

### `aa_scores.yaml`

```yaml
# AUTO-SCRAPED by scripts/pull_aa.py — do not hand-edit.
# Artificial Analysis Intelligence Index, keyed by hf_repo.
# variant records which reasoning-effort row won (highest index).
scores:
  moonshotai/Kimi-K3:
    intelligence_index: 57
    variant: max
    aa_model: Kimi K3 (max)
    source: https://artificialanalysis.ai/leaderboards/models
unmatched:
  - Qwen3.7 Max
  - Gemma 4 31B
```

`unmatched` lists AA rows that matched no tracked model. It plays the role
`new_orgs` plays for arena: the signal that AA rates something worth promoting.
It is expected to be long — most AA rows are proprietary models the tracker will
never carry — so it is informational, never an error.

### `scripts/render_readme.py`

`load_aa_scores(path)` returns `{lower_repo: {"index": int, "variant": str}}`,
tolerating a missing, empty or malformed file by returning `{}`.

`aa_cell(model, aa)` returns the index as a string, or `—` when absent. There is
no fallback path — `models.yaml` no longer carries a score.

Final columns:

```
| Model | Developer | Released | Params | Context | Modality | Arena | AA Index | License | Commercial |
```

Prose note: AA Index is the Artificial Analysis Intelligence Index, a 0–100
composite (Agents 34%, Coding 24%, Scientific Reasoning 24%, General 18%);
`—` means AA does not currently rate that model.

## Testing

Fixture-driven, no network in any test, matching the existing arena tests.

- A trimmed AA page saved to `tests/fixtures/aa_leaderboard.html`.
- Parsing: correct row count, header rows skipped, non-integer index rows dropped.
- Variant selection: highest index wins and the variant label is recorded;
  a model with only `(max)` and `(low)` rows still resolves.
- Matching: `Llama 3.3 70B` matches `Llama 3.3 70B Instruct`; a proprietary row
  matches nothing; matching works via the `hf_repo` tail as well as `name`.
- `unmatched` captures rows that matched nothing.
- Failure preservation: a fetch exception and a zero-row parse both leave an
  existing `aa_scores.yaml` byte-identical.
- Render: `aa_cell` returns the index, and `—` when the repo is absent.

`validate.py` must pass on a `models.yaml` with no `benchmark` block, and the
`validate` workflow's README re-render diff must be clean.

## Consequences accepted

- **13 of 16 rows show `—`** until 2026 models are promoted. The table carries
  less data than before in exchange for carrying only trustworthy data.
- **Scraping is brittle.** One table, two header rows, index in column 4. An AA
  redesign breaks it; the fixture test detects that, and treating a zero-row
  parse as failure keeps a broken run from erasing good data.
- **`validate.py` gets weaker.** No benchmark figure is required of any row.
- **AA's methodology moves.** The index is versioned (v4.1 at time of writing)
  and re-weighted periodically, so scores are not comparable across time. Only
  the current snapshot is stored; no history is kept.

## Out of scope

- **Vals Index.** Its per-model pages are server-rendered and do carry the score
  (`/models/kimi_kimi-k3` → `Accuracy (Vals Index) 74.70%`), but the slugs are
  not derivable (`zhipu_glm-5.2` and `z-ai_glm-5.2` both 404), the leaderboard
  page server-renders only the top ~5, and `/models` exposes no usable links
  without JS. Adding it needs either a curated slug field or a headless browser
  in CI. Deferred to its own spec once the AA pattern is proven.
- Feeding AA into `discover.py` to prioritise candidates the way arena rank does.
- AA's Openness Index as a column, despite its obvious fit for this tracker.
- AA's cost, speed and latency columns.
