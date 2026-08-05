# Automatic promotion: enrich, classify, and keep the queue small

**Date:** 2026-08-05
**Status:** approved design, not yet implemented

## Problem

Every discovered model lands in `candidates.yaml` and waits for a human to hand-edit
it into `models.yaml`. That has three failures.

**The queue only grows.** Rows leave only by promotion. A hand-deleted row is
re-discovered on the next run, because `known = tracked ∪ staged` and a deleted row
is in neither — there is no representation for "no". After a back-catalogue backfill
the queue reached 358 rows, essentially all of which need no decision.

**The review burden is on the wrong rows.** Of 358 staged rows, 341 carry no
third-party signal at all. The reviewer's attention is spent skipping research
artifacts rather than judging frontier releases.

**The models worth promoting are exactly the ones that cannot be promoted.**
Cross-tabulating notability against metadata completeness:

| | complete | incomplete |
|---|---|---|
| **has signal** (AA index or arena rank) | 2 | 15 |
| **no signal** | 110 | 231 |

A naive "promote what is complete" rule would publish 112 rows — `MagenticBrain`,
`HARC-Qwen2.5-7B-Instruct`, `BAR-7B`, granite previews — while blocking Kimi K3,
GLM-5.2, DeepSeek-V4, gpt-oss-120b and Qwen3-Next. **15 of the 17 signalled rows are
blocked by one field: `params_active_b` on an MoE row.** Every frontier model is MoE.

The two signalled rows that *are* complete are `granite-4.1-30b-base` and
`granite-4.1-3b-base` — AA indices of 9 and 5, the least notable of the set.

So this is not primarily a gating problem. It is an **enrichment** problem: gate
harder and you publish noise, enrich harder and the flagships become promotable.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Where the human gate lives | PR review; nothing commits to `main` directly | Softens the invariant from "never automated" to "only via PR". A wrong claim is still caught before it is published. |
| Notability bar | AA index **or** arena rank **or** ≥500k HF downloads | 45 of 358 clear it. Signal-only (17 rows) is a bar the project's own curation rejects: 13 of the 16 currently tracked models would fail it. |
| `commercial_use` | Auto-promote the inferred value, flagged as unverified | User's call, made knowingly. Mitigated by rendering unverified values distinguishably. |
| Cleanup | Rebuild the queue each run: keep only notable-and-incomplete | Cleanup is emergent. No expiry timers, no reject list, no accumulation. |

## Architecture

```
pull_arena.py ─┐
pull_aa.py    ─┤
               ├─→ discover.py ─→ classify ─→ ┬─ drop (not notable — never staged)
enrich.py     ─┘                              ├─ models.yaml     (notable + complete)
                                              └─ candidates.yaml (notable + gap)
                                                       ↓
                                                 one PR, both files
```

### `scripts/enrich.py` (new)

Best-effort field completion. Every function returns a value **and its provenance**,
or `None` — never a guess dressed as a fact.

- `active_params_from_card(repo, get_text)` → `(billions, quoted_source)` or `None`.
  Regexes vendor phrasings found in real cards: `"1.6T parameters (49B activated)"`,
  `"~428B parameters and ~23B activated"`, `"21B active parameters"`,
  `"Activated Parameters | 104B"`. Hand-testing hit 5 of 7 models. It must return
  `None` rather than computing from expert geometry: routing is not a simple ratio,
  and `validate.py` only enforces `active == total` for *dense* rows, so a wrong MoE
  figure passes silently and is indistinguishable from a real one.
- `license_string(info)` → the real licence identifier. When the HF tag is `other`,
  read `cardData.license_name` (this is how `kimi-k3`, `minimax-community`,
  `modified-mit`, `nvidia-open-model-license` were recovered). Also maps HF tag
  spellings to `validate.py` allowlist spellings (`llama3.1` → `llama-3.1-community`),
  which alone unblocks the 39 rows currently failing on Llama tags.
- `context_window(repo, info, get_json)` → extends the existing `config.json`
  fallback to `tokenizer_config.json` and the model card, for the 80 rows at 0.

### Classification

```python
is_notable(row)      # aa_index or arena_rank or downloads >= NOTABILITY_DOWNLOADS
missing_vitals(row)  # -> list[str], empty means promotable
```

`missing_vitals` reports, in this order:

1. `moe-active-params-unknown` — `architecture == "moe"` and `params_active_b == params_total_b`
2. `no-context-window` — `context_window` not a positive int
3. `license-not-allowlisted` — licence absent from `validate.LICENSES`
4. `inexact-repo-match` — `needs_hf_repo` is true
5. `family-already-tracked` — see below

Routing: `notable and not missing_vitals` → promote; `notable and missing_vitals` →
review; otherwise → not staged.

### Two invariants this change introduces

**Append-only.** Auto-promotion may only *add* rows to `models.yaml`. It must never
modify, reorder or delete an existing row. Everything already there stays exactly as
a human left it — that is what keeps the file human-owned despite a script writing to
it. A test must assert that promoting into a populated `models.yaml` leaves every
pre-existing row byte-identical.

**Family collisions route to review.** `GLM-5.1` (rank 22) and `GLM-5.2` (aa 51,
rank 12) are both notable; promoting both breaks the project's "one row per model"
convention. If a promotable row's family stem already appears in `models.yaml`, it is
routed to review with `family-already-tracked` so a human decides supersede-or-coexist.

The stem is `developer` plus the model name with **version tokens only** removed —
a token that is digits, dotted digits, or a single letter followed by digits
(`5.2`, `K3`, `V4`, `4`) — and with size and variant tokens (`70B`, `A22B`,
`Instruct`, `it`, `Base`) removed. Alphabetic distinguishing words are **kept**:

| repo | stem | effect |
|---|---|---|
| `zai-org/GLM-5.2` vs `GLM-5.1` | `zai-org/glm` | collide → review |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `meta-llama/llama-scout` | no collision |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | `meta-llama/llama-maverick` | with Scout — both stay |
| `moonshotai/Kimi-K3` vs `Kimi-K2.7-Code` | `moonshotai/kimi-k` vs `kimi-k-code` | no collision |

Keeping distinguishing words matters: `Llama 4 Scout` and `Llama 4 Maverick` are both
legitimately tracked today, and a stem that collapsed them to `llama` would send every
sibling release to review, defeating the point of automating. Over-collision is safe
but costly; this rule errs toward promoting siblings and catching version bumps.

### `commercial_use`

Auto-promoted rows carry the `COMMERCIAL_GUESS` value plus a new schema field
`commercial_use_verified: false`. `render_readme.py` marks unverified values
distinguishably, the way vendor-reported MMLU figures once carried `*`, so a reader
can tell an inferred claim from a checked one. A human setting it to `true` is an
ordinary edit that append-only preserves.

This is the highest-risk element of the design and it is a deliberate, informed
trade: an unverified legal claim can reach the README, caught only by PR review.

`commercial_use_verified` is a new `models.yaml` field and must be added to
`SCHEMA.md`. `validate.py` needs no change to accept it — it checks required fields
and does not reject extras — but it should treat the field as optional and default it
to `false` when absent, so the 16 existing hand-curated rows are not retroactively
marked verified without anyone having checked them.

## Data flow and files

| File | Written by | Human-editable |
|---|---|---|
| `models.yaml` | humans; **and** discover.py, append-only | yes — always wins |
| `candidates.yaml` | discover.py, rebuilt each run | yes — edits carry forward until the row promotes |
| `aa_scores.yaml`, `arena_agent_rankings.yaml` | scrapers | no |

A human filling a gap in `candidates.yaml` (e.g. typing the real `params_active_b`)
is the intended workflow: the row completes, and the next run promotes it out.

## Error handling

Enrichment failure is never fatal and never fabricates. A model card that cannot be
fetched, or has no activation figure, yields `None`, which becomes a `missing_vitals`
entry and routes the row to review — the same path as any other gap.

Scraper failures already preserve their committed sidecars; that behaviour is
unchanged. A run with no `aa_scores.yaml` simply has fewer notable rows.

## Testing

Fixture-driven, no network:

- `active_params_from_card` against saved card excerpts covering all four observed
  phrasings, plus a card with no figure (must return `None`, not 0).
- `license_string` for tag-is-`other` and for each Llama tag mapping.
- `is_notable` at the boundary (499_999 / 500_000) and via each signal independently.
- `missing_vitals` returns every applicable reason, not just the first.
- Family collision: a promotable `GLM-5.2` with `GLM-5.1` already tracked routes to
  review.
- **Append-only**: promoting into a populated `models.yaml` leaves pre-existing rows
  byte-identical, including hand-edited fields.
- Queue rebuild: a staged row that loses notability disappears; a staged row whose
  gap a human filled promotes and leaves the queue.
- End to end: `validate.py` passes on an auto-promoted `models.yaml`, and the README
  re-renders with no diff.

## Consequences accepted

- **`models.yaml` stops being exclusively human-written.** The PR is now the only
  thing between an inferred field and the published index.
- **Unverified `commercial_use` can be published.** Flagged and rendered as such, but
  a reader who ignores the marker sees a claim nobody checked.
- **The notability bar is a popularity proxy.** Downloads skew small — `gpt-oss-20b`
  (8.5M) outranks `GLM-5.2` (1.7M) — so a genuinely important large model with modest
  adoption and no leaderboard coverage will not auto-promote. It stays discoverable
  via the queue only if something else marks it notable.
- **First run is a step change.** The queue collapses 358 → ~12 and `models.yaml`
  gains a batch of rows at once. That PR deserves real review, not a skim.

## Out of scope

- Retiring or updating rows already in `models.yaml` (append-only by construction).
- Deduplicating the families already tracked.
- Verifying `commercial_use` automatically by parsing licence text.
- Backfilling `params_active_b` for the currently held rows (GLM-5.2,
  DeepSeek-V4-Flash-0731) — enrichment may resolve them; if not they stay in review.
