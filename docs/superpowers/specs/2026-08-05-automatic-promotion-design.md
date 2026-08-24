# Automatic promotion: enrich, classify, and keep the queue small

**Date:** 2026-08-05
**Status:** implemented on `feat/auto-promotion` (2026-08-05 → 2026-08-10). **The design below is as-approved; it was revised during implementation — see [Amendment](#amendment-2026-08-24) at the end before relying on any section.**

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

`names.repo_identity()` is **not** the right key here. It is an exact-model identity
that deliberately keeps size tokens, and it keeps version tokens too, so
`glm52` ≠ `glm51` and no collision would ever fire.

`family_stem()` is a new, coarser key: `repo_identity`'s output with **version tokens
additionally stripped**. A version token matches `^[A-Za-z]?\d+(\.\d+)*$` — an
optional leading letter then digits (`4`, `5.2`, `K3`, `V4`). Tokens ending in a
letter are sizes or expert counts, not versions, and are **kept** (`405B`, `17B`,
`16E`, `A22B`).

| repo | family stem | effect |
|---|---|---|
| `zai-org/GLM-5.2` vs `GLM-5.1` | `glm` vs `glm` | collide → review |
| `meta-llama/Llama-3.1-405B-Instruct` | `llama405b` | — |
| `meta-llama/Llama-3.1-8B-Instruct` | `llama8b` | distinct from 405B — both stay |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `llamascout17b16e` | — |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | `llamamaverick17b128e` | distinct from Scout — both stay |
| `moonshotai/Kimi-K3` vs `Kimi-K2.7-Code` | `kimi` vs `kimicode` | no collision |

Keeping sizes and distinguishing words matters: Llama-3.1-405B/8B and Llama-4
Scout/Maverick are all legitimately tracked today, and a stem that collapsed them
would send every sibling release to review, defeating the point of automating. The
rule fires on a *version bump at the same size* — which is exactly the
supersede-or-coexist judgement a human should make.

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

---

## Amendment 2026-08-24

Written after reviewing the shipped implementation on `feat/auto-promotion`
against the approved design above. Everything above this line is preserved
as-approved. This section records what changed, what the design got wrong, and
what is still open.

### The premise that broke: a signal is not a currency signal

The design treats an arena rank as self-refreshing evidence that a model is
current — the same way an AA score is. That was true when it was written: the
scraper read arena.ai's **Agent** board, 46 live models. Commit `797815a`
switched it to arena's **text** leaderboard, which is a *historical* ranking of
213 models going back to 2023 (#211 is `meta-llama/Llama-2-13b`).

An arena rank became evidence a model is **good**, not evidence it is
**current**. The design never separated those claims, and the gap was not
theoretical: a live `discover.py` run auto-promoted
`microsoft/Phi-3-mini-4k-instruct` (arena #186, released 2024-04-22) — ranked,
complete, schema-clean, and two years stale — along with 14 other 2023-2024
models. Downloads had the same defect for the same reason: it is a cumulative
lifetime count, so `Mistral-7B-v0.1` (2023, a *base* checkpoint) outranked
GLM-5.2 on raw downloads.

The correction splits one question into two, which the design had conflated:

| | question | function |
|---|---|---|
| **staging** | is this worth a human's attention in `candidates.yaml`? | `is_notable()` |
| **publication** | is this worth publishing to `models.yaml` with nobody looking? | `_clears_promotion_floor()` |

`is_notable` stays permissive — an old ranked model surfacing in the review
queue is the correct, harmless outcome. `_clears_promotion_floor` is strict.
Three gates that appear nowhere in the design above now do most of the real
work:

- **`NOTABILITY_DOWNLOADS_MAX_AGE_DAYS = 365`** — the downloads leg of
  *both* functions, and the arena leg of `_clears_promotion_floor` only
  (`2b4ecef`, `ac1dea1`). AA is exempt on both: Artificial Analysis genuinely
  delists models it no longer rates, so an `aa_index` being present at all is
  still evidence of currency.
- **`AA_PROMOTION_FLOOR = 20`** — an AA score below this stages the row but
  never publishes it (`5159bab`, moved from the notability bar to the promotion
  gate in `13133ea`). `granite-4.1-3b-base` at aa=5 is the case it exists to
  reject; the design had explicitly noted that row as one of only two complete
  signalled rows, without noticing that "AA rated it at all" is a weak reason to
  publish it.
- **`is_derivative_or_base()`** — distills and base checkpoints (`5159bab`).
  Four of the wrong auto-promotions were DeepSeek-R1 distills of an
  already-tracked model; three more were base checkpoints.

### `missing_vitals` has seven reasons, not five

The list in **Classification** above is stale. As shipped, in order:

1. `moe-active-params-unknown`
2. `no-context-window`
3. `license-not-allowlisted`
4. `inexact-repo-match`
5. `derivative-or-base` *(new — not in the design)*
6. `family-already-tracked`
7. `aa-below-promotion-floor` *(new — not in the design)*

A second gate the design also lacks: **`classify.schema_errors()`** (`a4a279e`)
runs `validate.row_errors()` — the exact per-row checks CI applies to
`models.yaml` — on a candidate before `route()` will promote it. The design
assumed candidate rows are machine-built and therefore well-formed. They are
not: `candidates.yaml` is hand-edited and carries forward, so a row can clear
every vitals check while holding `release_date: "sometime in 2025"`. Promoting
that appends it to `models.yaml` and then crashes `render_readme.py`'s date sort
with `TypeError` — and the render step in `discover.yml` has no
`continue-on-error`, so it kills the whole weekly PR with no PR ever opening.

### The queue-shrink prediction was wrong by 7×

> "The queue collapses 358 → ~12."

Actual, as of this amendment: **`models.yaml` 30 rows** (16 at design time),
**`candidates.yaml` 85 rows**. Reason counts across those 85 (a row can carry
several):

| reason | rows |
|---|---|
| `aa-below-promotion-floor` | 49 |
| `moe-active-params-unknown` | 47 |
| `schema-invalid: …` | 32 |
| `family-already-tracked` | 22 |
| `license-not-allowlisted` | 16 |
| `no-context-window` | 12 |
| `derivative-or-base` | 5 |
| `inexact-repo-match` | 1 |

The estimate assumed a 46-model arena board; the 213-model historical
leaderboard admits far more rows to staging, and the two new promotion gates
hold back rows the design would have published. The **direction** is right —
the flagships promoted, and no MoE row in `models.yaml` has
`params_active_b == params_total_b`, which was the design's central goal. The
magnitude is not, and the "first run is a step change, then it settles" framing
does not describe what happened.

### Fixed 2026-08-24: the queue-only-grows failure had re-emerged

The **Problem** section diagnoses this precisely — *"there is no representation
for 'no'"* — and the chosen fix, rebuilding the queue each run, only evicts rows
that **lose notability**. It does nothing for rows that are notable and
permanently blocked, and those are now the entire queue.

`family-already-tracked` is the clearest case: **22 of 85 rows**.
`allenai/Olmo-3.1-32B-Think` is notable, complete, schema-clean, and blocked
*solely* on colliding with the tracked `Olmo-3-32B-Think`. A human who reviews
it and decides "coexist" has nowhere to record that decision — `candidates.yaml`
is rebuilt each run and `models.yaml` is append-only — so the row regenerates
identically, forever, on every run.

The design put "deduplicating the families already tracked" out of scope. That
exclusion does not cover this: a **new release colliding with a tracked one** is
the pipeline's normal steady state, not a backlog item.

**The fix, shipped 2026-08-24.** `family_collision_reviewed: true`, set by a
human on the candidate row, is the reviewer's *coexist* answer coming back.
`classify.missing_vitals()` skips the collision check when it is set, so the
next run promotes the row and it leaves the queue;
`discover.PROMOTION_STRIP_FIELDS` drops the marker on the way into
`models.yaml`, so the field exists only for as long as the question does. No
new file, and carry-forward needed no change — `merge_candidates` already
copies staged rows whole, and a staged row is `known`, so nothing rebuilds
over it.

Two constraints the implementation is deliberate about:

- **`is True`, not truthiness.** `candidates.yaml` is hand-edited and
  `validate.py` never reads it, so this is the only place a typo can be caught.
  `family_collision_reviewed: "no"` is a truthy *string* — a bare `if` would
  read a reviewer's explicit rejection as approval and publish the row. A test
  pins this against the naive implementation.
- **It clears the collision only.** Every other reason still blocks promotion,
  so the marker can never become a blanket promote override.

*Supersede* remains out of scope and stays a hand edit: retiring a row would
break append-only, which the design correctly treats as load-bearing for
keeping `models.yaml` human-owned. `discover.py` only ever appends.

The 22 rows currently carrying `family-already-tracked` are unchanged — the
mechanism is now available, but each one is an editorial judgement nobody has
made yet.

### Smaller defects found in review, not yet fixed

- **`aa-below-promotion-floor` is misnamed** and it is the queue's top reason
  (49/85). It also fires on a stale arena rank and on stale downloads —
  `classify.py:266` concedes this in a comment. On a row with no AA score at
  all it points a reviewer at the wrong field. Splitting it into
  `signal-too-weak` / `signal-too-stale` would cost nothing.
- **`schema_errors()` double-reports.** 28 of its 32 entries restate a reason
  `missing_vitals` already gave (16 × `license-not-allowlisted`,
  12 × `no-context-window`), so the row carries both the terse reason and a
  `schema-invalid: …` echo of it. The layer is correct and necessary; it should
  suppress errors already named.
- **Enrichment shipped narrower than specified.** The design promised
  `context_window` from `tokenizer_config.json` **and the model card**. Only
  `enrich.context_from_tokenizer` exists; there is no card path. 12 rows are
  still parked on `no-context-window`.
- **`enrich.py`'s shape differs, benignly.** The design specified
  `active_params_from_card(repo, get_text)`; it shipped split into
  `fetch_card(repo, get_text)` + `active_params_from_card(text)`, which is
  easier to test and does not change behaviour. Multi-variant cards yielding
  two distinct figures correctly return `None` (`3106d54`).
- **`commercial_use_verified` defaulting lives in the renderer**, not in
  `validate.py` as the design suggested. `validate.row_errors` type-checks the
  field; `render_readme.commercial_badge` treats absent as unverified and
  appends `?`. Functionally what the design asked for.
