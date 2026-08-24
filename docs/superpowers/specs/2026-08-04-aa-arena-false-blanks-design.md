# Close the three false-blank paths in the AA / Arena render-time join

**Date:** 2026-08-04
**Status:** implemented 2026-08-04 (PR #6)

## Problem

A `—` in the README's **Arena** or **AA Index** column is supposed to mean one
thing: the upstream leaderboard does not rate that model. For all 22 rows
currently published that is true — every blank belongs to a pre-2026 model, and
neither `arena_agent_rankings.yaml` (44 rows, all current-generation) nor
`aa_scores.yaml` (22 scores, `unmatched` scanned) contains an entry for any of
them. The published table is honest today.

Three mechanisms in the code will nevertheless print `—` for a model that *is*
rated. Two are already sitting in `candidates.yaml`, waiting to be promoted.

### 1. Cross-source repo split

Both joins key on an exact, lowercased `hf_repo` string, and the two sources
disagree about which repo string identifies a model.

Arena resolved rank 34 to `deepseek-ai/DeepSeek-V4-Flash`. AA matched its score
of 50 to `deepseek-ai/DeepSeek-V4-Flash-0731`, the dated snapshot. Both exist as
separate rows in `candidates.yaml`, each carrying exactly one of the two
numbers, plus a third `-DSpark` row carrying neither. Whichever a reviewer
promotes, the other column renders `—` for a model both sources rate.

### 2. Arena rank lost to the quantization filter

Arena resolved rank 42 to `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`.
`nvfp4` is in `hf_meta.EXCLUDE_PATTERNS`, so `discover.py` drops the row — it is
the only one of the 14 arena-resolved repos absent from `candidates.yaml`. The
filter is right to reject a quantization; the cost is that the rank goes with
it, and no reviewer ever sees rank 42 in the queue.

### 3. Asymmetric org-prefix stripping in the arena name fallback

`render_readme.py` carries `_ORG_TAIL_WORDS` and `_arena_display_name()`, a
drifted duplicate of `pull_arena.py`'s `ORG_DISPLAY_ALIASES` and
`normalize_model_name()`. The renderer's copy strips a *trailing* org token but
not a *leading* one, so arena's `"Tencent Hy3"` slugs to `tencenthy3` and never
matches `models.yaml`'s name `"Hy3"` → `hy3`; likewise `"Thinking Machines
Inkling"` vs `"Inkling"`. `pull_arena.py` already solved this internally with
`without_leading_vendor()`; the renderer never got the fix.

Those two rows print a rank today only because `resolved_repo` is populated. The
name index exists precisely for the case where HF resolution failed transiently
(`render_readme.py:80-87`) — and for exactly these org-prefixed names, it will
not fire when it is needed.

## Decision

Add an **identity fallback** to the render-time join, fix the repo choice at the
source in `pull_arena.py`, and consolidate the normalization vocabulary into
`names.py` so the renderer and the scraper cannot drift again.

### Decisions taken, with rationale

| Decision | Choice | Why |
|---|---|---|
| Join policy for #1 | Automatic identity-slug fallback | Explicit `hf_repo_aliases` in the schema only fixes rows a human already noticed; the split is mechanical, so the join can close it. Provenance survives — `aa_scores.yaml` still records `aa_model`. |
| Does the identity key strip size? | **No** | See "The finding" below. This is the single most important constraint in the spec. |
| Ambiguous identity | Refuse to join, print | Consistent with `match_to_tracked`'s existing double-claim guard. A wrong number is worse than no number. |
| Fix location for #2 | `pull_arena.resolve_row` | Fixing the resolved repo at the source also repairs the review queue, which the render-time fallback cannot reach. |
| `pull_aa.py` matching | Unchanged | Its join has no reported defect (22 scores, zero orphans, no collisions), and re-proving all 22 matches needs a live AA scrape the sandbox blocks. |
| Shared-code home | `names.py` | `render_readme.py` deliberately imports only `names` (which imports only `re`) to stay offline; it cannot import from `pull_arena.py`. |
| `display_identity` return type | Ordered tuple, not a string | `_LEADING_ORG_PHRASES` contains `"mistral"`, so `"Mistral Small 3"` would reduce to `small3` — weak enough to collide with any other vendor's "Small 3". `score_match` already tries the full name first and the vendor-stripped form as a fallback; the shared function must preserve that. |

### The finding: two normalizations, not one

The first prototype of this design reused `pull_arena.strip_repo_decorations` as
the join key. Against real data it collapsed **`Llama-3.1-405B-Instruct` and
`Llama-3.1-8B-Instruct` onto one key** (`llama31`), and likewise
`Qwen2.5-72B`/`7B`, `gpt-oss-120b`/`20b`, `granite-4.1-30b`/`3b`.

That function strips size tokens — correct for its own job, wrong as a join key.
CLAUDE.md's "one row per model — a family flagship or *distinct sizes*" makes
size part of a model's identity in this dataset.

They therefore remain two functions with different contracts:

| | `strip_repo_decorations` (existing) | `repo_identity` (new) |
|---|---|---|
| Consumer | `pull_arena.score_match` | render-time joins |
| Strips size tokens | yes | **no** |
| Rationale | compares against a display name that may omit size; guarded by `_MIN_PREFIX_RATIO` and a high/medium/low rating | unguarded equality join across a whole file; size is identity |

With size preserved, the key produces **0 collisions** across every file that
feeds the join — `aa_scores.yaml` (22 repos), `models.yaml` (22), and
arena-resolved repos (14) — while still closing both target cases:

```
deepseek-ai/DeepSeek-V4-Flash-0731              -> deepseekv4flash          ┐ join
deepseek-ai/DeepSeek-V4-Flash                   -> deepseekv4flash          ┘
deepseek-ai/DeepSeek-V4-Flash-DSpark            -> deepseekv4flashdspark      distinct
nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4  -> nemotron3ultra550ba55b   ┐ join
nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B        -> nemotron3ultra550ba55b   ┘
meta-llama/Llama-3.1-405B-Instruct              -> llama31405b              ┐ stay
meta-llama/Llama-3.1-8B-Instruct                -> llama318b                ┘ distinct
```

`candidates.yaml` does show 54 identity collisions across its 358 rows (dated
snapshots and `-Base`/`-Instruct` pairs of the same weights). That file is not a
join source for the renderer, so it does not affect this design — but it is why
the collision guard is mandatory rather than defensive.

## Architecture

### `names.py` — the single normalization vocabulary

Still imports only `re`, preserving the renderer's offline guarantee. Absorbs
from `pull_arena.py`: `ORG_DISPLAY_ALIASES`, `_LEADING_ORG_PHRASES`,
`_PRECISION_TOKENS`, `_SIZE_TOKEN`, `normalize_model_name` (renamed
`normalize_display`), `without_leading_vendor`, `strip_repo_decorations`.
`pull_arena.py` imports them back, so its behaviour is unchanged.

Two new exports:

- `repo_identity(repo_id) -> str` — drops the author, a duplicated vendor
  prefix, and trailing precision / variant / date tokens. Keeps size tokens.
  Date token: a trailing all-digit token of length 2, 4, 6, or 8, stripped only
  while at least one token remains.
- `display_identity(display) -> tuple[str, ...]` — applies `normalize_display`
  first (drops everything from the first `·`, drops parenthetical effort tags,
  drops a trailing org alias), then returns ordered candidate keys: the full
  name, then the `without_leading_vendor` form, deduplicated. A name with no
  leading vendor therefore yields a 1-tuple.

### `render_readme.py` — identity fallback with a collision guard

Deletes `_ORG_TAIL_WORDS` and `_arena_display_name()` outright.

`load_aa_scores()` and `load_arena_ranks()` each build an additional
`identities` index at load time. Entries are grouped by `repo_identity`; any
identity claimed by two or more entries is **dropped from the index and
printed**, never joined.

Lookup order, with the field compared on each side made explicit:

| Step | `models.yaml` side | sidecar side |
|---|---|---|
| 1 | `hf_repo`, lowercased | sidecar key / `resolved_repo`, lowercased |
| 2 | `repo_identity(hf_repo)` | `repo_identity(` sidecar key / `resolved_repo` `)` |
| 3 (arena only) | `display_identity(name)`, any candidate | `display_identity(model)`, any candidate |

`aa_cell` stops after step 2 and returns `—`; `arena_cell` runs all three. Step 3
is arena-only because `aa_scores.yaml` is keyed by repo, and the AA display name
it records (`aa_model`) is provenance, not a join key.

The guard covers the sidecar side. Two `models.yaml` rows sharing an identity
would both claim the same sidecar entry; that is a duplicate-row bug, caught by
the `validate.py` check in "Also in scope" below rather than at render time.

### `pull_arena.py` — prefer un-quantized repos

`resolve_row`'s scoring loop breaks ties on confidence toward the repo with the
better format rank (see "Consequence for the un-quantized preference" below).
The existing early `break` on `best_conf == "high"` is removed: it
short-circuits before a better-format repo can be seen. Replaced by collecting
all hits and selecting `max` on `(confidence_rank, format_rank)`.

### Keeping the two quant vocabularies in step

**Amended 2026-08-04 after checking the data — the original form of this section
was wrong.** It called for a test asserting every `names.PRECISION_TOKENS` entry
is matched by `hf_meta.EXCLUDE_PATTERNS`. That test cannot pass, and should not:
`bf16`, `fp16`, and `fp32` are absent from `EXCLUDE_PATTERNS` deliberately.

NVIDIA publishes Nemotron 3 Ultra **only** as `-BF16` repos — there is no bare
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`. Six such repos sit in
`candidates.yaml` today. Adding `bf16` to `EXCLUDE_PATTERNS` would delete the
model from the tracker entirely.

The two vocabularies answer different questions. "Is this token a decoration to
ignore when comparing two names?" (`PRECISION_TOKENS`) is not "is this repo a
derivative to reject?" (`EXCLUDE_PATTERNS`). Neither set is a subset of the
other in either direction — `EXCLUDE_PATTERNS` also carries `lora`, `adapter`,
`reranker`, `merge`, none of which are precision markers.

`names.py` therefore splits the set to make the distinction nameable:

```python
NATIVE_FORMATS = {"bf16", "fp16", "fp32"}          # a vendor's primary release
QUANT_FORMATS  = {"fp8", "int4", "int8", "nvfp4", "mxfp8", "w4a16",
                  "w8a8", "4bit", "8bit", "gguf", "awq", "gptq"}
PRECISION_TOKENS = NATIVE_FORMATS | QUANT_FORMATS  # both are name decorations
```

The invariant that *does* hold, and is worth a test: every `QUANT_FORMATS`
entry must be matched by `EXCLUDE_PATTERNS` (verified — all 12 match as
`model-<token>`). A second test pins `NATIVE_FORMATS` as deliberately **not**
excluded, citing Nemotron 3 Ultra, so a future contributor cannot "tidy up" the
gap without deleting a model.

### Consequence for the un-quantized preference

The preference in `pull_arena.resolve_row` cannot be a boolean: when every
candidate carries a precision token, clean-vs-not cannot choose. It becomes a
three-level rank, applied after confidence:

| Rank | Condition | Example |
|---|---|---|
| 2 | no precision token | `zai-org/GLM-5.2` |
| 1 | `NATIVE_FORMATS` only | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` |
| 0 | any `QUANT_FORMATS` token | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` |

This resolves the Nemotron case at the source: the BF16 repo outranks the NVFP4
one, and `should_track` accepts it, so rank 42 reaches the review queue.

### Issue #2 is also self-healing via the identity fallback

`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` is already staged in
`candidates.yaml`, and `repo_identity` maps it to `nemotron3ultra550ba55b` — the
same identity as the NVFP4 repo arena resolved. So the render-time fallback
joins rank 42 onto it on promotion even before `pull_arena.py` is next run with
network access. The source fix above remains worth making: it repairs the review
queue, which the renderer cannot reach.

## Also in scope

Flagged during design as beyond the reported defects, and approved:

1. `validate.py` errors on a duplicate `repo_identity` within `models.yaml`,
   enforcing "one row per model" mechanically. `models.yaml` is clean today, so
   this costs nothing on landing.
2. `discover.py`'s existing skip message gains the arena rank:
   `- {repo} skipped ({reason})` → `- {repo} skipped ({reason}, arena rank 42)`.
   This covers the residual case where a vendor published only a quantized repo,
   so the preference in `pull_arena.py` has nothing clean to select.

## Error handling

Every new path degrades to current behaviour. An ambiguous identity yields `—`
plus a printed warning. A malformed sidecar keeps its existing `{}` return.
`names.py` gains no I/O. No new failure mode reaches CI.

## The honesty property

**This change must not alter today's README.** Verified against current data:
with the identity fallback active, 0 of 22 rows change any cell, so
`validate.yml`'s `git diff --exit-code README.md` step stays green.

The fix arms the join for the DeepSeek-V4-Flash and Nemotron-3-Ultra rows *when
they are promoted*. It invents no number now. This property is worth a
regression test, because a future change to `repo_identity` that quietly widens
matching would show up as a README diff rather than as silent data corruption.

## Testing

- `test_names.py` — `repo_identity` keeps size and drops precision, variant, and
  date tokens; `display_identity` ordering and dedup; the size-collision cases
  (`llama31405b` vs `llama318b`, `gptoss120b` vs `gptoss20b`) as explicit
  regression tests against the rejected first design.
- `test_render_readme.py` — identity fallback hits for the DeepSeek-Flash split;
  collision guard refuses and warns; org-prefixed display fallback resolves
  `"Tencent Hy3"` → `Hy3` and `"Thinking Machines Inkling"` → `Inkling`.
- `test_arena_resolve.py` — a clean repo beats a quantized one at equal
  confidence; a high-confidence quantized hit no longer short-circuits the
  search before a clean repo is examined.
- `test_hf_meta.py` — every `names.QUANT_FORMATS` entry matches
  `hf_meta.EXCLUDE_PATTERNS`; `names.NATIVE_FORMATS` is asserted **not** matched,
  with Nemotron 3 Ultra named in the docstring as the reason.
- `test_pull_aa.py` — unchanged. `pull_aa.py` is not touched.

## Out of scope

- **`pull_aa.py` matching.** Adopting `display_identity` there would gain
  vendor-prefix tolerance for AA rows like `"NVIDIA Nemotron 3 Nano"`, but none
  of those models are in `models.yaml` today, so the gain is latent. Deferred to
  a separate spec once a concrete missed match exists to point at.
- **`hf_repo_aliases` in the schema.** Rejected in favour of the automatic
  fallback. Worth revisiting only if a case appears that normalization cannot
  reach — a genuine rename, or a vendor re-upload under an unrelated name.
- **Re-running the scrapers.** `arena_agent_rankings.yaml` is marked
  AUTO-SCRAPED / do-not-hand-edit, and `arena.ai` and `huggingface.co` are
  unreachable from this sandbox. The Nemotron repo choice in committed data will
  only change on the next networked run of `pull_arena.py` (weekly CI, or a
  local run). The code fix and its tests land independently of that.
