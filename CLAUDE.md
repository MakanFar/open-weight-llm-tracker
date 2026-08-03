# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A curated, machine-readable index of open-weight LLMs. `models.yaml` is the **single source of truth**; the README table and everything else are generated from it. A set of Python scripts keep the index current by discovering new models on Hugging Face and staging them for human review.

## Commands

```bash
pip install -r requirements.txt

python -m pytest tests/ -v                    # full test suite (CI runs this)
python -m pytest tests/test_arena_resolve.py  # a single test file
python -m pytest tests/test_arena_resolve.py::test_name  # a single test

python scripts/validate.py        # schema-check models.yaml (CI gate; exits non-zero on any problem)
python scripts/render_readme.py   # regenerate README.md from models.yaml
python scripts/pull_hf.py         # dry-run: auto-fill fields from HF (--write to apply)
python scripts/pull_aa.py         # scrape AA Intelligence Index -> aa_scores.yaml
python scripts/discover.py        # org sweep + arena merge -> candidates.yaml
python scripts/pull_arena.py      # scrape arena leaderboard -> arena_agent_rankings.yaml
```

Note: `pytest` is not on the base interpreter here — use the repo's `.venv` (`.venv/bin/python -m pytest ...`) or a venv where `requirements.txt` is installed.

Network caveat: `pull_hf.py`, `discover.py`, and `pull_arena.py` all reach out to `huggingface.co` / `arena.ai`, which some sandboxes block. They are designed to run on a normal machine or in CI. `pull_arena.py --no-resolve` parses without any HF calls; `pull_arena.py --html page.html` runs fully offline from a saved page.

## Two-file data model

- **`models.yaml`** — the reviewed, published index. Only this file is validated and rendered. Fields are documented in [SCHEMA.md](SCHEMA.md).
- **`candidates.yaml`** — a staging queue of *unreviewed* auto-discovered models, written by `discover.py`. Rows here carry extra discovery-only fields (`discovered_via`, `arena_rank`, `downloads`, `needs_hf_repo`, `resolution_confidence`). **These fields must be stripped when a row is promoted into `models.yaml`** — `validate.py` only checks `models.yaml`, so they would otherwise leak in. Nothing automated ever writes to `models.yaml`; promotion is always a human editing the file.

## Discovery pipeline (the core architecture)

The point of the pipeline is to surface frontier releases without a human hand-watching Hugging Face, while keeping a human as the final gate. Understanding it requires reading `discover.py`, `pull_arena.py`, and `hf_meta.py` together.

**`hf_meta.py`** is the shared construction path. Both discovery sources build candidate rows through `should_track()` (the filter) and `candidate_from_repo()` (the row factory), so a candidate has exactly one set of filter rules and one shape. `EXCLUDE_PATTERNS` drops quants/adapters/merges/non-generative heads; `ACCEPTED_LICENSES` and `COMMERCIAL_GUESS` are HF's license vocabulary (not clean SPDX), and every commercial-use value is a *guess* a reviewer must re-check.

**`discover.py`** does an **org sweep**, not a global scan. It issues one `list_models(author=org)` call per entry in `ORG_ALLOWLIST`. This is deliberate: sorting all of HF by upload date returns finetunes and quantizations and essentially never a frontier release. **Adding an org to `ORG_ALLOWLIST` is how the tracker gains coverage.** A failure on one org is logged and skipped, never aborting the sweep (`list_models()` is a generator, so the call is wrapped in `list(...)` inside the `try` to force the HTTP request where the `except` can catch it).

**`pull_arena.py`** scrapes the arena.ai Agent Arena leaderboard and resolves each ranked model's display name to an HF repo. Key rule: **open-weight status is decided by whether a public weights repo actually resolves on HF** — never from the org name and never from arena's license label. Name→repo matching uses `score_match()` with `_MIN_PREFIX_RATIO` (a prefix must account for ≥70% of the longer slug, which rejects brand-prefix noise like `Grok 4.5` vs `grok`). `HF_AUTHOR_HINTS` scopes the HF search per org; an org mapped to `None` means "no HF namespace, never search". Inexact matches are flagged `needs_hf_repo: true` for a human to confirm. `KEYWORD_MAP` is only a search hint — it makes no open-weight claim.

**The merge.** `discover.py`'s `main()` calls `load_arena()` + `arena_candidates()` and merges arena-resolved repos into the org-sweep candidates, deduping on lowercased `hf_repo` and sorting so arena-ranked models lead the review queue (best rank first), unranked org-sweep finds following newest-first. **Arena is never load-bearing**: a missing, empty, or malformed `arena_agent_rankings.yaml` degrades gracefully to an org-sweep-only run, and `discover.py` always exits 0. `load_arena()` validates shape after parsing so valid-YAML-wrong-shape files can't crash it.

**Widening coverage.** An allowlist only finds what it knows. `pull_arena.py` records leaderboard orgs it has no mapping for in the `new_orgs` key of `arena_agent_rankings.yaml`, and `discover.py` reprints them — the signal to add an org to `ORG_ALLOWLIST` (and its namespace to `HF_AUTHOR_HINTS`). `unmapped_orgs()` computes this over *every* parsed row, not just resolved ones, so it surfaces exactly the orgs lacking a mapping.

## README is fully generated — don't hand-edit it

`render_readme.py` rewrites the **entire** `README.md`, not just the region between the `MODELS_TABLE` markers. The surrounding prose is the `body` string inside `render_readme.py:main()`. Any hand-edit to `README.md` — table or prose — is silently reverted on the next render, and the `validate` workflow's `git diff --exit-code README.md` step then fails. **To change README prose, edit the `body` string in `render_readme.py`.**

## CI

- **`validate.yml`** (every PR + push to main): runs `pytest`, then `validate.py`, then re-renders the README and fails if it differs from the committed one.
- **`discover.yml`** (weekly, Mondays 13:00 UTC + manual dispatch): runs `pull_arena.py` (with `continue-on-error` so arena never blocks discovery) then `discover.py`, and opens a PR with the updated `candidates.yaml`. **The PR is the human-in-the-loop approval step** — a reviewer fills the `TODO` fields, confirms any `needs_hf_repo: true` row, moves approved rows into `models.yaml` (stripping discovery-only fields), and merges.

## Conventions that keep the data trustworthy

- **No benchmark field.** The anchor number is the Artificial Analysis Intelligence Index in `aa_scores.yaml`, written by `scripts/pull_aa.py` and joined on `hf_repo` at render time. AA covers recent models only, so `—` is expected and correct for older rows.
- **Set `commercial_use` by reading the actual license**, not the word "open". Values are `true` / `false` / `conditional`.
- **One row per model** — a family flagship or distinct sizes, never both.
- **MoE**: total params in `params_total_b`, routed/active in `params_active_b`; for dense models the two are equal (validated).
- New license strings must be added to the allowlist in `validate.py` in the same change.
