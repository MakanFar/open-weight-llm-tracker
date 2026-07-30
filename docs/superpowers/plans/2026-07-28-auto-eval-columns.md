# Auto MMLU + Agent Arena Columns and context_window Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auto-populated MMLU and Agent Arena columns to the README (sourced from committed sidecar files, joined at render), and stop `discover.py` from emitting `context_window: 0`.

**Architecture:** All network access stays in fetch scripts whose output is committed; `render_readme.py` reads only committed files so it remains deterministic and offline (the CI `git diff` gate depends on this). A new `scripts/pull_leaderboard.py` writes `leaderboard_scores.yaml`; arena rank is read from the existing `arena_agent_rankings.yaml`; `render_readme.py` joins both by lowercased `hf_repo`. A shared `config.json` context fetch in `hf_meta.py` fixes discovery's `context_window`.

**Tech Stack:** Python 3.9+ (repo `.venv`), PyYAML, `urllib` (stdlib, for HTTP), pytest. No new dependencies.

## Global Constraints

- **Run everything with the repo venv:** `.venv/bin/python -m pytest ...`, `.venv/bin/python scripts/...`. Bare `python`/`pytest` lacks the deps.
- **`render_readme.py` must never make a network call.** It reads only committed files. Any live data comes pre-fetched from a committed sidecar.
- **Nothing automated writes to `models.yaml`.** It stays human-curated; auto values live only in sidecars and are joined at render.
- **`models.yaml` `benchmark.score` stays required** in `validate.py` and is the MMLU fallback — do not remove or relax it.
- **Graceful degradation:** a missing, empty, or malformed sidecar must never raise out of render — fall back to the manual score / a blank Arena cell. Mirror the shape-validation style already in `discover.py:load_arena()`.
- **No new pip dependencies.** Use `urllib.request` for HTTP, matching `scripts/pull_hf.py`.

---

### Task 1: Shared `config.json` context fetch in `hf_meta.py`

Add a pure, injectable helper that resolves a repo's context window from
`config.json`, isolating the network call so it is unit-testable offline.

**Files:**
- Modify: `scripts/hf_meta.py` (add `_http_get_json`, `fetch_context_window`)
- Test: `tests/test_hf_meta.py`

**Interfaces:**
- Produces:
  - `hf_meta._http_get_json(url) -> dict` — default HTTP JSON getter (urllib).
  - `hf_meta.fetch_context_window(repo, get_json=_http_get_json) -> int | None` —
    fetches `https://huggingface.co/{repo}/resolve/main/config.json`, returns the
    first present int among `max_position_embeddings`, `max_sequence_length`,
    `n_positions` (including nested `text_config` / `llm_config`), else `None`.
    Never raises: any exception from `get_json` yields `None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hf_meta.py`:

```python
import hf_meta


def test_fetch_context_window_reads_top_level_key():
    cfg = {"max_position_embeddings": 131072}
    ctx = hf_meta.fetch_context_window("org/model", get_json=lambda url: cfg)
    assert ctx == 131072


def test_fetch_context_window_reads_nested_text_config():
    cfg = {"text_config": {"max_sequence_length": 8192}}
    ctx = hf_meta.fetch_context_window("org/model", get_json=lambda url: cfg)
    assert ctx == 8192


def test_fetch_context_window_returns_none_when_absent():
    ctx = hf_meta.fetch_context_window("org/model", get_json=lambda url: {"foo": 1})
    assert ctx is None


def test_fetch_context_window_swallows_fetch_errors():
    def boom(url):
        raise RuntimeError("gated repo")
    assert hf_meta.fetch_context_window("org/model", get_json=boom) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hf_meta.py -k fetch_context_window -v`
Expected: FAIL — `AttributeError: module 'hf_meta' has no attribute 'fetch_context_window'`

- [ ] **Step 3: Write the minimal implementation**

Add to `scripts/hf_meta.py`. Put the imports at the top with the existing imports:

```python
import json
import urllib.request
```

Add these near the other helpers (after `context_of`, before `should_track`):

```python
_CONFIG_URL = "https://huggingface.co/{repo}/resolve/main/config.json"


def _http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "owlt-puller/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _ctx_from_config(cfg):
    if not isinstance(cfg, dict):
        return None
    for k in CTX_KEYS:
        if isinstance(cfg.get(k), int):
            return cfg[k]
    for sub in ("text_config", "llm_config"):
        inner = cfg.get(sub) or {}
        if isinstance(inner, dict):
            for k in CTX_KEYS:
                if isinstance(inner.get(k), int):
                    return inner[k]
    return None


def fetch_context_window(repo, get_json=_http_get_json):
    """Best-effort context length from the repo's config.json. None on any failure."""
    try:
        cfg = get_json(_CONFIG_URL.format(repo=repo))
    except Exception:
        return None
    return _ctx_from_config(cfg)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hf_meta.py -k fetch_context_window -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/hf_meta.py tests/test_hf_meta.py
git commit -m "feat(hf_meta): add injectable config.json context fetch"
```

---

### Task 2: Use the config fetch in discovery so candidates get a real context_window

Thread the fetch into `candidate_from_repo` and the two discovery call sites so a
candidate whose API `expand` lacks context falls back to `config.json` instead of
landing as `0`.

**Files:**
- Modify: `scripts/hf_meta.py` (`candidate_from_repo` gains a `context_window` param)
- Modify: `scripts/discover.py` (`sweep_orgs`, `arena_candidates` resolve context and pass it)
- Test: `tests/test_hf_meta.py`, `tests/test_discover_sweep.py`

**Interfaces:**
- Consumes: `hf_meta.fetch_context_window`, `hf_meta.context_of` (Task 1 + existing).
- Produces:
  - `hf_meta.candidate_from_repo(info, discovered_via, arena_rank=None, needs_hf_repo=None, resolution_confidence=None, context_window=None)`
    — when `context_window` is not `None`, the row uses it; otherwise it uses
    `context_of(info) or 0` (unchanged legacy behavior).
  - `hf_meta.resolve_context(info, get_json=_http_get_json) -> int` — returns
    `context_of(info)` if present, else `fetch_context_window(info.id, get_json)`,
    else `0`.
  - `discover.sweep_orgs(api, orgs, min_params, known, get_json=hf_meta._http_get_json)`
    and `discover.arena_candidates(api, rows, min_params, known, get_json=hf_meta._http_get_json)`
    — both now accept an injectable `get_json` used for the context fallback.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hf_meta.py` (uses the existing fake ModelInfo builder in that
file; if none exists, use a `types.SimpleNamespace`):

```python
import types


def _info(repo_id, config=None):
    return types.SimpleNamespace(
        id=repo_id, config=config or {}, safetensors={"total": 7_000_000_000},
        card_data={"license": "apache-2.0"}, created_at="2026-01-01T00:00:00Z",
        downloads=10,
    )


def test_resolve_context_prefers_api_expand():
    info = _info("org/m", config={"max_position_embeddings": 4096})
    assert hf_meta.resolve_context(info, get_json=lambda url: {"n_positions": 999}) == 4096


def test_resolve_context_falls_back_to_config_json():
    info = _info("org/m", config={})  # API expand empty
    ctx = hf_meta.resolve_context(info, get_json=lambda url: {"max_position_embeddings": 32768})
    assert ctx == 32768


def test_resolve_context_zero_when_nothing_resolves():
    info = _info("org/m", config={})
    assert hf_meta.resolve_context(info, get_json=lambda url: {}) == 0


def test_candidate_uses_explicit_context_window():
    info = _info("org/m", config={})
    row = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"], context_window=65536)
    assert row["context_window"] == 65536
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hf_meta.py -k "resolve_context or explicit_context" -v`
Expected: FAIL — `AttributeError: module 'hf_meta' has no attribute 'resolve_context'`

- [ ] **Step 3: Write the minimal implementation**

In `scripts/hf_meta.py`, add after `fetch_context_window`:

```python
def resolve_context(info, get_json=_http_get_json):
    """API expand first, then config.json, then 0."""
    ctx = context_of(info)
    if isinstance(ctx, int) and ctx > 0:
        return ctx
    return fetch_context_window(info.id, get_json) or 0
```

Change `candidate_from_repo`'s signature and the `context_window` line:

```python
def candidate_from_repo(info, discovered_via, arena_rank=None,
                        needs_hf_repo=None, resolution_confidence=None,
                        context_window=None):
```

Replace the existing line:

```python
        "context_window": context_of(info) or 0,   # 0 => fill during review
```

with:

```python
        "context_window": context_window if context_window is not None
        else (context_of(info) or 0),   # 0 => fill during review
```

- [ ] **Step 4: Wire the discovery call sites**

In `scripts/discover.py`, change `sweep_orgs` to accept and use `get_json`:

```python
def sweep_orgs(api, orgs, min_params, known, get_json=hf_meta._http_get_json):
```

Replace the append inside the `for info in models:` loop:

```python
            candidates.append(
                hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"]))
```

with:

```python
            ctx = hf_meta.resolve_context(info, get_json)
            candidates.append(hf_meta.candidate_from_repo(
                info, discovered_via=["org-sweep"], context_window=ctx))
```

Change `arena_candidates` similarly:

```python
def arena_candidates(api, rows, min_params, known, get_json=hf_meta._http_get_json):
```

Replace its append:

```python
        out.append(hf_meta.candidate_from_repo(
            info, discovered_via=["arena"], arena_rank=row.get("rank"),
            needs_hf_repo=row.get("needs_hf_repo"),
            resolution_confidence=row.get("resolution_confidence")))
```

with:

```python
        ctx = hf_meta.resolve_context(info, get_json)
        out.append(hf_meta.candidate_from_repo(
            info, discovered_via=["arena"], arena_rank=row.get("rank"),
            needs_hf_repo=row.get("needs_hf_repo"),
            resolution_confidence=row.get("resolution_confidence"),
            context_window=ctx))
```

- [ ] **Step 5: Update existing discovery tests to inject a no-network get_json**

In `tests/test_discover_sweep.py`, any call to `sweep_orgs(...)` must pass a fake
so no test hits the network. Find each `discover.sweep_orgs(api, ...)` call and add
`get_json=lambda url: {}` as the final argument. Example:

```python
cands, skips = discover.sweep_orgs(api, ["Qwen"], 3.0, set(), get_json=lambda url: {})
```

Add one test proving the config fallback reaches the candidate. Make one fake
model's API `config` empty and return context via `get_json`:

```python
def test_context_window_filled_from_config_json(monkeypatch):
    api = FakeApi(models={"Qwen": [_fake_info("Qwen/Qwen9-30B", config={})]})
    cands, _ = discover.sweep_orgs(
        api, ["Qwen"], 3.0, set(),
        get_json=lambda url: {"max_position_embeddings": 40960})
    assert cands[0]["context_window"] == 40960
```

(Adjust `FakeApi` / `_fake_info` to match the helpers already in that test file;
if `_fake_info` doesn't take `config`, extend it to set `.config`.)

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_hf_meta.py tests/test_discover_sweep.py tests/test_discover_merge.py -v`
Expected: PASS (all existing + new tests)

- [ ] **Step 7: Commit**

```bash
git add scripts/hf_meta.py scripts/discover.py tests/test_hf_meta.py tests/test_discover_sweep.py
git commit -m "feat(discover): fill context_window from config.json instead of 0"
```

---

### Task 3: `pull_leaderboard.py` writes `leaderboard_scores.yaml`

New fetcher that maps each `models.yaml` repo to an MMLU score from the HF Open
LLM Leaderboard, with the network call isolated behind an injectable function so
the parse/map logic is fully tested offline.

**Files:**
- Create: `scripts/pull_leaderboard.py`
- Create: `leaderboard_scores.yaml` (seed committed as `scores: {}` in Task 5)
- Modify: `SCHEMA.md` (document the sidecar)
- Test: `tests/test_pull_leaderboard.py`

**Interfaces:**
- Produces:
  - `pull_leaderboard.extract_mmlu(row) -> float | None` — reads an MMLU value
    from a leaderboard row dict; tolerant of key casing, returns `None` if absent.
  - `pull_leaderboard.build_scores(repos, rows) -> dict` — `{repo: {"mmlu": float, "source": URL}}`
    for repos present in `rows` with an MMLU; repos absent or without MMLU are omitted.
    Join is case-insensitive on repo id.
  - `pull_leaderboard.LEADERBOARD_URL` — the source URL string stored in each entry.
  - `pull_leaderboard.fetch_rows(get_json=hf_meta._http_get_json) -> list[dict]` —
    thin live fetch (verify against the live API on first real run; parse logic is
    what's tested).
  - `pull_leaderboard.main()` — reads `models.yaml` repos, fetches, writes
    `leaderboard_scores.yaml`; logs and exits 0 on any fetch failure.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pull_leaderboard.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pull_leaderboard as pl


def test_extract_mmlu_reads_various_key_casings():
    assert pl.extract_mmlu({"MMLU": 78.6}) == 78.6
    assert pl.extract_mmlu({"mmlu": 78.6}) == 78.6
    assert pl.extract_mmlu({"MMLU-PRO": 50.0, "MMLU": 78.6}) == 78.6  # prefers plain MMLU
    assert pl.extract_mmlu({"other": 1}) is None


def test_build_scores_maps_repos_case_insensitively():
    rows = [
        {"fullname": "meta-llama/Llama-3.1-405B-Instruct", "MMLU": 88.6},
        {"fullname": "some/UnlistedModel", "MMLU": 10.0},
    ]
    repos = ["meta-llama/Llama-3.1-405B-Instruct", "google/gemma-2-27b-it"]
    scores = pl.build_scores(repos, rows)
    assert scores == {
        "meta-llama/Llama-3.1-405B-Instruct": {"mmlu": 88.6, "source": pl.LEADERBOARD_URL}
    }


def test_build_scores_skips_rows_without_mmlu():
    rows = [{"fullname": "org/m", "MMLU-PRO": 40.0}]
    assert pl.build_scores(["org/m"], rows) == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pull_leaderboard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pull_leaderboard'`

- [ ] **Step 3: Write the implementation**

Create `scripts/pull_leaderboard.py`:

```python
#!/usr/bin/env python3
"""
Fetch MMLU scores for tracked models from the HF Open LLM Leaderboard and write
them to a committed sidecar, leaderboard_scores.yaml, keyed by hf_repo.

WHY A SIDECAR:
    render_readme.py must stay offline and deterministic (CI re-renders and
    diffs the README). So all network access lives here; render only reads the
    committed file. Nothing is written to models.yaml — it stays human-curated.

COVERAGE CAVEAT:
    The Open LLM Leaderboard v2 was archived; many 2026 frontier models are not
    listed. render_readme.py falls back to the manual benchmark.score for any
    repo missing here, so an empty or partial file is safe.

Usage:
  .venv/bin/python scripts/pull_leaderboard.py            # fetch + write
  .venv/bin/python scripts/pull_leaderboard.py --no-fetch # rewrite from empty (offline)
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_meta

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
OUT = ROOT / "leaderboard_scores.yaml"

LEADERBOARD_URL = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
# HF datasets-server rows endpoint for the aggregated leaderboard contents.
# Verify field names against the live response on first real run; the parse
# logic below is tolerant and fully unit-tested with fixtures.
ROWS_URL = ("https://datasets-server.huggingface.co/rows"
            "?dataset=open-llm-leaderboard/contents&config=default&split=train"
            "&offset={offset}&length=100")
_REPO_KEYS = ("fullname", "eval_name", "model", "Model")


def extract_mmlu(row):
    """Read a plain-MMLU score from a leaderboard row, tolerant of key casing."""
    for k in row:
        if k.lower() == "mmlu":
            v = row[k]
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _repo_of(row):
    for k in _REPO_KEYS:
        v = row.get(k)
        if isinstance(v, str) and "/" in v:
            return v
    return None


def build_scores(repos, rows):
    """{repo: {mmlu, source}} for tracked repos found in rows with an MMLU."""
    by_repo = {}
    for row in rows:
        repo = _repo_of(row)
        if repo:
            mmlu = extract_mmlu(row)
            if mmlu is not None:
                by_repo[repo.lower()] = mmlu
    out = {}
    for repo in repos:
        mmlu = by_repo.get(repo.lower())
        if mmlu is not None:
            out[repo] = {"mmlu": mmlu, "source": LEADERBOARD_URL}
    return out


def fetch_rows(get_json=hf_meta._http_get_json):
    """Best-effort paginated fetch of leaderboard rows. [] on failure."""
    rows = []
    offset = 0
    while True:
        try:
            page = get_json(ROWS_URL.format(offset=offset))
        except Exception as exc:
            print(f"  ! leaderboard fetch failed at offset {offset}: {exc}")
            break
        items = (page or {}).get("rows") or []
        if not items:
            break
        rows.extend(it.get("row", it) for it in items)
        if len(items) < 100:
            break
        offset += 100
    return rows


def _tracked_repos():
    doc = yaml.safe_load(DATA.read_text()) or {}
    return [m["hf_repo"] for m in doc.get("models", []) if m.get("hf_repo")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true",
                    help="write scores: {} without any network call")
    args = ap.parse_args()

    repos = _tracked_repos()
    rows = [] if args.no_fetch else fetch_rows()
    scores = build_scores(repos, rows)

    header = ("# AUTO-FETCHED by scripts/pull_leaderboard.py — do not hand-edit.\n"
              "# MMLU from the HF Open LLM Leaderboard, keyed by hf_repo.\n"
              "# render_readme.py falls back to models.yaml benchmark.score when a\n"
              "# repo is absent here.\n")
    OUT.write_text(header + yaml.safe_dump({"scores": scores}, sort_keys=True,
                                           allow_unicode=True, width=100))
    print(f"Wrote {len(scores)} leaderboard score(s) to {OUT.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pull_leaderboard.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Document the sidecar in SCHEMA.md**

Append a section to `SCHEMA.md` after the `candidates.yaml` section:

```markdown
## `leaderboard_scores.yaml` (generated, joined at render)

Written by `scripts/pull_leaderboard.py`. Maps `hf_repo` → an MMLU score from the
HF Open LLM Leaderboard:

```yaml
scores:
  meta-llama/Llama-3.1-405B-Instruct:
    mmlu: 88.6
    source: "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
```

`render_readme.py` reads this file and uses the leaderboard score for the MMLU
column when a repo is present, otherwise it falls back to that row's manual
`benchmark.score` in `models.yaml`. The file is committed so the render stays
offline; a missing/empty/malformed file just means every row uses its manual score.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/pull_leaderboard.py tests/test_pull_leaderboard.py SCHEMA.md
git commit -m "feat: pull_leaderboard.py fetches MMLU into a committed sidecar"
```

---

### Task 4: `render_readme.py` joins the sidecars into MMLU + Arena columns

Load the two sidecars, join by lowercased `hf_repo`, render an MMLU column
(leaderboard value plain, manual fallback marked `*`) and a new Arena column
(rank or `—`), plus a legend and updated prose.

**Files:**
- Modify: `scripts/render_readme.py`
- Test: `tests/test_render_readme.py` (create if absent)

**Interfaces:**
- Consumes: `leaderboard_scores.yaml`, `arena_agent_rankings.yaml` (existing shape:
  `arena_agent` list of rows with `resolved_repo` and `rank`).
- Produces:
  - `render_readme.load_leaderboard(path) -> dict` — `{lower_repo: mmlu_float}`;
    `{}` on missing/malformed.
  - `render_readme.load_arena_ranks(path) -> dict` — `{lower_repo: rank_int}`;
    `{}` on missing/malformed.
  - `render_readme.mmlu_cell(model, lb) -> str` — leaderboard score as `"78.6"`,
    else manual `benchmark.score` as `"78.6*"`, else `"?"`.
  - `render_readme.arena_cell(model, ranks) -> str` — `str(rank)` or `"—"`.
  - `build_table(models, lb, ranks)` — signature gains `lb` and `ranks`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_readme.py`:

```python
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import render_readme as rr


def _model(**kw):
    base = dict(name="M", developer="Org", release_date=date(2025, 1, 1),
                params_total_b=7, params_active_b=7, architecture="dense",
                context_window=4096, modality="text", license="mit",
                commercial_use=True, hf_repo="org/m",
                benchmark={"name": "MMLU", "score": 70.0, "source": "vendor"})
    base.update(kw)
    return base


def test_mmlu_cell_prefers_leaderboard_plain():
    m = _model()
    assert rr.mmlu_cell(m, {"org/m": 78.6}) == "78.6"


def test_mmlu_cell_falls_back_to_manual_marked():
    m = _model()
    assert rr.mmlu_cell(m, {}) == "70.0*"


def test_arena_cell_shows_rank_or_dash():
    m = _model(hf_repo="org/m")
    assert rr.arena_cell(m, {"org/m": 5}) == "5"
    assert rr.arena_cell(m, {}) == "—"


def test_load_leaderboard_tolerates_missing_file(tmp_path):
    assert rr.load_leaderboard(tmp_path / "nope.yaml") == {}


def test_load_leaderboard_parses_scores(tmp_path):
    f = tmp_path / "lb.yaml"
    f.write_text("scores:\n  Org/M:\n    mmlu: 78.6\n")
    assert rr.load_leaderboard(f) == {"org/m": 78.6}


def test_load_arena_ranks_parses_resolved_rows(tmp_path):
    f = tmp_path / "arena.yaml"
    f.write_text("arena_agent:\n- resolved_repo: Org/M\n  rank: 3\n- resolved_repo: null\n  rank: 4\n")
    assert rr.load_arena_ranks(f) == {"org/m": 3}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_render_readme.py -v`
Expected: FAIL — `AttributeError: module 'render_readme' has no attribute 'mmlu_cell'`

- [ ] **Step 3: Write the implementation**

In `scripts/render_readme.py`, add loaders and cell helpers above `build_table`:

```python
LEADERBOARD = ROOT / "leaderboard_scores.yaml"
ARENA = ROOT / "arena_agent_rankings.yaml"


def load_leaderboard(path=LEADERBOARD):
    """{lower_repo: mmlu_float}. {} on missing/malformed."""
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    scores = doc.get("scores") if isinstance(doc, dict) else None
    out = {}
    if isinstance(scores, dict):
        for repo, entry in scores.items():
            if isinstance(entry, dict) and isinstance(entry.get("mmlu"), (int, float)):
                out[str(repo).lower()] = float(entry["mmlu"])
    return out


def load_arena_ranks(path=ARENA):
    """{lower_repo: rank_int}. {} on missing/malformed."""
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    rows = doc.get("arena_agent") if isinstance(doc, dict) else None
    out = {}
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict) and r.get("resolved_repo") and isinstance(r.get("rank"), int):
                out[str(r["resolved_repo"]).lower()] = r["rank"]
    return out


def mmlu_cell(model, lb):
    repo = (model.get("hf_repo") or "").lower()
    if repo in lb:
        return f"{lb[repo]:g}"
    score = (model.get("benchmark") or {}).get("score")
    return f"{score:g}*" if isinstance(score, (int, float)) else "?"


def arena_cell(model, ranks):
    return str(ranks.get((model.get("hf_repo") or "").lower(), "—"))
```

Add `from pathlib import Path` if not already imported (it is — `ROOT` uses it).

Change `build_table` to take and use the joins. Replace the `def build_table(models):`
line and the header/sep/row construction:

```python
def build_table(models, lb, ranks):
    # newest first, then by size
    models = sorted(
        models,
        key=lambda m: (m.get("release_date") or date.min, m.get("params_total_b", 0)),
        reverse=True,
    )
    head = ("| Model | Developer | Released | Params | Context | Modality | "
            "Arena | MMLU | License | Commercial |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for m in models:
        name = m["name"]
        if m.get("weights_url"):
            name = f"[{name}]({m['weights_url']})"
        rows.append(
            f"| {name} | {m['developer']} | {m['release_date']} | "
            f"{human_params(m['params_total_b'], m['params_active_b'], m['architecture'])} | "
            f"{human_ctx(m['context_window'])} | {m['modality']} | "
            f"{arena_cell(m, ranks)} | {mmlu_cell(m, lb)} | "
            f"`{m['license']}` | {commercial_badge(m['commercial_use'])} |"
        )
    return "\n".join(rows)
```

In `main()`, load the joins and pass them, and add the legend to the prose.
Replace `table = build_table(doc["models"])` with:

```python
    lb = load_leaderboard()
    ranks = load_arena_ranks()
    table = build_table(doc["models"], lb, ranks)
```

In the `body` string, replace the benchmark-caveat blockquote with one that also
explains the columns and the marker (this is the README prose — it must be edited
here, never in README.md):

```python
        "> **Columns:** **MMLU** is from the HF Open LLM Leaderboard where "
        "available; values marked `*` fall back to a vendor/manual figure and are "
        "not harness-comparable. **Arena** is the Agent Arena rank (`—` = not "
        "currently ranked). See each row's `benchmark.source` in the YAML.\n\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_render_readme.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/render_readme.py tests/test_render_readme.py
git commit -m "feat(render): join leaderboard + arena into MMLU and Arena columns"
```

---

### Task 5: Seed the sidecar, regenerate the README, full verification

Create the committed default sidecar, regenerate the README so the committed copy
matches render output (the CI gate), and run the whole suite + validator.

**Files:**
- Create: `leaderboard_scores.yaml`
- Modify: `README.md` (regenerated)

- [ ] **Step 1: Seed a deterministic empty sidecar**

Run: `.venv/bin/python scripts/pull_leaderboard.py --no-fetch`
Expected: `Wrote 0 leaderboard score(s) to leaderboard_scores.yaml`

This commits a deterministic `scores: {}` so dev/CI is offline and every MMLU cell
uses the manual fallback until a maintainer runs `pull_leaderboard.py` live (a
later data-refresh PR, like discovery).

- [ ] **Step 2: Regenerate the README**

Run: `.venv/bin/python scripts/render_readme.py`
Expected: `Rendered README.md with 16 models.`

- [ ] **Step 3: Confirm the table has the new columns**

Run: `grep -n "Arena | MMLU" README.md`
Expected: one match (the header row). Manually eyeball that rows now show `—` in
Arena and `NN.N*` in MMLU.

- [ ] **Step 4: Run the full suite and the validator**

Run: `.venv/bin/python -m pytest tests/ -v && .venv/bin/python scripts/validate.py`
Expected: all tests PASS, then `OK — 16 models, no problems found.`

- [ ] **Step 5: Confirm render is idempotent (the CI gate)**

Run: `.venv/bin/python scripts/render_readme.py && git diff --exit-code README.md`
Expected: exit 0, no diff (proves a second render matches the committed file).

- [ ] **Step 6: Commit**

```bash
git add leaderboard_scores.yaml README.md
git commit -m "data: seed leaderboard sidecar and regenerate README with new columns"
```

---

## Self-Review

**Spec coverage:**
- Auto MMLU column from HF Open LLM Leaderboard → Task 3 (fetch) + Task 4 (render join). ✓
- MMLU fallback to manual score → `mmlu_cell` (Task 4), `benchmark.score` kept required. ✓
- Auto Arena column from `arena_agent_rankings.yaml` → `load_arena_ranks` + `arena_cell` (Task 4). ✓
- context_window fix via `config.json` → Task 1 (helper) + Task 2 (wired into discovery). ✓
- Sidecar + offline render invariant → Tasks 3–5; render reads only committed files, seeded empty sidecar (Task 5). ✓
- Graceful degradation → `load_leaderboard`/`load_arena_ranks`/`fetch_context_window` all return empty/None on failure, tested. ✓
- Shared context path in `hf_meta.py` (used by discovery; available to `pull_hf.py`) → Task 1/2. ✓
- SCHEMA + README prose docs → Task 3 Step 5, Task 4 Step 3. ✓

**Placeholder scan:** No TBD/TODO-in-plan; every code step shows full code. The only
in-code `# TODO` strings are the pre-existing candidate-field markers, unchanged.

**Type consistency:** `get_json` signature (`url -> dict`) consistent across
`_http_get_json`, `fetch_context_window`, `resolve_context`, `fetch_rows`.
`context_window` param consistent between `candidate_from_repo` and the discover
call sites. `lb` is `{lower_repo: float}` and `ranks` is `{lower_repo: int}`
consistently between the loaders, the cell helpers, and `build_table`.

**Known live-fetch risk (carried from the spec, not a plan defect):** `fetch_rows`
and the leaderboard row field names (`ROWS_URL`, `_REPO_KEYS`, MMLU key) target the
datasets-server API and must be verified on the first real networked run. All
parse/join/render logic is fixture-tested and independent of that verification;
the seeded empty sidecar keeps everything green until then.
