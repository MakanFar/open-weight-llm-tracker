# Artificial Analysis Intelligence Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the non-functional HF-sourced MMLU column with the Artificial Analysis Intelligence Index, scraped into a committed sidecar and joined to `models.yaml` by name.

**Architecture:** A new `scripts/pull_aa.py` scrapes AA's single server-rendered table, picks the highest-scoring reasoning-effort variant per model, matches display names to `models.yaml` locally (no HF calls), and writes `aa_scores.yaml` keyed by `hf_repo`. `render_readme.py` joins that sidecar into an `AA Index` column. The `benchmark:` block, `pull_leaderboard.py` and `leaderboard_scores.yaml` are deleted outright.

**Tech Stack:** Python 3.9+, `requests`, `beautifulsoup4`, `PyYAML`, `pytest`. All already in `requirements.txt`.

## Global Constraints

- Run everything with the repo venv: `.venv/bin/python`. `pytest` is not on the base interpreter.
- TDD is mandatory: write the failing test, watch it fail, implement minimally, watch it pass, commit.
- No test may touch the network. AA parsing is tested against a committed fixture.
- Never overwrite committed data on failure. A failed fetch or a zero-row parse must leave the existing sidecar byte-identical.
- Open-weight status is never read from AA. It remains decided solely by HF repo resolution.
- `render_readme.py` rewrites the whole README; prose lives in the `body` string in `main()`.
- The full suite must pass at the end of every task: `.venv/bin/python -m pytest tests/ -q`.
- Baseline at plan time: 164 tests passing, working tree clean.

---

### Task 1: Extract shared name helpers into `scripts/names.py`

`pull_arena.py` owns `slug()` and `_VARIANT_SUFFIXES`; `pull_aa.py` needs both. Extract them so there is one implementation, and widen suffix stripping to handle space separators (`"Llama 3.3 70B Instruct"`) as well as hyphen/underscore (`"Llama-3.3-70B-Instruct"`). That widening is what lets AA's `Llama 3.3 70B` match your `Llama 3.3 70B Instruct` in Task 4.

**Files:**
- Create: `scripts/names.py`
- Modify: `scripts/pull_arena.py` (remove local `slug`, `_VARIANT_SUFFIXES`; import from `names`)
- Test: `tests/test_names.py`

**Interfaces:**
- Consumes: nothing
- Produces: `names.slug(text) -> str`, `names.VARIANT_SUFFIXES: tuple[str, ...]`, `names.strip_variant_suffix(text) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_names.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import names


def test_slug_lowercases_and_drops_punctuation():
    assert names.slug("GLM-5.2 (Max)") == "glm52max"
    assert names.slug("Llama 3.3 70B") == "llama3370b"
    assert names.slug("") == ""


def test_strip_variant_suffix_handles_hyphen_and_space():
    """Repos write 'Llama-3.3-70B-Instruct'; AA writes 'Llama 3.3 70B'."""
    assert names.strip_variant_suffix("Llama-3.3-70B-Instruct") == "Llama-3.3-70B"
    assert names.strip_variant_suffix("Llama 3.3 70B Instruct") == "Llama 3.3 70B"
    assert names.strip_variant_suffix("gemma-3-27b-it") == "gemma-3-27b"
    assert names.strip_variant_suffix("Qwen2.5-7B-Chat") == "Qwen2.5-7B"


def test_strip_variant_suffix_leaves_other_names_alone():
    assert names.strip_variant_suffix("GLM-5.2") == "GLM-5.2"
    assert names.strip_variant_suffix("Kimi-K3") == "Kimi-K3"


def test_strip_variant_suffix_does_not_eat_a_word_ending_in_it():
    """'Summit' ends in 'it' but is not a variant suffix."""
    assert names.strip_variant_suffix("Model-Summit") == "Model-Summit"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_names.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'names'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/names.py`:

```python
#!/usr/bin/env python3
"""Shared model-name normalisation.

Both leaderboard scrapers (pull_arena.py, pull_aa.py) compare vendor display
names against Hugging Face repo slugs, so the comparison rules live here rather
than being duplicated or imported from one scraper into the other.
"""
import re

# Suffixes that mark an instruction-tuned variant of the same weights. A repo
# differing from a leaderboard name only by one of these is the same model.
VARIANT_SUFFIXES = ("instruct", "it", "chat", "base")

# Separator before a suffix: repos use - or _, leaderboards use a space.
_SUFFIX_RE = re.compile(
    r"[-_\s](?:" + "|".join(VARIANT_SUFFIXES) + r")$", re.IGNORECASE)


def slug(text):
    """Lowercase and strip every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def strip_variant_suffix(text):
    """Drop one trailing variant suffix. 'Llama 3.3 70B Instruct' -> 'Llama 3.3 70B'.

    The separator is required, so a word that merely ends in the letters of a
    suffix ("Summit") is untouched.
    """
    return _SUFFIX_RE.sub("", text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_names.py -q`
Expected: PASS, 4 tests

- [ ] **Step 5: Point `pull_arena.py` at the shared module**

**Do not touch `score_match`, `strip_repo_decorations` or `_strip_decorations`.** Those are token-based and deliberately strip size tokens (`70B`, `A55B`) as well as variant suffixes, which is correct for pairwise ratio matching but wrong for dictionary keying — see the note in Task 3. This step is a pure move of two names.

In `scripts/pull_arena.py`:

1. Add `import names` beside the existing `sys.path.insert(...)` line.
2. Delete the local `slug` function (lines 202-204).
3. Delete the `_VARIANT_SUFFIXES` tuple (line 106) and its two comment lines above it.
4. Add a module-level alias so the remaining internals keep working unchanged:

```python
slug = names.slug
_VARIANT_SUFFIXES = names.VARIANT_SUFFIXES
```

Place those two lines immediately after the `import names` line. Every existing call site (`score_match`, `carry_forward_resolutions`, `_strip_decorations`) then resolves through the shared module with no further edits.

- [ ] **Step 6: Run the full suite to prove nothing regressed**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, 168 tests (164 baseline + 4 new)

If any arena test fails, the cause is the widened separator now stripping a suffix it previously kept. Fix `names.py`, not the arena test.

- [ ] **Step 7: Commit**

```bash
git add scripts/names.py scripts/pull_arena.py tests/test_names.py
git commit -m "refactor: share model-name normalisation between scrapers"
```

---

### Task 2: Parse the AA leaderboard table

**Files:**
- Create: `scripts/pull_aa.py`
- Create: `tests/fixtures/aa_leaderboard.html`
- Test: `tests/test_pull_aa.py`

**Interfaces:**
- Consumes: nothing
- Produces: `pull_aa.parse_leaderboard(html) -> list[dict]` with keys `model`, `creator`, `intelligence_index`

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/aa_leaderboard.html`. This mirrors the live markup: one `<table>`, a grouped header row, a column header row, then data rows. Two rows are the same model at different reasoning efforts (used in Task 3), and one row has a non-numeric index (must be dropped).

```html
<!doctype html>
<html><body>
<table>
  <thead>
    <tr><th></th><th>Features</th><th>Intelligence</th><th>Price</th><th>Speed</th></tr>
    <tr>
      <th>Model</th><th>Context Window</th><th>Creator</th>
      <th>Artificial Analysis Intelligence Index</th><th>Cost per Task USD</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Claude Opus 5 (max)</td><td>1M</td><td>Anthropic</td><td>61</td><td>$2.34</td></tr>
    <tr><td>Kimi K3 (max)</td><td>256k</td><td>Moonshot AI</td><td>57</td><td>$2.34</td></tr>
    <tr><td>Kimi K3 (low)</td><td>256k</td><td>Moonshot AI</td><td>47</td><td>$0.90</td></tr>
    <tr><td>GLM-5.2</td><td>200k</td><td>Z.ai</td><td>34</td><td>$0.41</td></tr>
    <tr><td>Llama 3.3 70B</td><td>131k</td><td>Meta</td><td>9</td><td>$0.12</td></tr>
    <tr><td>Unrated Model</td><td>8k</td><td>Nobody</td><td>&mdash;</td><td>$0.00</td></tr>
  </tbody>
</table>
</body></html>
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_pull_aa.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pull_aa

FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "aa_leaderboard.html").read_text()


def test_parse_reads_every_scored_row():
    rows = pull_aa.parse_leaderboard(FIXTURE)
    assert [r["model"] for r in rows] == [
        "Claude Opus 5 (max)", "Kimi K3 (max)", "Kimi K3 (low)",
        "GLM-5.2", "Llama 3.3 70B",
    ]


def test_parse_reads_creator_and_index():
    rows = pull_aa.parse_leaderboard(FIXTURE)
    kimi = next(r for r in rows if r["model"] == "Kimi K3 (max)")
    assert kimi["creator"] == "Moonshot AI"
    assert kimi["intelligence_index"] == 57


def test_parse_drops_rows_with_no_numeric_index():
    """Header rows and unrated models both fail the integer check."""
    rows = pull_aa.parse_leaderboard(FIXTURE)
    assert all(r["model"] != "Unrated Model" for r in rows)
    assert all(r["model"] != "Model" for r in rows)


def test_parse_returns_empty_for_markup_with_no_table():
    assert pull_aa.parse_leaderboard("<html><body><p>nope</p></body></html>") == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pull_aa.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pull_aa'`

- [ ] **Step 4: Write minimal implementation**

Create `scripts/pull_aa.py`:

```python
#!/usr/bin/env python3
"""
Scrape the Artificial Analysis Intelligence Index into a committed sidecar,
aa_scores.yaml, keyed by hf_repo.

WHY THIS REPLACED THE HF LEADERBOARD:
    scripts/pull_leaderboard.py could not fill the benchmark column. HF Open
    LLM Leaderboard v2 publishes no plain MMLU, it is archived so no 2026 model
    appears in it, and the HF model card API returns no structured eval data at
    all — 0 of 42 tracked/candidate repos carry a model-index.

WHAT THE INDEX IS:
    A 0-100 composite: Agents 34%, Coding 24%, Scientific Reasoning 24%,
    General 18%. It is versioned and re-weighted periodically, so scores are
    NOT comparable across time. Only the current snapshot is stored.

OPENNESS IS NOT READ FROM AA:
    AA's page carries no openness column, and we would not trust it if it did.
    Proprietary rows simply fail to match models.yaml and fall out. Open-weight
    status stays decided by HF repo resolution alone.
"""
import re
import sys
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
OUT = ROOT / "aa_scores.yaml"

LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"

# Cell positions in the data rows, matching the column header:
#   Model | Context Window | Creator | AA Intelligence Index | Cost | ...
_MODEL, _CREATOR, _INDEX = 0, 2, 3


def parse_leaderboard(html):
    """Rows of {model, creator, intelligence_index} from the leaderboard table.

    Header rows are not skipped positionally — they are discarded by the same
    integer check that discards unrated models, so a change in how many header
    rows AA emits cannot silently shift the data.
    """
    table = BeautifulSoup(html, "html.parser").find("table")
    if table is None:
        return []
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) <= _INDEX:
            continue
        raw = cells[_INDEX].strip()
        if not re.fullmatch(r"\d+", raw):
            continue
        rows.append({
            "model": cells[_MODEL],
            "creator": cells[_CREATOR],
            "intelligence_index": int(raw),
        })
    return rows
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pull_aa.py -q`
Expected: PASS, 4 tests

- [ ] **Step 6: Commit**

```bash
git add scripts/pull_aa.py tests/fixtures/aa_leaderboard.html tests/test_pull_aa.py
git commit -m "feat(aa): parse the Artificial Analysis leaderboard table"
```

---

### Task 3: Pick the highest reasoning-effort variant per model

AA lists the same weights several times at different efforts, with a wide spread (Kimi K3: 57 at `max`, 47 at `low`). Take the highest and record which row won, so the number is auditable. "Base row only" is not an option — AA publishes no bare `Kimi K3` row.

**Files:**
- Modify: `scripts/pull_aa.py`
- Test: `tests/test_pull_aa.py`

**Interfaces:**
- Consumes: `parse_leaderboard` rows from Task 2
- Produces: `pull_aa.best_by_slug(rows) -> dict[str, dict]` keyed by `names.slug` of the variant-stripped model name, each value `{model_slug, aa_model, variant, intelligence_index}`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pull_aa.py`:

```python
def test_best_by_slug_keeps_the_highest_variant():
    rows = pull_aa.parse_leaderboard(FIXTURE)
    best = pull_aa.best_by_slug(rows)
    kimi = best["kimik3"]
    assert kimi["intelligence_index"] == 57
    assert kimi["variant"] == "max"
    assert kimi["aa_model"] == "Kimi K3 (max)"


def test_best_by_slug_labels_a_row_with_no_parenthetical():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    assert best["glm52"]["variant"] == "default"
    assert best["glm52"]["intelligence_index"] == 34


def test_best_by_slug_collapses_variants_to_one_entry_per_model():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    assert sorted(best) == ["claudeopus5", "glm52", "kimik3", "llama3370b"]


def test_best_by_slug_is_order_independent():
    """The winner must not depend on which variant the page lists first."""
    rows = [
        {"model": "M (low)", "creator": "C", "intelligence_index": 10},
        {"model": "M (max)", "creator": "C", "intelligence_index": 20},
    ]
    assert pull_aa.best_by_slug(rows)["m"]["intelligence_index"] == 20
    assert pull_aa.best_by_slug(rows[::-1])["m"]["intelligence_index"] == 20


def test_best_by_slug_keeps_different_sizes_apart():
    """Size is identity: 72B and 7B are different models, not variants.

    pull_arena's _strip_decorations drops size tokens, which is right for its
    pairwise ratio matching and fatally wrong here — both would key to 'qwen25'
    and one would silently overwrite the other.
    """
    rows = [
        {"model": "Qwen2.5 72B", "creator": "Alibaba", "intelligence_index": 30},
        {"model": "Qwen2.5 7B", "creator": "Alibaba", "intelligence_index": 12},
    ]
    best = pull_aa.best_by_slug(rows)
    assert sorted(best) == ["qwen2572b", "qwen257b"]
    assert best["qwen2572b"]["intelligence_index"] == 30
    assert best["qwen257b"]["intelligence_index"] == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pull_aa.py -q`
Expected: FAIL with `AttributeError: module 'pull_aa' has no attribute 'best_by_slug'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/pull_aa.py`:

```python
_PAREN_RE = re.compile(r"\(([^)]*)\)")


def split_variant(display):
    """('Kimi K3 (max)') -> ('Kimi K3', 'max'). No parenthetical -> 'default'."""
    found = _PAREN_RE.search(display)
    variant = found.group(1).strip().lower() if found else "default"
    base = _PAREN_RE.sub(" ", display)
    base = re.sub(r"\s+", " ", base).strip()
    return base, variant


def best_by_slug(rows):
    """Highest-scoring variant per model, keyed by slug.

    AA lists the same weights at several reasoning efforts and the spread is
    wide (Kimi K3 scores 57 at max, 47 at low), so the winner is chosen
    explicitly rather than by whichever row happens to come last. The variant
    that won is recorded so the number can be traced back to a row.
    """
    best = {}
    for row in rows:
        base, variant = split_variant(row["model"])
        key = names.slug(names.strip_variant_suffix(base))
        if not key:
            continue
        current = best.get(key)
        if current is None or row["intelligence_index"] > current["intelligence_index"]:
            best[key] = {
                "model_slug": key,
                "aa_model": row["model"],
                "variant": variant,
                "intelligence_index": row["intelligence_index"],
            }
    return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pull_aa.py -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/pull_aa.py tests/test_pull_aa.py
git commit -m "feat(aa): pick the highest reasoning-effort variant per model"
```

---

### Task 4: Match AA models to `models.yaml`

**Files:**
- Modify: `scripts/pull_aa.py`
- Test: `tests/test_pull_aa.py`

**Interfaces:**
- Consumes: `best_by_slug` output from Task 3
- Produces: `pull_aa.tracked_models(path) -> list[dict]`, `pull_aa.match_to_tracked(best, tracked) -> (scores, unmatched)` where `scores` is `{hf_repo: {intelligence_index, variant, aa_model, source}}` and `unmatched` is a sorted list of AA display names

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pull_aa.py`:

```python
import yaml

TRACKED = [
    {"name": "Kimi K3", "hf_repo": "moonshotai/Kimi-K3"},
    {"name": "Llama 3.3 70B Instruct", "hf_repo": "meta-llama/Llama-3.3-70B-Instruct"},
    {"name": "Some Model", "hf_repo": "zai-org/GLM-5.2"},
]


def test_match_joins_on_the_model_name():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    scores, _ = pull_aa.match_to_tracked(best, TRACKED)
    assert scores["moonshotai/Kimi-K3"]["intelligence_index"] == 57
    assert scores["moonshotai/Kimi-K3"]["variant"] == "max"


def test_match_tolerates_an_instruct_suffix_on_our_side():
    """AA says 'Llama 3.3 70B'; models.yaml says 'Llama 3.3 70B Instruct'."""
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    scores, _ = pull_aa.match_to_tracked(best, TRACKED)
    assert scores["meta-llama/Llama-3.3-70B-Instruct"]["intelligence_index"] == 9


def test_match_falls_back_to_the_repo_tail():
    """'Some Model' does not match, but the repo tail GLM-5.2 does."""
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    scores, _ = pull_aa.match_to_tracked(best, TRACKED)
    assert scores["zai-org/GLM-5.2"]["intelligence_index"] == 34


def test_match_reports_aa_rows_that_hit_nothing():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    _, unmatched = pull_aa.match_to_tracked(best, TRACKED)
    assert unmatched == ["Claude Opus 5 (max)"]


def test_match_records_the_source_url():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    scores, _ = pull_aa.match_to_tracked(best, TRACKED)
    assert scores["moonshotai/Kimi-K3"]["source"] == pull_aa.LEADERBOARD_URL


def test_tracked_models_reads_models_yaml(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text(yaml.safe_dump({"models": [
        {"name": "M", "hf_repo": "org/m"},
        {"name": "No repo"},
    ]}))
    assert pull_aa.tracked_models(f) == [{"name": "M", "hf_repo": "org/m"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pull_aa.py -q`
Expected: FAIL with `AttributeError: module 'pull_aa' has no attribute 'match_to_tracked'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/pull_aa.py`:

```python
def tracked_models(path=DATA):
    """models.yaml rows that carry an hf_repo."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    rows = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("hf_repo")]


def _keys_for(model):
    """Every slug a tracked model may be known by on a leaderboard."""
    candidates = {model["name"], model["hf_repo"].split("/")[-1]}
    return {names.slug(names.strip_variant_suffix(c)) for c in candidates if c}


def match_to_tracked(best, tracked):
    """Join AA entries onto tracked models. Returns (scores, unmatched).

    Matching is local: AA publishes display names, not repo ids, and doing the
    lookup here avoids depending on HF search, whose rate limits have already
    caused silent data loss elsewhere in this repo.

    unmatched is expected to be long — most AA rows are proprietary models this
    tracker will never carry — so it is informational, never an error.
    """
    scores, used = {}, set()
    for model in tracked:
        for key in _keys_for(model):
            entry = best.get(key)
            if entry is None:
                continue
            scores[model["hf_repo"]] = {
                "intelligence_index": entry["intelligence_index"],
                "variant": entry["variant"],
                "aa_model": entry["aa_model"],
                "source": LEADERBOARD_URL,
            }
            used.add(key)
            break
    unmatched = sorted(e["aa_model"] for k, e in best.items() if k not in used)
    return scores, unmatched
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pull_aa.py -q`
Expected: PASS, 14 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/pull_aa.py tests/test_pull_aa.py
git commit -m "feat(aa): match leaderboard rows to tracked models by name"
```

---

### Task 5: Write the sidecar, preserving it on failure

The failure rule is the whole point of this task. `pull_leaderboard.py` returned `[]` on a 429, which was indistinguishable from "no results", and it overwrote eight freshly fetched scores with an empty file. A zero-row parse counts as failure too — it means AA's markup changed.

**Files:**
- Modify: `scripts/pull_aa.py`
- Test: `tests/test_pull_aa.py`

**Interfaces:**
- Consumes: `match_to_tracked` from Task 4
- Produces: `pull_aa.fetch_html(url, get=requests.get) -> str | None`, `pull_aa.write_scores(path, scores, unmatched)`, `pull_aa.refresh(path, html, tracked) -> int | None`, `pull_aa.main()`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pull_aa.py`:

```python
def test_refresh_writes_scores_and_unmatched(tmp_path):
    out = tmp_path / "aa_scores.yaml"
    n = pull_aa.refresh(out, FIXTURE, TRACKED)
    assert n == 3
    doc = yaml.safe_load(out.read_text())
    assert doc["scores"]["moonshotai/Kimi-K3"]["intelligence_index"] == 57
    assert doc["unmatched"] == ["Claude Opus 5 (max)"]


def test_refresh_leaves_the_sidecar_untouched_when_the_fetch_failed(tmp_path):
    out = tmp_path / "aa_scores.yaml"
    out.write_text("scores:\n  org/m:\n    intelligence_index: 42\n")
    before = out.read_text()

    assert pull_aa.refresh(out, None, TRACKED) is None
    assert out.read_text() == before


def test_refresh_treats_a_zero_row_parse_as_failure(tmp_path):
    """Empty parse means AA's markup changed — do not erase good data."""
    out = tmp_path / "aa_scores.yaml"
    out.write_text("scores:\n  org/m:\n    intelligence_index: 42\n")
    before = out.read_text()

    assert pull_aa.refresh(out, "<html><body>redesigned</body></html>", TRACKED) is None
    assert out.read_text() == before


def test_refresh_writes_an_empty_score_set_when_rows_parsed_but_matched_nothing(tmp_path):
    """Parsed fine, matched nobody: that is a real answer, not a failure."""
    out = tmp_path / "aa_scores.yaml"
    assert pull_aa.refresh(out, FIXTURE, []) == 0
    assert yaml.safe_load(out.read_text())["scores"] == {}


def test_fetch_html_returns_none_on_error():
    def boom(url, **kw):
        raise RuntimeError("HTTP Error 429: Too Many Requests")
    assert pull_aa.fetch_html(pull_aa.LEADERBOARD_URL, get=boom) is None


def test_fetch_html_returns_none_on_bad_status():
    class Resp:
        status_code = 503
        text = "nope"
    assert pull_aa.fetch_html(pull_aa.LEADERBOARD_URL, get=lambda u, **kw: Resp()) is None


def test_fetch_html_returns_the_body_on_success():
    class Resp:
        status_code = 200
        text = "<html>ok</html>"
    assert pull_aa.fetch_html(pull_aa.LEADERBOARD_URL, get=lambda u, **kw: Resp()) == "<html>ok</html>"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pull_aa.py -q`
Expected: FAIL with `AttributeError: module 'pull_aa' has no attribute 'refresh'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/pull_aa.py`:

```python
HEADER = (
    "# AUTO-SCRAPED by scripts/pull_aa.py — do not hand-edit.\n"
    "# Artificial Analysis Intelligence Index (0-100 composite), by hf_repo.\n"
    "# variant records which reasoning-effort row won (the highest scoring).\n"
    "# The index is re-weighted between versions, so values are NOT comparable\n"
    "# across time. unmatched lists AA rows no tracked model claimed.\n"
)


def fetch_html(url, get=requests.get):
    """Leaderboard HTML, or None on any failure."""
    try:
        resp = get(url, timeout=30,
                   headers={"User-Agent": "Mozilla/5.0 (owlt-aa/1.0)"})
    except Exception as exc:
        print(f"  ! AA fetch failed: {exc}")
        return None
    if getattr(resp, "status_code", None) != 200:
        print(f"  ! AA returned HTTP {getattr(resp, 'status_code', '?')}")
        return None
    return resp.text


def write_scores(path, scores, unmatched):
    Path(path).write_text(HEADER + yaml.safe_dump(
        {"scores": scores, "unmatched": unmatched},
        sort_keys=True, allow_unicode=True, width=100))


def refresh(path, html, tracked):
    """Rewrite the sidecar. Returns the score count, or None if nothing was written.

    None means the run failed and the committed file was left alone. A zero-row
    parse is a failure: AA is never legitimately empty, so an empty parse means
    the markup changed and writing would erase good data.
    """
    if html is None:
        print(f"  ! leaving {Path(path).name} unchanged")
        return None
    rows = parse_leaderboard(html)
    if not rows:
        print(f"  ! parsed 0 rows — AA markup may have changed; "
              f"leaving {Path(path).name} unchanged")
        return None
    scores, unmatched = match_to_tracked(best_by_slug(rows), tracked)
    write_scores(path, scores, unmatched)
    return len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", help="parse a saved page instead of fetching")
    args = ap.parse_args()

    html = Path(args.html).read_text() if args.html else fetch_html(LEADERBOARD_URL)
    n = refresh(OUT, html, tracked_models())
    if n is None:
        return
    print(f"Wrote {n} AA score(s) to {OUT.name}")


if __name__ == "__main__":
    main()
```

Add `import argparse` to the imports at the top of the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pull_aa.py -q`
Expected: PASS, 21 tests

- [ ] **Step 5: Run it for real against the live site**

Run: `.venv/bin/python scripts/pull_aa.py`
Expected: `Wrote 3 AA score(s) to aa_scores.yaml`

Three is correct at time of writing — AA covers only Llama 4 Scout, Llama 4 Maverick and Llama 3.3 70B of the 16 tracked models. If it prints a fetch failure, re-run; if it prints `parsed 0 rows`, AA's markup changed and Task 2's selectors need revisiting.

- [ ] **Step 6: Commit**

```bash
git add scripts/pull_aa.py tests/test_pull_aa.py aa_scores.yaml
git commit -m "feat(aa): write the score sidecar, preserving it on any failure"
```

---

### Task 6: Render the AA Index column

**Files:**
- Modify: `scripts/render_readme.py`
- Test: `tests/test_render_readme.py`

**Interfaces:**
- Consumes: `aa_scores.yaml` from Task 5
- Produces: `render_readme.load_aa_scores(path) -> dict`, `render_readme.aa_cell(model, aa) -> str`

- [ ] **Step 1: Write the failing test**

In `tests/test_render_readme.py`, delete `test_mmlu_cell_prefers_leaderboard_plain`, `test_mmlu_cell_falls_back_to_manual_marked`, `test_mmlu_cell_ignores_an_mmlu_pro_score`, `test_mmlu_pro_cell_shows_only_mmlu_pro_scores`, `test_mmlu_pro_cell_never_falls_back_to_the_manual_figure`, `test_table_has_distinct_mmlu_and_mmlu_pro_columns`, `test_load_leaderboard_tolerates_missing_file`, `test_load_leaderboard_parses_metric_and_score`, `test_load_leaderboard_skips_entries_missing_a_metric`, and the `_lb` helper. Then append:

```python
def test_aa_cell_shows_the_index():
    m = _model(hf_repo="moonshotai/Kimi-K3")
    aa = {"moonshotai/kimi-k3": {"index": 57, "variant": "max"}}
    assert rr.aa_cell(m, aa) == "57"


def test_aa_cell_dashes_when_aa_does_not_rate_the_model():
    """There is no fallback: models.yaml no longer carries a score."""
    assert rr.aa_cell(_model(hf_repo="org/m"), {}) == "—"


def test_load_aa_scores_parses_the_sidecar(tmp_path):
    f = tmp_path / "aa.yaml"
    f.write_text("scores:\n  Moonshot/Kimi-K3:\n    intelligence_index: 57\n"
                 "    variant: max\n")
    assert rr.load_aa_scores(f) == {
        "moonshot/kimi-k3": {"index": 57, "variant": "max"}}


def test_load_aa_scores_tolerates_a_missing_file(tmp_path):
    assert rr.load_aa_scores(tmp_path / "nope.yaml") == {}


def test_load_aa_scores_skips_entries_with_no_numeric_index(tmp_path):
    f = tmp_path / "aa.yaml"
    f.write_text("scores:\n  org/m:\n    variant: max\n")
    assert rr.load_aa_scores(f) == {}


def test_table_has_an_aa_index_column_and_no_mmlu():
    table = rr.build_table([_model(hf_repo="org/m")],
                           {"org/m": {"index": 42, "variant": "max"}}, {})
    head = table.splitlines()[0]
    assert "| AA Index |" in head
    assert "MMLU" not in head
    assert "42" in table.splitlines()[2]
```

Also update `_model()` at the top of the file to drop the now-removed `benchmark` key:

```python
def _model(**kw):
    base = dict(name="M", developer="Org", release_date=date(2025, 1, 1),
                params_total_b=7, params_active_b=7, architecture="dense",
                context_window=4096, modality="text", license="mit",
                commercial_use=True, hf_repo="org/m")
    base.update(kw)
    return base
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_render_readme.py -q`
Expected: FAIL with `AttributeError: module 'render_readme' has no attribute 'aa_cell'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/render_readme.py`: delete `load_leaderboard`, `_leaderboard_score`, `mmlu_cell`, `mmlu_pro_cell`, and the `LEADERBOARD` constant. Add:

```python
AA = ROOT / "aa_scores.yaml"


def load_aa_scores(path=AA):
    """{lower_repo: {"index": int, "variant": str}}. {} on missing/malformed."""
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    scores = doc.get("scores") if isinstance(doc, dict) else None
    out = {}
    if isinstance(scores, dict):
        for repo, entry in scores.items():
            if not isinstance(entry, dict):
                continue
            idx = entry.get("intelligence_index")
            if isinstance(idx, int) and not isinstance(idx, bool):
                out[str(repo).lower()] = {
                    "index": idx, "variant": entry.get("variant")}
    return out


def aa_cell(model, aa):
    """Artificial Analysis Intelligence Index, or — when AA does not rate it.

    There is deliberately no fallback to a manual figure: models.yaml no longer
    carries one, because MMLU (~86) and the AA index (~10-57) are different
    scales and sharing a column invited a comparison that does not exist.
    """
    entry = aa.get((model.get("hf_repo") or "").lower())
    return str(entry["index"]) if entry else "—"
```

Change `build_table` to take `aa` instead of `lb`:

```python
def build_table(models, aa, ranks):
```

and its header, separator and row lines:

```python
    head = ("| Model | Developer | Released | Params | Context | Modality | "
            "Arena | AA Index | License | Commercial |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
```

```python
            f"{arena_cell(m, ranks)} | {aa_cell(m, aa)} | "
```

In `main()`, replace the leaderboard load and the call:

```python
    aa = load_aa_scores()
    ranks = load_arena_ranks()
    table = build_table(doc["models"], aa, ranks)
```

Replace the Columns prose note in the `body` string with:

```python
        "> **Columns:** **AA Index** is the [Artificial Analysis Intelligence "
        "Index](https://artificialanalysis.ai/leaderboards/models) — a 0–100 "
        "composite of agentic, coding, scientific-reasoning and general "
        "evaluations. `—` means Artificial Analysis does not currently rate that "
        "model; it drops older models, so coverage skews to recent releases. The "
        "index is re-weighted between versions, so values are not comparable "
        "across time. **Arena** is the Agent Arena rank (`—` = not currently "
        "ranked).\n\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_render_readme.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/render_readme.py tests/test_render_readme.py
git commit -m "feat(render): replace the MMLU columns with AA Index"
```

---

### Task 7: Remove the `benchmark` field and the HF leaderboard scraper

**Files:**
- Delete: `scripts/pull_leaderboard.py`, `leaderboard_scores.yaml`, `tests/test_pull_leaderboard.py`
- Modify: `models.yaml` (all 16 rows), `scripts/validate.py:19-22,81-85`, `SCHEMA.md`, `CLAUDE.md`, `README.md` (regenerated)
- Test: `tests/test_validate.py` if present, otherwise the validate run in Step 4

**Interfaces:**
- Consumes: everything above
- Produces: a `models.yaml` with no `benchmark` block that passes `validate.py`

- [ ] **Step 1: Delete the HF leaderboard scraper**

```bash
git rm scripts/pull_leaderboard.py leaderboard_scores.yaml tests/test_pull_leaderboard.py
```

- [ ] **Step 2: Strip the `benchmark` block from every models.yaml row**

`models.yaml` is the hand-curated source of truth. Do **not** round-trip it through `yaml.safe_dump` — that would reflow every row, drop the comment header, and bury the real change in a whole-file diff. Remove the block line-wise instead, so the diff shows exactly 64 deleted lines and nothing else.

Each row's block is exactly four lines at a fixed indent:

```yaml
    benchmark:
      name: MMLU
      score: 88.6
      source: vendor
```

Run this once:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
p = Path("models.yaml")
lines = p.read_text().splitlines(keepends=True)
out, i, removed = [], 0, 0
while i < len(lines):
    if lines[i].rstrip("\n") == "    benchmark:":
        i += 1
        while i < len(lines) and lines[i].startswith("      "):
            i += 1
        removed += 1
        continue
    out.append(lines[i])
    i += 1
p.write_text("".join(out))
print(f"removed benchmark from {removed} rows")
PY
```

Expected: `removed benchmark from 16 rows`

- [ ] **Step 2b: Replace the benchmark note in the models.yaml header**

Lines 5-9 of `models.yaml` are a comment block entirely about `benchmark.score`. Replace those five lines with:

```yaml
# NOTE ON BENCHMARKS: this file stores no benchmark score. The anchor number is
# the Artificial Analysis Intelligence Index, fetched by scripts/pull_aa.py into
# aa_scores.yaml and joined on hf_repo at render time. A hand-copied score with
# no provenance is worse than no score.
```

Then confirm nothing references the removed field:

```bash
grep -n "benchmark" models.yaml
```

Expected: only the new comment lines, no `benchmark:` key.

- [ ] **Step 3: Drop benchmark from validate.py**

In `scripts/validate.py`, remove `"benchmark",` from the `REQUIRED` list, and delete the whole benchmark block:

```python
        # benchmark block
        b = m.get("benchmark") or {}
        for bf in ("name", "score", "source"):
            if b.get(bf) in (None, ""):
                errors.append(f"[{tag}] benchmark.{bf} is required")
```

- [ ] **Step 4: Run validate and the full suite**

Run: `.venv/bin/python scripts/validate.py`
Expected: `OK — 16 models, no problems found.`

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS. The count drops by the 15 deleted `test_pull_leaderboard.py` tests and the 9 deleted MMLU render tests, and rises by the 25 added across Tasks 1–6.

- [ ] **Step 5: Update the docs**

In `SCHEMA.md`, delete the three `benchmark.*` table rows and replace the "One canonical benchmark" convention bullet with:

```markdown
- **Benchmark numbers are not stored here.** The anchor number is the Artificial
  Analysis Intelligence Index, fetched by `scripts/pull_aa.py` into
  `aa_scores.yaml` and joined at render time. Nothing hand-copies a score into
  `models.yaml` — a figure with no provenance is worse than no figure.
```

In `CLAUDE.md`, replace line 59 with:

```markdown
- **No benchmark field.** The anchor number is the Artificial Analysis Intelligence Index in `aa_scores.yaml`, written by `scripts/pull_aa.py` and joined on `hf_repo` at render time. AA covers recent models only, so `—` is expected and correct for older rows.
```

In `CLAUDE.md`, replace the `python scripts/pull_leaderboard.py` line in the Commands block with:

```bash
python scripts/pull_aa.py           # scrape AA Intelligence Index -> aa_scores.yaml
```

- [ ] **Step 6: Regenerate the README and confirm CI would pass**

```bash
.venv/bin/python scripts/render_readme.py
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/validate.py
git diff --stat README.md
```

Expected: the README table has an `AA Index` column, no MMLU columns, and three rows carry a number.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: replace the HF MMLU column with the AA Intelligence Index

Deletes pull_leaderboard.py, leaderboard_scores.yaml and the benchmark block.
HF Open LLM Leaderboard v2 publishes no plain MMLU and is archived, and the HF
card API returns no structured eval data for any tracked model, so the column
could never be filled automatically. AA Index replaces it."
```

- [ ] **Step 8: Update the discovery workflow**

In `.github/workflows/discover.yml`, add a step after the arena pull so the sidecar refreshes weekly:

```yaml
      - name: Pull Artificial Analysis scores
        continue-on-error: true    # AA must never block discovery
        run: python scripts/pull_aa.py
```

and add `aa_scores.yaml` to the `add-paths` list of the `create-pull-request` step.

```bash
git add .github/workflows/discover.yml
git commit -m "ci: refresh aa_scores.yaml on the weekly discovery run"
```
