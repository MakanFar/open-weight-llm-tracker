# Automatic Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discovery enriches each model from authoritative sources, then routes it: notable-and-complete rows are appended to `models.yaml`, notable-and-incomplete rows stay in `candidates.yaml`, and everything else is never staged — all landing as one reviewable PR.

**Architecture:** A new `scripts/enrich.py` recovers vendor-published activation figures and real licence strings that HF metadata alone does not expose. A new `scripts/classify.py` decides notable / promotable / review. `discover.py` rebuilds the queue each run from those verdicts and appends promotions to `models.yaml` without ever touching an existing row.

**Tech Stack:** Python 3.9+, `requests`, `PyYAML`, `pytest`. All already in `requirements.txt`.

## Global Constraints

- Run everything with the repo venv: `.venv/bin/python`. `pytest` is not on the base interpreter.
- TDD is mandatory: write the failing test, watch it fail for the right reason, implement minimally, watch it pass, commit.
- No test may touch the network. Enrichment is tested against committed fixtures and injected fetchers.
- **Never fabricate a field.** Enrichment returns `None` when a value is not published; `None` becomes a review reason. Computing `params_active_b` from expert geometry is forbidden — `validate.py` only enforces `active == total` for *dense* rows, so a wrong MoE figure passes silently.
- **Append-only:** promotion may only add rows to `models.yaml`. It must never modify, reorder or delete an existing row.
- Notability bar: `aa_index` or `arena_rank` or `downloads >= 500_000`.
- `render_readme.py` stays offline and deterministic; CI re-renders and fails on any diff.
- Baseline at plan time: 220 tests passing, `main` at `84d8898`, working tree has an uncommitted 358-row `candidates.yaml` from a backfill.

---

### Task 1: `family_stem` — the version-collapsing key

`names.repo_identity()` is an exact-model key: it keeps size AND version tokens, so `glm52 != glm51`. Family collision needs a coarser key that collapses version bumps but keeps distinct sizes and distinct product lines apart.

**Files:**
- Modify: `scripts/names.py`
- Test: `tests/test_names.py`

**Interfaces:**
- Consumes: `names.repo_identity(repo_id) -> str`, `names.slug(text) -> str`
- Produces: `names.family_stem(repo_id) -> str`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_names.py`:

```python
def test_family_stem_collapses_a_version_bump():
    """GLM-5.1 and GLM-5.2 are the same family; promoting both breaks
    one-row-per-model."""
    assert names.family_stem("zai-org/GLM-5.2") == names.family_stem("zai-org/GLM-5.1")
    assert names.family_stem("zai-org/GLM-5.2") == "glm"


def test_family_stem_keeps_distinct_sizes_apart():
    """Both are tracked today; collapsing them would send siblings to review."""
    assert names.family_stem("meta-llama/Llama-3.1-405B-Instruct") == "llama405b"
    assert names.family_stem("meta-llama/Llama-3.1-8B-Instruct") == "llama8b"


def test_family_stem_keeps_distinct_product_lines_apart():
    scout = names.family_stem("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    maverick = names.family_stem("meta-llama/Llama-4-Maverick-17B-128E-Instruct")
    assert scout != maverick
    assert scout == "llamascout17b16e"


def test_family_stem_strips_letter_prefixed_versions():
    assert names.family_stem("moonshotai/Kimi-K3") == "kimi"
    assert names.family_stem("moonshotai/Kimi-K2.7-Code") == "kimicode"


def test_family_stem_keeps_expert_and_size_tokens():
    """A22B / 16E end in a letter — they are counts, not versions."""
    assert names.family_stem("Qwen/Qwen3-235B-A22B") == "qwen235ba22b"


def test_family_stem_of_an_empty_tail_is_empty():
    assert names.family_stem("org/") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_names.py -q`
Expected: FAIL with `AttributeError: module 'names' has no attribute 'family_stem'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/names.py`, directly below `repo_identity`:

```python
# A token that is a version, not a size: an optional leading letter then digits
# and dots. "4", "5.2", "K3", "V4" match; "405B", "17B", "16E", "A22B" do not,
# because they end in a letter and therefore denote a size or an expert count.
_VERSION_TOKEN = re.compile(r"^[A-Za-z]?\d+(?:\.\d+)*$")


def family_stem(repo_id):
    """A model FAMILY key: repo_identity with version tokens removed.

    Deliberately coarser than repo_identity and deliberately finer than a bare
    vendor name. repo_identity keeps versions, so it can never detect that
    GLM-5.2 supersedes GLM-5.1. Stripping more than versions would be worse:
    dropping sizes collapses Llama-3.1-405B onto Llama-3.1-8B, and dropping
    words collapses Llama-4-Scout onto Llama-4-Maverick — all four are
    legitimately tracked as separate rows.

    So this fires on exactly one shape: a version bump at the same size, which
    is the supersede-or-coexist call a human should make.
    """
    author, _, tail = repo_id.rpartition("/")
    parts = [p for p in re.split(r"[-_.]", tail) if p]

    if author and parts and parts[0].lower() == author.split("/")[-1].lower():
        parts = parts[1:]

    while len(parts) > 1 and (parts[-1].lower() in PRECISION_TOKENS
                              or parts[-1].lower() in VARIANT_SUFFIXES
                              or DATE_TOKEN.match(parts[-1])):
        parts.pop()

    return slug("-".join(p for p in parts if not _VERSION_TOKEN.match(p)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_names.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, 226 tests (220 baseline + 6 new)

- [ ] **Step 6: Commit**

```bash
git add scripts/names.py tests/test_names.py
git commit -m "feat(names): family_stem, a version-collapsing model family key"
```

---

### Task 2: Read vendor-published activation figures from model cards

15 of 17 signalled candidates are blocked on `params_active_b`. The figure is not in any HF API field, but vendors state it in the model card. Hand-testing found it for 5 of 7 models in four distinct phrasings.

**Files:**
- Create: `scripts/enrich.py`
- Create: `tests/fixtures/cards/` (four fixture files)
- Test: `tests/test_enrich.py`

**Interfaces:**
- Consumes: nothing
- Produces: `enrich.CARD_URL`, `enrich.fetch_card(repo, get_text) -> str | None`, `enrich.active_params_from_card(text) -> tuple[float, str] | None`

- [ ] **Step 1: Create the fixtures**

Create four files under `tests/fixtures/cards/`, each an excerpt of a real card.

`deepseek.md`:
```markdown
We introduce two Mixture-of-Experts (MoE) language models — **DeepSeek-V4-Pro**
with 1.6T parameters (49B activated) and **DeepSeek-V4-Flash** with 284B
parameters (13B activated).
```

`minimax.md`:
```markdown
MiniMax-M3 is a native multimodal model with 1M context. It has ~428B parameters
and ~23B activated parameters.
```

`hy3.md`:
```markdown
**Hy3** is a 295B-parameter Mixture-of-Experts (MoE) model with 21B active
parameters and 3.8B MTP layer parameters, developed by the Tencent Hy Team.
```

`kimi.md`:
```markdown
<td><strong>Activated Parameters</strong></td>
<td>104B</td>
<td><strong>Number of Layers</strong></td>
<td>93</td>
```

`nofigure.md`:
```markdown
# GLM-5.2

GLM-5.2 is our latest flagship model. See the tech report for details.
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_enrich.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import enrich

CARDS = Path(__file__).resolve().parent / "fixtures" / "cards"


def _card(name):
    return (CARDS / name).read_text()


def test_reads_parenthesised_activated_figure():
    got = enrich.active_params_from_card(_card("deepseek.md"))
    assert got is not None
    value, quote = got
    assert value == 49.0
    assert "49B activated" in quote


def test_reads_approximate_activated_parameters():
    value, _ = enrich.active_params_from_card(_card("minimax.md"))
    assert value == 23.0


def test_reads_active_parameters_phrasing():
    value, _ = enrich.active_params_from_card(_card("hy3.md"))
    assert value == 21.0


def test_reads_a_table_cell():
    value, _ = enrich.active_params_from_card(_card("kimi.md"))
    assert value == 104.0


def test_returns_none_when_the_card_states_no_figure():
    """None means 'ask a human', never 0 — a 0 would look like a real value."""
    assert enrich.active_params_from_card(_card("nofigure.md")) is None


def test_returns_none_for_empty_input():
    assert enrich.active_params_from_card("") is None
    assert enrich.active_params_from_card(None) is None


def test_fetch_card_returns_none_on_failure():
    def boom(url):
        raise RuntimeError("404")
    assert enrich.fetch_card("org/m", get_text=boom) is None


def test_fetch_card_returns_the_body():
    assert enrich.fetch_card("org/m", get_text=lambda url: "# card") == "# card"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'enrich'`

- [ ] **Step 4: Write minimal implementation**

Create `scripts/enrich.py`:

```python
#!/usr/bin/env python3
"""
Recover fields that Hugging Face metadata does not expose, from sources that
publish them authoritatively.

WHY THIS EXISTS:
    Of 358 discovered candidates, models carrying a third-party signal split 2
    complete / 15 incomplete, while unsignalled models split 110 / 231. A gate
    on metadata completeness alone therefore publishes research artifacts and
    blocks every flagship — and 15 of the 17 signalled rows fail on ONE field,
    params_active_b, because every frontier model is MoE.

    That field is absent from the HF API but stated in the model card. So the
    fix is to read harder, not to gate harder.

NEVER FABRICATE:
    Every function here returns None rather than a guess. params_active_b in
    particular must not be computed from expert geometry: routing is not a
    simple ratio, and validate.py enforces active == total only for DENSE
    rows, so a wrong MoE figure passes validation silently and is then
    indistinguishable from a vendor-published one.
"""
import re
import urllib.request

CARD_URL = "https://huggingface.co/{repo}/resolve/main/README.md"


def _http_get_text(url):
    """Default fetcher. Injected in tests so no test touches the network."""
    req = urllib.request.Request(url, headers={"User-Agent": "owlt-enrich/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")

# Vendor phrasings observed in real cards, most specific first. Each must
# capture the number in group 1 and its unit in group 2.
_ACTIVE_PATTERNS = (
    # "1.6T parameters (49B activated)"
    re.compile(r"\(\s*~?\s*(\d+(?:\.\d+)?)\s*([BTM])\s*activated\s*\)", re.I),
    # "~23B activated parameters" / "21B active parameters"
    re.compile(r"~?\s*(\d+(?:\.\d+)?)\s*([BTM])\s*activ(?:e|ated)\s+param", re.I),
    # "Activated Parameters</td><td>104B"
    re.compile(r"Activ(?:e|ated)\s+Parameters?.{0,120}?>\s*~?(\d+(?:\.\d+)?)\s*([BTM])",
               re.I | re.S),
    # "| Activated Parameters | 104B |"
    re.compile(r"Activ(?:e|ated)\s+Parameters?\s*\|\s*~?(\d+(?:\.\d+)?)\s*([BTM])", re.I),
)

_UNIT_TO_B = {"b": 1.0, "t": 1000.0, "m": 0.001}


def active_params_from_card(text):
    """(billions, quoted_source) from a model card, or None if not stated.

    None is the important case: it routes the row to human review. Returning 0
    or a computed estimate would be worse than useless, because nothing
    downstream can tell an invented number from a published one.
    """
    if not text:
        return None
    for pattern in _ACTIVE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        value = float(m.group(1)) * _UNIT_TO_B[m.group(2).lower()]
        quote = " ".join(m.group(0).split())
        return round(value, 1), quote
    return None


def fetch_card(repo, get_text=_http_get_text):
    """The repo's README.md, or None on any failure."""
    try:
        return get_text(CARD_URL.format(repo=repo))
    except Exception:
        return None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: PASS, 8 tests

- [ ] **Step 6: Verify against the live cards it was derived from**

Run:
```bash
.venv/bin/python -c "
import sys, urllib.request; sys.path.insert(0,'scripts')
import enrich
def get(u): return urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'owlt/1.0'}), timeout=30).read().decode('utf-8','replace')
for r in ['deepseek-ai/DeepSeek-V4-Pro','MiniMaxAI/MiniMax-M3','tencent/Hy3','moonshotai/Kimi-K3','XiaomiMiMo/MiMo-V2.5-Pro','zai-org/GLM-5.2']:
    print(r, enrich.active_params_from_card(enrich.fetch_card(r, get)))
"
```
Expected: real figures for DeepSeek-V4-Pro (49.0), MiniMax-M3 (23.0), Hy3 (21.0), Kimi-K3 (104.0), MiMo-V2.5-Pro (42.0); `None` for GLM-5.2, which publishes no figure. If any of the first five returns `None`, add its phrasing as a new fixture and a new pattern — do not loosen an existing pattern until it matches something it should not.

- [ ] **Step 7: Commit**

```bash
git add scripts/enrich.py tests/test_enrich.py tests/fixtures/cards/
git commit -m "feat(enrich): read vendor-published activation figures from model cards"
```

---

### Task 3: Recover real licence strings

79 rows fail the allowlist on `license: other`, and 39 more on Llama tag spellings. Both are recoverable: `cardData.license_name` holds the real identifier, and the Llama failures are a spelling mismatch between HF tags and `validate.LICENSES`.

**Files:**
- Modify: `scripts/enrich.py`
- Test: `tests/test_enrich.py`

**Interfaces:**
- Consumes: `enrich` from Task 2
- Produces: `enrich.LICENSE_TAG_MAP`, `enrich.license_string(info) -> str | None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_enrich.py`:

```python
class FakeInfo:
    def __init__(self, card_data):
        self.card_data = card_data


def test_license_passes_through_a_usable_tag():
    assert enrich.license_string(FakeInfo({"license": "mit"})) == "mit"
    assert enrich.license_string(FakeInfo({"license": "apache-2.0"})) == "apache-2.0"


def test_license_maps_hf_llama_tags_to_allowlist_spellings():
    """HF tags say llama3.1; validate.LICENSES says llama-3.1-community."""
    assert enrich.license_string(FakeInfo({"license": "llama3.1"})) == "llama-3.1-community"
    assert enrich.license_string(FakeInfo({"license": "llama3.3"})) == "llama-3.3-community"
    assert enrich.license_string(FakeInfo({"license": "llama4"})) == "llama-4-community"


def test_license_recovers_the_real_name_when_the_tag_is_other():
    """'other' is not a licence. cardData.license_name carries the real one."""
    info = FakeInfo({"license": "other", "license_name": "kimi-k3"})
    assert enrich.license_string(info) == "kimi-k3"


def test_license_normalises_a_recovered_name():
    info = FakeInfo({"license": "other", "license_name": "MiniMax Community"})
    assert enrich.license_string(info) == "minimax-community"


def test_license_returns_none_when_other_has_no_name():
    assert enrich.license_string(FakeInfo({"license": "other"})) is None


def test_license_returns_none_when_absent():
    assert enrich.license_string(FakeInfo({})) is None
    assert enrich.license_string(FakeInfo(None)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: FAIL with `AttributeError: module 'enrich' has no attribute 'license_string'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/enrich.py`:

```python
# HF licence tags whose spelling differs from validate.LICENSES. The tracker
# uses the vendor's own name for the licence; HF uses a short tag.
LICENSE_TAG_MAP = {
    "llama2": "llama-2-community",
    "llama3": "llama-3-community",
    "llama3.1": "llama-3.1-community",
    "llama3.2": "llama-3.2-community",
    "llama3.3": "llama-3.3-community",
    "llama4": "llama-4-community",
}


def _card_dict(info):
    cd = getattr(info, "card_data", None)
    if cd is None:
        return {}
    return cd.to_dict() if hasattr(cd, "to_dict") else dict(cd)


def license_string(info):
    """The real licence identifier, or None if it cannot be determined.

    'other' is not a licence, it is HF's way of saying "custom". The actual
    name lives in cardData.license_name — that is how kimi-k3,
    minimax-community, modified-mit and nvidia-open-model-license were
    recovered from rows the tag alone marked unusable.
    """
    data = _card_dict(info)
    tag = data.get("license")
    if not isinstance(tag, str) or not tag:
        return None
    tag = tag.strip().lower()
    if tag != "other":
        return LICENSE_TAG_MAP.get(tag, tag)

    name = data.get("license_name")
    if not isinstance(name, str) or not name.strip():
        return None
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-") or None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: PASS, 15 tests

- [ ] **Step 4b: Write the failing test for context-window recovery**

80 discovered rows sit at `context_window: 0` because neither the API expand nor `config.json` carried a length. `tokenizer_config.json` usually does. Append to `tests/test_enrich.py`:

```python
def test_context_from_tokenizer_config():
    cfg = {"model_max_length": 262144}
    assert enrich.context_from_tokenizer("org/m", get_json=lambda u: cfg) == 262144


def test_context_ignores_the_sentinel_length():
    """Transformers writes a huge int32 sentinel meaning 'unset'."""
    cfg = {"model_max_length": 1000000000000000019884624838656}
    assert enrich.context_from_tokenizer("org/m", get_json=lambda u: cfg) is None


def test_context_returns_none_when_absent_or_unfetchable():
    assert enrich.context_from_tokenizer("org/m", get_json=lambda u: {}) is None

    def boom(url):
        raise RuntimeError("404")
    assert enrich.context_from_tokenizer("org/m", get_json=boom) is None
```

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: FAIL with `AttributeError: module 'enrich' has no attribute 'context_from_tokenizer'`

- [ ] **Step 4c: Implement it**

Add to `scripts/enrich.py`:

```python
TOKENIZER_URL = "https://huggingface.co/{repo}/resolve/main/tokenizer_config.json"

# Transformers writes this when model_max_length is unset. It is not a context
# length; treating it as one would publish a 1e30-token window.
_SENTINEL_MAX_LENGTH = 1_000_000_000_000_000

def context_from_tokenizer(repo, get_json):
    """Context length from tokenizer_config.json, or None.

    Tried only after config.json has already failed — 80 discovered rows had
    no length in either the API expand or config.json, and this is where the
    remainder publish it.
    """
    try:
        cfg = get_json(TOKENIZER_URL.format(repo=repo))
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    n = cfg.get("model_max_length")
    if isinstance(n, int) and not isinstance(n, bool) and 0 < n < _SENTINEL_MAX_LENGTH:
        return n
    return None
```

Run: `.venv/bin/python -m pytest tests/test_enrich.py -q`
Expected: PASS

- [ ] **Step 5: Add the mapped Llama spellings to the validator allowlist**

`LICENSE_TAG_MAP` maps to `llama-2-community` and `llama-3-community`, which are not yet in `scripts/validate.py`. Add both so a mapped row can pass:

```python
LICENSES = {
    "apache-2.0", "mit", "bsd-3-clause",
    "llama-2-community", "llama-3-community",
    "llama-3.1-community", "llama-3.2-community",
    "llama-3.3-community", "llama-4-community",
    "qwen", "gemma", "deepseek",
    "cc-by-nc-4.0", "cc-by-4.0",
}
```

Run: `.venv/bin/python scripts/validate.py`
Expected: `OK — 16 models, no problems found.`

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, 241 tests

- [ ] **Step 7: Commit**

```bash
git add scripts/enrich.py scripts/validate.py tests/test_enrich.py
git commit -m "feat(enrich): recover real licence strings from cardData"
```

---

### Task 4: The classifier

**Files:**
- Create: `scripts/classify.py`
- Test: `tests/test_classify.py`

**Interfaces:**
- Consumes: `names.family_stem` (Task 1), `validate.LICENSES`
- Produces: `classify.NOTABILITY_DOWNLOADS`, `classify.is_notable(row) -> bool`, `classify.missing_vitals(row, tracked_stems) -> list[str]`, `classify.route(row, tracked_stems) -> str` returning `"promote"` / `"review"` / `"drop"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_classify.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import classify


def _row(**kw):
    base = dict(name="M", hf_repo="org/m", developer="org",
                params_total_b=70.0, params_active_b=70.0, architecture="dense",
                context_window=131072, license="mit", downloads=0)
    base.update(kw)
    return base


# --- notability ------------------------------------------------------------

def test_notable_via_aa_index():
    assert classify.is_notable(_row(aa_index=57)) is True


def test_notable_via_arena_rank():
    assert classify.is_notable(_row(arena_rank=12)) is True


def test_notable_via_downloads_at_the_boundary():
    assert classify.is_notable(_row(downloads=500_000)) is True
    assert classify.is_notable(_row(downloads=499_999)) is False


def test_not_notable_with_no_signal_and_few_downloads():
    """This is what keeps research artifacts out of models.yaml."""
    assert classify.is_notable(_row(downloads=1200)) is False


# --- vitals ----------------------------------------------------------------

def test_complete_dense_row_has_no_missing_vitals():
    assert classify.missing_vitals(_row(), set()) == []


def test_moe_with_active_equal_to_total_is_incomplete():
    """The blocker on 15 of 17 signalled rows."""
    row = _row(architecture="moe", params_total_b=753.3, params_active_b=753.3)
    assert "moe-active-params-unknown" in classify.missing_vitals(row, set())


def test_moe_with_a_real_active_figure_is_complete():
    row = _row(architecture="moe", params_total_b=753.3, params_active_b=32.0)
    assert classify.missing_vitals(row, set()) == []


def test_zero_context_window_is_incomplete():
    assert "no-context-window" in classify.missing_vitals(_row(context_window=0), set())


def test_unallowlisted_licence_is_incomplete():
    assert "license-not-allowlisted" in classify.missing_vitals(_row(license="other"), set())


def test_inexact_repo_match_is_incomplete():
    assert "inexact-repo-match" in classify.missing_vitals(_row(needs_hf_repo=True), set())


def test_family_already_tracked_is_incomplete():
    """GLM-5.2 must not auto-promote while GLM-5.1 is tracked."""
    row = _row(hf_repo="zai-org/GLM-5.2")
    assert "family-already-tracked" in classify.missing_vitals(row, {"glm"})


def test_reports_every_reason_not_just_the_first():
    row = _row(architecture="moe", params_total_b=100.0, params_active_b=100.0,
               context_window=0, license="other")
    reasons = classify.missing_vitals(row, set())
    assert set(reasons) >= {"moe-active-params-unknown", "no-context-window",
                            "license-not-allowlisted"}


# --- routing ---------------------------------------------------------------

def test_notable_and_complete_promotes():
    assert classify.route(_row(aa_index=29), set()) == "promote"


def test_notable_and_incomplete_goes_to_review():
    row = _row(aa_index=51, architecture="moe",
               params_total_b=753.3, params_active_b=753.3)
    assert classify.route(row, set()) == "review"


def test_not_notable_is_dropped_even_when_complete():
    """110 complete-but-unremarkable rows must never reach models.yaml."""
    assert classify.route(_row(downloads=10), set()) == "drop"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_classify.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'classify'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/classify.py`:

```python
#!/usr/bin/env python3
"""
Decide what happens to a discovered model: promote, review, or drop.

WHY A NOTABILITY BAR EXISTS:
    Completeness and worth are anti-correlated here. Of 358 discovered rows,
    those carrying a third-party signal were 2 complete / 15 incomplete, while
    those with no signal were 110 complete / 231 incomplete. Promoting on
    completeness alone would publish MagenticBrain, BAR-7B and granite
    previews while blocking Kimi K3, GLM-5.2 and gpt-oss-120b.

WHY DOWNLOADS AND NOT JUST THE LEADERBOARDS:
    A signal-only bar admits 17 rows and would reject 13 of the 16 models this
    tracker already curates by hand — a bar its own editorial practice
    rejects. Downloads catches the flagships no leaderboard rates (Llama-3-8B,
    the Qwen3-2507 line, GLM-4.7-Flash). It skews small, which is a known and
    accepted weakness: a large model with modest adoption and no leaderboard
    coverage will not auto-promote.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names
import validate

NOTABILITY_DOWNLOADS = 500_000


def is_notable(row):
    """True if the model is worth publishing without a human asking for it."""
    if row.get("aa_index") is not None or row.get("arena_rank") is not None:
        return True
    return (row.get("downloads") or 0) >= NOTABILITY_DOWNLOADS


def missing_vitals(row, tracked_stems):
    """Every reason this row cannot be promoted unreviewed. [] means it can.

    Returns ALL reasons rather than the first, so one review pass shows a
    human everything the row needs.
    """
    reasons = []

    if row.get("architecture") == "moe" and \
            row.get("params_active_b") == row.get("params_total_b"):
        reasons.append("moe-active-params-unknown")

    ctx = row.get("context_window")
    if not isinstance(ctx, int) or isinstance(ctx, bool) or ctx <= 0:
        reasons.append("no-context-window")

    if row.get("license") not in validate.LICENSES:
        reasons.append("license-not-allowlisted")

    if row.get("needs_hf_repo"):
        reasons.append("inexact-repo-match")

    stem = names.family_stem(row.get("hf_repo") or "")
    if stem and stem in tracked_stems:
        reasons.append("family-already-tracked")

    return reasons


def route(row, tracked_stems):
    """'promote' | 'review' | 'drop'."""
    if not is_notable(row):
        return "drop"
    return "review" if missing_vitals(row, tracked_stems) else "promote"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_classify.py -q`
Expected: PASS, 16 tests

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, 257 tests

- [ ] **Step 6: Commit**

```bash
git add scripts/classify.py tests/test_classify.py
git commit -m "feat(classify): notability bar and vital-field gate"
```

---

### Task 5: Wire enrichment into candidate construction

Enrichment must run before classification, or every MoE row is incomplete regardless of what its card says.

**Files:**
- Modify: `scripts/discover.py`
- Test: `tests/test_discover_enrich.py`

**Interfaces:**
- Consumes: `enrich.fetch_card`, `enrich.active_params_from_card`, `enrich.license_string`
- Produces: `discover.enrich_row(row, info, get_text, get_json) -> dict` (mutates and returns)

- [ ] **Step 1: Write the failing test**

Create `tests/test_discover_enrich.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover
from test_enrich import FakeInfo


def _row(**kw):
    base = dict(name="M", hf_repo="org/m", architecture="moe",
                params_total_b=753.3, params_active_b=753.3, license="other")
    base.update(kw)
    return base


CARD = "It has ~428B parameters and ~23B activated parameters."


def test_enrich_fills_active_params_from_the_card():
    row = _row()
    discover.enrich_row(row, FakeInfo({"license": "mit"}), get_text=lambda u: CARD)
    assert row["params_active_b"] == 23.0
    assert "23B activated" in row["params_active_source"]


def test_enrich_leaves_active_params_alone_when_the_card_is_silent():
    row = _row()
    discover.enrich_row(row, FakeInfo({"license": "mit"}), get_text=lambda u: "no figure")
    assert row["params_active_b"] == 753.3
    assert "params_active_source" not in row


def test_enrich_does_not_touch_a_dense_row():
    """Dense rows must keep active == total; validate.py enforces it."""
    row = _row(architecture="dense", params_total_b=70.0, params_active_b=70.0)
    discover.enrich_row(row, FakeInfo({"license": "mit"}), get_text=lambda u: CARD)
    assert row["params_active_b"] == 70.0


def test_enrich_recovers_the_licence():
    row = _row()
    discover.enrich_row(row, FakeInfo({"license": "other", "license_name": "kimi-k3"}),
                        get_text=lambda u: "")
    assert row["license"] == "kimi-k3"


def test_enrich_keeps_the_original_licence_when_nothing_better_is_found():
    row = _row(license="other")
    discover.enrich_row(row, FakeInfo({"license": "other"}), get_text=lambda u: "")
    assert row["license"] == "other"


def test_enrich_survives_a_card_fetch_failure():
    def boom(url):
        raise RuntimeError("429")
    row = _row()
    discover.enrich_row(row, FakeInfo({"license": "mit"}), get_text=boom)
    assert row["params_active_b"] == 753.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_discover_enrich.py -q`
Expected: FAIL with `AttributeError: module 'discover' has no attribute 'enrich_row'`

- [ ] **Step 3: Write minimal implementation**

Add `import enrich` beside the existing `import hf_meta` in `scripts/discover.py`, then add:

```python
def enrich_row(row, info, get_text):
    """Fill fields the HF API does not expose, in place. Never fabricates.

    Only MoE rows get an activation lookup: a dense row's active params equal
    its total by definition, and validate.py enforces that, so writing a
    card-derived figure onto one could only break it.

    params_active_source records the exact sentence the number came from, so a
    reviewer can check the claim without re-reading the card.
    """
    if row.get("architecture") == "moe" and \
            row.get("params_active_b") == row.get("params_total_b"):
        found = enrich.active_params_from_card(
            enrich.fetch_card(row["hf_repo"], get_text))
        if found is not None:
            row["params_active_b"], row["params_active_source"] = found

    if not row.get("context_window"):
        ctx = enrich.context_from_tokenizer(row["hf_repo"], get_json)
        if ctx:
            row["context_window"] = ctx

    lic = enrich.license_string(info)
    if lic:
        row["license"] = lic
    return row
```

`enrich_row`'s signature is therefore `enrich_row(row, info, get_text, get_json)` — two injected fetchers, because the card is text and tokenizer_config is JSON. Add a matching test:

```python
def test_enrich_recovers_a_missing_context_window():
    row = _row(context_window=0)
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: "", get_json=lambda u: {"model_max_length": 262144})
    assert row["context_window"] == 262144
```

Every other test in this task passes `get_json=lambda u: {}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_discover_enrich.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, 263 tests

- [ ] **Step 6: Commit**

```bash
git add scripts/discover.py tests/test_discover_enrich.py
git commit -m "feat(discover): enrich a candidate row before classifying it"
```

---

### Task 6: Append-only promotion into models.yaml

This is the task that reverses the project's oldest invariant, so its guard rails are the deliverable as much as its feature.

**Files:**
- Modify: `scripts/discover.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `names.family_stem`, `classify.route`
- Produces: `discover.tracked_stems(path) -> set[str]`, `discover.PROMOTION_STRIP_FIELDS`, `discover.promotion_row(candidate) -> dict`, `discover.append_models(path, rows) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_promote.py`:

```python
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover

EXISTING = """\
# Open-weight tracker — SOURCE OF TRUTH
# Hand-curated. Do not reformat.

models:
  - name: Llama 3.3 70B Instruct
    hf_repo: meta-llama/Llama-3.3-70B-Instruct
    developer: Meta
    release_date: 2024-12-06
    params_total_b: 70
    params_active_b: 70
    architecture: dense
    context_window: 131072
    modality: text
    license: llama-3.3-community
    commercial_use: conditional
    license_notes: "Hand-written note that must survive."
"""


def _candidate(**kw):
    base = dict(name="GLM-5.2", hf_repo="zai-org/GLM-5.2", developer="zai-org",
                release_date="2026-06-16", params_total_b=753.3,
                params_active_b=32.0, architecture="moe", context_window=1048576,
                modality="text", license="mit", commercial_use=True,
                weights_url="https://huggingface.co/zai-org/GLM-5.2",
                discovered_via=["arena"], arena_rank=12, aa_index=51,
                downloads=1651533, needs_hf_repo=False,
                resolution_confidence="high", params_active_source="32B activated")
    base.update(kw)
    return base


def test_tracked_stems_reads_models_yaml(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)
    assert discover.tracked_stems(p) == {"llama70b"}


def test_promotion_row_strips_discovery_only_fields():
    row = discover.promotion_row(_candidate())
    for field in ("discovered_via", "arena_rank", "aa_index", "downloads",
                  "needs_hf_repo", "resolution_confidence"):
        assert field not in row, f"{field} must not leak into models.yaml"


def test_promotion_row_keeps_the_activation_provenance():
    """A reviewer must be able to check where 32B came from."""
    assert discover.promotion_row(_candidate())["params_active_source"] == "32B activated"


def test_promotion_row_marks_commercial_use_unverified():
    assert discover.promotion_row(_candidate())["commercial_use_verified"] is False


def test_append_leaves_existing_rows_byte_identical(tmp_path):
    """The invariant. A human's hand-edited row must survive untouched."""
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)

    discover.append_models(p, [discover.promotion_row(_candidate())])

    text = p.read_text()
    assert EXISTING.rstrip("\n") in text, "existing content was rewritten"
    assert "Hand-written note that must survive." in text
    assert text.startswith("# Open-weight tracker"), "comment header lost"


def test_append_adds_the_new_row(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)

    n = discover.append_models(p, [discover.promotion_row(_candidate())])

    assert n == 1
    doc = yaml.safe_load(p.read_text())
    assert [m["hf_repo"] for m in doc["models"]] == [
        "meta-llama/Llama-3.3-70B-Instruct", "zai-org/GLM-5.2"]


def test_append_of_nothing_leaves_the_file_untouched(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)
    before = p.read_text()

    assert discover.append_models(p, []) == 0
    assert p.read_text() == before


def test_appended_file_still_parses_and_validates_shape(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)
    discover.append_models(p, [discover.promotion_row(_candidate())])

    doc = yaml.safe_load(p.read_text())
    new = doc["models"][-1]
    for field in ("name", "developer", "release_date", "params_total_b",
                  "params_active_b", "architecture", "context_window",
                  "modality", "license", "commercial_use"):
        assert field in new
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_promote.py -q`
Expected: FAIL with `AttributeError: module 'discover' has no attribute 'tracked_stems'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/discover.py`:

```python
# Discovery-only fields. SCHEMA.md requires these stripped on promotion —
# validate.py checks models.yaml only, so they would otherwise leak in.
PROMOTION_STRIP_FIELDS = ("discovered_via", "arena_rank", "aa_index",
                          "downloads", "needs_hf_repo", "resolution_confidence")


def tracked_stems(path=DATA):
    """Family stems already present in models.yaml."""
    return {names.family_stem(r["hf_repo"]) for r in _rows_of(path)
            if r.get("hf_repo")}


def promotion_row(candidate):
    """A candidate reshaped for models.yaml.

    commercial_use_verified is stamped False because the value came from a
    licence-tag inference, not from anyone reading the licence. The renderer
    marks unverified values so a reader can tell an inferred claim from a
    checked one; a human setting it True is an ordinary edit that append-only
    preserves.
    """
    row = {k: v for k, v in candidate.items() if k not in PROMOTION_STRIP_FIELDS}
    row.setdefault("commercial_use_verified", False)
    return row


def append_models(path, rows):
    """Append rows to models.yaml. Returns the count appended.

    APPEND-ONLY, and literally so: the existing file is not parsed and
    re-dumped, it is read as text and added to. Round-tripping through
    yaml.safe_dump would reflow every hand-curated row and drop the comment
    header, which is precisely the human ownership this file exists to hold.
    """
    rows = list(rows)
    if not rows:
        return 0
    body = yaml.safe_dump({"models": rows}, sort_keys=False,
                          allow_unicode=True, width=100)
    body = body[len("models:\n"):] if body.startswith("models:\n") else body
    existing = Path(path).read_text()
    if not existing.endswith("\n"):
        existing += "\n"
    Path(path).write_text(existing + body)
    return len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_promote.py -q`
Expected: PASS, 8 tests

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, 271 tests

- [ ] **Step 6: Commit**

```bash
git add scripts/discover.py tests/test_promote.py
git commit -m "feat(discover): append-only promotion into models.yaml"
```

---

### Task 7: Route the queue, and document the reversed invariant

**Files:**
- Modify: `scripts/discover.py` (`refresh`, `main`), `scripts/render_readme.py`, `SCHEMA.md`, `CLAUDE.md`, `.github/workflows/discover.yml`
- Test: `tests/test_discover_queue.py`

**Interfaces:**
- Consumes: everything above
- Produces: `refresh(...)` returns `(promoted, queue, skips, new_orgs)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_discover_queue.py`:

```python
def test_refresh_splits_promotable_from_reviewable(tmp_path):
    """The whole point: complete+notable leaves, incomplete+notable waits,
    unremarkable never appears."""
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "Ready", "hf_repo": "org/ready", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 70.0,
         "params_active_b": 70.0, "architecture": "dense",
         "context_window": 131072, "modality": "text", "license": "mit",
         "commercial_use": True, "aa_index": 40},
        {"name": "Gappy", "hf_repo": "org/gappy", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 700.0,
         "params_active_b": 700.0, "architecture": "moe",
         "context_window": 131072, "modality": "text", "license": "mit",
         "commercial_use": True, "aa_index": 45},
        {"name": "Noise", "hf_repo": "org/noise", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 7.0,
         "params_active_b": 7.0, "architecture": "dense",
         "context_window": 4096, "modality": "text", "license": "mit",
         "commercial_use": True, "downloads": 12},
    ]}))
    api = FakeApi({})

    promoted, queue, _, _ = discover.refresh(
        api, 3.0, data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        aa_path=tmp_path / "nope.yaml",
        use_arena=False, get_json=lambda url: {},
        get_text=lambda url: "", today=TODAY)

    assert [r["hf_repo"] for r in promoted] == ["org/ready"]
    assert [r["hf_repo"] for r in queue] == ["org/gappy"]


def test_refresh_records_why_a_queued_row_is_queued(tmp_path):
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "Gappy", "hf_repo": "org/gappy", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 700.0,
         "params_active_b": 700.0, "architecture": "moe",
         "context_window": 0, "modality": "text", "license": "mit",
         "commercial_use": True, "aa_index": 45}]}))

    _, queue, _, _ = discover.refresh(
        FakeApi({}), 3.0, data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        aa_path=tmp_path / "nope.yaml", use_arena=False,
        get_json=lambda url: {}, get_text=lambda url: "", today=TODAY)

    assert set(queue[0]["needs_review"]) >= {"moe-active-params-unknown",
                                             "no-context-window"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_discover_queue.py -q`
Expected: FAIL — `refresh()` returns 3 values, not 4, so the unpack raises `ValueError`.

- [ ] **Step 3: Rewrite `refresh` to classify**

Replace the tail of `refresh` in `scripts/discover.py` (from `candidates = merge_candidates(...)` to the `return`) with:

```python
    candidates = merge_candidates(staged + org_rows, arena_rows)

    aa_index = load_aa_index(aa_path)
    annotate_aa(candidates, aa_index)

    stems = tracked_stems(data_path)
    promoted, queue = [], []
    for row in candidates:
        verdict = classify.route(row, stems)
        if verdict == "drop":
            continue
        if verdict == "promote":
            promoted.append(row)
            stems.add(names.family_stem(row["hf_repo"]))
            continue
        row["needs_review"] = classify.missing_vitals(row, stems)
        queue.append(row)

    print(f"  {len(promoted)} promotable, {len(queue)} need review, "
          f"{len(candidates) - len(promoted) - len(queue)} below the bar")

    return promoted, queue, skips, new_orgs
```

Add `import classify` and `import names` beside the existing imports. Thread `get_text=enrich._http_get_text` through `refresh`'s signature and call `enrich_row(row, info, get_text, get_json)` inside `sweep_orgs` and `arena_candidates` where each row is built.

Adding a promoted row's stem to `stems` inside the loop matters: two promotable versions of the same family in one run must not both promote — the second sees the first's stem and routes to review.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_discover_queue.py -q`
Expected: PASS

- [ ] **Step 5: Update `main()` to write both files**

```python
    promoted, queue, _, new_orgs = refresh(api, args.min_params,
                                           use_arena=not args.no_arena,
                                           max_age_days=args.max_age_days)
    n = append_models(DATA, [promotion_row(r) for r in promoted])
    write_candidates(CANDIDATES, queue)
    print(f"\nPromoted {n} model(s) into {DATA.name}; "
          f"{len(queue)} awaiting review in {CANDIDATES.name}")
```

- [ ] **Step 6: Render unverified commercial_use distinguishably**

In `scripts/render_readme.py`, replace `commercial_badge`:

```python
def commercial_badge(model):
    """Commercial-use badge. Unverified values carry a trailing ?.

    A row promoted automatically carries a value inferred from the licence
    TAG, not from anyone reading the licence. Rendering it identically to a
    checked value would publish an unverified legal claim as a settled one.
    """
    label = {True: "Yes", False: "No",
             "conditional": "Conditional"}.get(model.get("commercial_use"),
                                               str(model.get("commercial_use")))
    return label if model.get("commercial_use_verified") else f"{label}?"
```

Update its call site in `build_table` from `commercial_badge(m['commercial_use'])` to `commercial_badge(m)`, and add to the Columns prose in `main()`'s `body` string:

```python
        "A trailing `?` on **Commercial** marks a value inferred from the "
        "licence tag and not yet checked against the licence text. "
```

- [ ] **Step 7: Update the docs and the workflow**

In `CLAUDE.md`, replace the sentence "Nothing automated ever writes to `models.yaml`; promotion is always a human editing the file." with:

```markdown
`discover.py` may **append** auto-promoted rows to `models.yaml`, but only ever appends — it never modifies, reorders or deletes an existing row, so anything already there stays exactly as a human left it. Every automated change lands as a PR; nothing commits to `main` directly. A row auto-promotes only if it clears the notability bar (AA index, arena rank, or ≥500k downloads) and has no missing vitals; otherwise it waits in `candidates.yaml` with a `needs_review` list.
```

In `SCHEMA.md`, add to the `models.yaml` field table:

```markdown
| `commercial_use_verified` | no | bool | `false` (or absent) means the value was inferred from the licence tag, not read from the licence. The README marks these with `?`. |
| `params_active_source` | no | string | For auto-promoted MoE rows: the sentence in the model card the activation figure came from. |
```

and to the `candidates.yaml` discovery-only table:

```markdown
| `needs_review` | list | Why the row was not auto-promoted, e.g. `moe-active-params-unknown`. Strip on promotion. |
```

Add `needs_review` to `PROMOTION_STRIP_FIELDS` in `discover.py`.

In `.github/workflows/discover.yml`, add `models.yaml` to the `create-pull-request` `add-paths` list, and change the PR body to:

```yaml
          body: |
            Automated discovery.

            **`models.yaml`** — rows auto-promoted because they cleared the
            notability bar with no missing vitals. `commercial_use` on these is
            inferred from the licence tag, not read from the licence: check any
            row rendered with a trailing `?` before merging.

            **`candidates.yaml`** — rows that need a decision. Each carries a
            `needs_review` list saying what is missing. Fill the gap here and the
            next run promotes the row automatically.
```

- [ ] **Step 8: Full verification**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/render_readme.py && git diff --stat README.md
```
Expected: suite passes; `OK — 16 models, no problems found.`; README diff shows only the new `?` markers and the added prose sentence.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: route discoveries to promotion, review, or drop

models.yaml is now append-only-writable by discover.py. Nothing commits to
main; every change lands as a PR. Unverified commercial_use renders with a
trailing ? so an inferred legal claim is not shown as a checked one."
```

---

### Task 8: First real run

The prior seven tasks are inert until run against live data. This task is where the 358-row queue collapses, and it is the one most likely to surface a bad regex or an over-eager bar.

**Files:**
- Modify: `candidates.yaml`, `models.yaml`, `aa_scores.yaml`, `README.md` (all regenerated)

- [ ] **Step 1: Reset the working queue**

The uncommitted 358-row backfill is input, not something to preserve. Confirm what is on disk:

```bash
git status --short candidates.yaml
.venv/bin/python -c "
import yaml;from pathlib import Path
print(len(yaml.safe_load(Path('candidates.yaml').read_text())['models']),'rows staged')"
```

- [ ] **Step 2: Dry-run the classifier over the existing queue**

Before writing anything, see what the rules decide:

```bash
.venv/bin/python -c "
import sys,yaml,collections;sys.path.insert(0,'scripts')
from pathlib import Path
import classify,discover
rows=yaml.safe_load(Path('candidates.yaml').read_text())['models']
stems=discover.tracked_stems()
c=collections.Counter(classify.route(r,stems) for r in rows)
print(dict(c))
for r in rows:
    if classify.route(r,stems)=='promote': print('  PROMOTE',r['hf_repo'])
"
```

Expected: mostly `drop`, a handful of `review`, and a small `promote` list. **Read the promote list before continuing.** If it contains a model you would not put in the index by hand, stop and report it — the notability bar needs discussion, not a code change.

- [ ] **Step 3: Run the real pipeline**

```bash
.venv/bin/python scripts/pull_arena.py
.venv/bin/python scripts/pull_aa.py
.venv/bin/python scripts/discover.py
```

- [ ] **Step 4: Verify the invariant held on real data**

```bash
git diff models.yaml | grep -E '^-' | grep -v '^---' || echo "APPEND-ONLY HELD: no deletions"
```
Expected: `APPEND-ONLY HELD`. Any `-` line other than `---` means an existing row was modified — stop, revert `models.yaml`, and report.

- [ ] **Step 5: Validate and render**

```bash
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/render_readme.py
.venv/bin/python -m pytest tests/ -q
```
Expected: validate reports the new model count with no problems.

- [ ] **Step 6: Report before committing**

Summarise: how many promoted and which, how many await review and why, how far the queue fell, and whether any promoted row looks wrong. **Do not commit the data change without reporting it** — this is the run that first writes to the source of truth, and a human should see the list.

- [ ] **Step 7: Commit once approved**

```bash
git add models.yaml candidates.yaml aa_scores.yaml arena_agent_rankings.yaml README.md
git commit -m "data: first automatic promotion run"
```
