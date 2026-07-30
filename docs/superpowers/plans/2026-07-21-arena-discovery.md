# Arena-Assisted Model Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the global Hugging Face firehose query with an org-scoped sweep, and use the arena.ai leaderboard to prioritize candidates and surface unknown orgs — so `candidates.yaml` stops coming back empty.

**Architecture:** Three flat scripts in `scripts/`. A new `hf_meta.py` owns the single definition of a candidate row and the filter predicates. `pull_arena.py` scrapes the leaderboard and resolves display names to HF repos (resolution success is the open-weight test). `discover.py` sweeps allowlisted orgs, optionally merges arena's resolved repos, and writes `candidates.yaml`. Arena is never load-bearing: if it is unreachable, the org sweep still produces candidates.

**Tech Stack:** Python 3.11, `huggingface_hub` (installed: 1.8.0), `PyYAML`, `requests`, `beautifulsoup4`, `jsonschema`, `pytest`.

## Global Constraints

- Source of truth is `models.yaml`. No task in this plan modifies it, and no task changes its schema. Arena data is discovery-only.
- `scripts/discover.py` must exit 0 always. The GitHub Action decides whether the `candidates.yaml` diff is non-empty.
- Arena failure is never fatal to discovery. Unreachable arena, changed markup, or zero parsed rows must log a warning and continue with the org sweep.
- Tests must not perform network I/O. All HF and HTTP interaction is injected as a callable so tests pass fakes.
- `huggingface_hub` 1.8.0 does **not** support the `direction` or `library` kwargs on `list_models`. `sort="created_at"` alone returns newest-first (verified empirically 2026-07-21). Keep the existing `inspect.signature` guard when passing optional kwargs.
- Minimum params default: `3.0` (billions). Existing `--min-params` flag semantics are preserved.
- Every new/changed script keeps its module docstring accurate — these docstrings are the repo's primary documentation.

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/hf_meta.py` | **Create.** Single definition of a candidate row + all filter predicates. Shared by both discovery sources. |
| `scripts/pull_arena.py` | **Rewrite.** Scrape leaderboard, normalize names, resolve to HF repos, write `arena_agent_rankings.yaml`. |
| `scripts/discover.py` | **Rewrite.** Org sweep, arena merge, write `candidates.yaml`. |
| `tests/fixtures/arena_sample.html` | **Create.** Hand-written leaderboard fixture mirroring the real table structure. |
| `tests/test_hf_meta.py` | **Create.** Filter predicate + candidate construction tests. |
| `tests/test_arena_names.py` | **Create.** Name normalization + match scoring tests. |
| `tests/test_arena_parse.py` | **Create.** `parse_leaderboard` against the HTML fixture. |
| `tests/test_arena_resolve.py` | **Create.** Resolution with an injected fake search. |
| `tests/test_discover_merge.py` | **Create.** Merge, dedup, and sort ordering tests. |
| `requirements.txt` | **Modify.** Add `pytest`. |
| `.github/workflows/validate.yml` | **Modify.** Add a pytest step. |
| `SCHEMA.md` | **Modify.** Document `candidates.yaml`-only fields. |

**Deleted in Task 5:** `get_org_overview`, `is_organization`, `_ORG_CACHE`, `AUTHOR_BLOCKLIST`, `--orgs-only`, `build_query`, and the `json`/`urllib` imports in `discover.py`. These existed only to police an unbounded query.

---

### Task 1: Shared candidate construction (`hf_meta.py`)

**Files:**
- Create: `scripts/hf_meta.py`
- Create: `tests/test_hf_meta.py`

> Do **not** add `tests/__init__.py`. Without it, pytest inserts each test file's own directory onto `sys.path`, which is what lets Task 5 do `from test_hf_meta import FakeInfo`. Adding it makes `tests` a package and breaks that import.

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `EXCLUDE_PATTERNS: re.Pattern`
  - `ACCEPTED_LICENSES: set[str]`, `COMMERCIAL_GUESS: dict`, `EXPAND: list[str]`, `CTX_KEYS: tuple[str, ...]`
  - `is_derivative(repo_id: str) -> bool`
  - `license_of(info) -> str | None`
  - `context_of(info) -> int | None`
  - `params_b_of(info) -> float | None`
  - `to_date(v) -> datetime.date | None`
  - `should_track(info, min_params: float) -> tuple[bool, str | None]` — second element is a skip reason: `"derivative"`, `"no_params"`, `"small"`, or `"license"`
  - `candidate_from_repo(info, discovered_via: list[str], arena_rank: int | None = None) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hf_meta.py`:

```python
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import hf_meta


class FakeInfo:
    """Stand-in for huggingface_hub.ModelInfo — only the attributes we read."""

    def __init__(self, id, total=None, license=None, ctx=None,
                 created_at=None, downloads=0):
        self.id = id
        self.safetensors = {"total": total} if total is not None else None
        self.card_data = {"license": license} if license is not None else {}
        self.config = {"max_position_embeddings": ctx} if ctx is not None else {}
        self.created_at = created_at
        self.downloads = downloads


@pytest.mark.parametrize("repo_id", [
    "zai-org/GLM-5.2-FP8",
    "MiniMaxAI/MiniMax-M3-MXFP8",
    "google/gemma-4-12B-it-qat-w4a16-ct",
    "google/gemma-4-12B-it-qat-q4_0-gguf",
    "deepseek-ai/eagle3_qwen3_8b_ttt7",
    "deepseek-ai/dflash_gemma4_12b_block7",
    "nvidia/Nemotron-3-Embed-8B-BF16",
    "TheBloke/Llama-2-70B-AWQ",
])
def test_derivatives_are_excluded(repo_id):
    assert hf_meta.is_derivative(repo_id) is True


@pytest.mark.parametrize("repo_id", [
    "zai-org/GLM-5.2",
    "MiniMaxAI/MiniMax-M3",
    "moonshotai/Kimi-K2.6",
    "moonshotai/Kimi-K2.7-Code",
    "google/gemma-4-31b-it",
    "deepseek-ai/DeepSeek-V4",
])
def test_real_models_are_kept(repo_id):
    assert hf_meta.is_derivative(repo_id) is False


def test_should_track_rejects_small_model():
    info = FakeInfo("org/tiny-1b", total=1_000_000_000, license="apache-2.0")
    keep, reason = hf_meta.should_track(info, min_params=3.0)
    assert keep is False
    assert reason == "small"


def test_should_track_rejects_zero_param_artifact():
    info = FakeInfo("google/tabfm-1.0.0-jax", total=None, license="apache-2.0")
    keep, reason = hf_meta.should_track(info, min_params=3.0)
    assert keep is False
    assert reason == "no_params"


def test_should_track_rejects_unaccepted_license():
    info = FakeInfo("org/model-8b", total=8_000_000_000, license="cc-by-nc-nd-4.0")
    keep, reason = hf_meta.should_track(info, min_params=3.0)
    assert keep is False
    assert reason == "license"


def test_should_track_accepts_real_model():
    info = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit")
    keep, reason = hf_meta.should_track(info, min_params=3.0)
    assert keep is True
    assert reason is None


def test_candidate_from_repo_shape():
    info = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit",
                    ctx=131072, created_at="2026-06-16T07:39:20+00:00",
                    downloads=42)
    c = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"])

    assert c["name"] == "GLM-5.2"
    assert c["hf_repo"] == "zai-org/GLM-5.2"
    assert c["developer"] == "zai-org"
    assert c["release_date"] == date(2026, 6, 16)
    assert c["params_total_b"] == 753.3
    assert c["params_active_b"] == 753.3
    assert c["context_window"] == 131072
    assert c["license"] == "mit"
    assert c["commercial_use"] is True
    assert c["discovered_via"] == ["org-sweep"]
    assert "arena_rank" not in c


def test_candidate_from_repo_carries_arena_rank():
    info = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit")
    c = hf_meta.candidate_from_repo(info, discovered_via=["arena"], arena_rank=10)
    assert c["arena_rank"] == 10
    assert c["discovered_via"] == ["arena"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hf_meta.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hf_meta'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/hf_meta.py`:

```python
#!/usr/bin/env python3
"""
Shared Hugging Face metadata helpers.

Both discovery sources (the org sweep in discover.py and arena-resolved repos
in pull_arena.py) build candidate rows through this module, so a candidate has
exactly one construction path and one set of filter rules.
"""
import re
from datetime import date, datetime

# License tags we accept (HF's vocabulary, NOT clean SPDX). Named open-weight
# licenses are included on purpose so we don't miss Llama/Gemma/Qwen.
ACCEPTED_LICENSES = {
    "apache-2.0", "mit", "bsd-3-clause",
    "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3", "llama4",
    "gemma", "qwen", "deepseek",
    "cc-by-4.0", "cc-by-nc-4.0", "other",
}

# license tag -> our commercial_use guess (ALWAYS re-checked by a human)
COMMERCIAL_GUESS = {
    "apache-2.0": True, "mit": True, "bsd-3-clause": True,
    "cc-by-4.0": True,
    "cc-by-nc-4.0": False,
    "llama2": "conditional", "llama3": "conditional", "llama3.1": "conditional",
    "llama3.2": "conditional", "llama3.3": "conditional", "llama4": "conditional",
    "gemma": "conditional", "qwen": "conditional", "deepseek": True,
    "other": "conditional",
}

# Substrings in a repo id that mark something we don't track. Covers
# quantizations, speculative-decoding drafts, and non-generative heads.
#   *-FP8 / *-MXFP8 / *-NVFP4 / *-w4a16   -> quantizations
#   eagle3_* / dflash_* / draft           -> speculative decoding artifacts
#   *-Embed / reranker                    -> not text-generation models
EXCLUDE_PATTERNS = re.compile(
    r"(gguf|awq|gptq|-int4|-int8|-fp8|mxfp8|nvfp4|w4a16|w8a8|-qat-|-bnb|-mlx|"
    r"-onnx|lora|adapter|draft|eagle3|dflash|-embed|reranker|"
    r"-4bit|-8bit|quantized|merge)",
    re.IGNORECASE,
)

CTX_KEYS = ("max_position_embeddings", "max_sequence_length", "n_positions")
EXPAND = ["safetensors", "cardData", "config", "downloads",
          "createdAt", "lastModified", "gated", "tags", "library_name"]


def is_derivative(repo_id):
    """True if the repo id marks a quant/adapter/merge/non-generative variant."""
    return bool(EXCLUDE_PATTERNS.search(repo_id))


def license_of(info):
    cd = getattr(info, "card_data", None) or {}
    lic = cd.get("license") if isinstance(cd, dict) else getattr(cd, "license", None)
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    return lic


def context_of(info):
    cfg = getattr(info, "config", None) or {}
    if isinstance(cfg, dict):
        for k in CTX_KEYS:
            if isinstance(cfg.get(k), int):
                return cfg[k]
        # some configs nest under "text_config" / "llm_config"
        for sub in ("text_config", "llm_config"):
            inner = cfg.get(sub) or {}
            for k in CTX_KEYS:
                if isinstance(inner.get(k), int):
                    return inner[k]
    return None


def params_b_of(info):
    st = getattr(info, "safetensors", None)
    if st is None:
        return None
    total = st.get("total") if isinstance(st, dict) else getattr(st, "total", None)
    if isinstance(total, (int, float)) and total > 0:
        return round(total / 1e9, 1)
    return None


def to_date(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def should_track(info, min_params):
    """Return (keep, skip_reason). skip_reason is None when keep is True."""
    if is_derivative(info.id):
        return False, "derivative"
    params = params_b_of(info)
    if params is None:
        return False, "no_params"
    if params < min_params:
        return False, "small"
    if license_of(info) not in ACCEPTED_LICENSES:
        return False, "license"
    return True, None


def candidate_from_repo(info, discovered_via, arena_rank=None):
    """Build a candidates.yaml row from an HF ModelInfo.

    Caller is responsible for having run should_track() first.
    """
    repo = info.id
    author = repo.split("/")[0]
    params = params_b_of(info)
    lic = license_of(info)

    candidate = {
        "name": repo.split("/")[-1],
        "hf_repo": repo,
        "developer": author,
        "release_date": to_date(getattr(info, "created_at", None)) or date.today(),
        "params_total_b": params,
        "params_active_b": params,   # TODO: set active params for MoE by hand
        "architecture": "dense",     # TODO: mark 'moe' if applicable
        "context_window": context_of(info) or 0,   # 0 => fill during review
        "modality": "text",
        "license": lic,
        "commercial_use": COMMERCIAL_GUESS.get(lic, "conditional"),
        "license_notes": "AUTO-DISCOVERED — verify license terms.",
        "benchmark": {"name": "MMLU", "score": None,
                      "source": "TODO: fill from a leaderboard"},
        "weights_url": f"https://huggingface.co/{repo}",
        "downloads": getattr(info, "downloads", None),
        "discovered_via": list(discovered_via),
        "notes": "Auto-discovered candidate; review before merging into models.yaml.",
    }
    if arena_rank is not None:
        candidate["arena_rank"] = arena_rank
    return candidate
```

> Note: the `TODO:` strings above are **data values written into `candidates.yaml`** for human reviewers to fill in — they are the existing repo convention, not plan placeholders. Keep them verbatim.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hf_meta.py -v`
Expected: PASS — 20 passed (14 parametrized + 6 named)

- [ ] **Step 5: Commit**

```bash
git add scripts/hf_meta.py tests/test_hf_meta.py
git commit -m "feat: add shared hf_meta candidate construction and filters"
```

---

### Task 2: Arena name normalization and match scoring

**Files:**
- Modify: `scripts/pull_arena.py` (replace `KEYWORD_MAP` at :60-86 and `derive_org` at :104-111)
- Create: `tests/test_arena_names.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `KEYWORD_MAP: dict[str, str]` — keyword → org name. **The `open_weight` boolean is removed.**
  - `ORG_DISPLAY_ALIASES: set[str]` — lowercase org display names to strip from leaderboard labels
  - `derive_org(name: str) -> tuple[str | None, str | None]` — returns `(org, matched_keyword)`, a **2-tuple** (was a 3-tuple)
  - `normalize_model_name(display: str) -> str`
  - `slug(text: str) -> str`
  - `score_match(query: str, repo_id: str) -> str` — one of `"high"`, `"medium"`, `"low"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_arena_names.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pull_arena


@pytest.mark.parametrize("display,expected", [
    ("Anthropic Claude Fable 5 (High) Anthropic · Proprietary",
     "Anthropic Claude Fable 5"),
    ("GLM 5.2 (Max) Z.ai · MIT · SiliconFlow", "GLM 5.2"),
    ("GLM 5.1 Z.ai · MIT · SiliconFlow", "GLM 5.1"),
    ("Kimi K3 Moonshot · Proprietary", "Kimi K3"),
    ("Kimi K2.7 Code Moonshot · Modified MIT", "Kimi K2.7 Code"),
    ("DeepSeek V4 Pro DeepSeek · MIT", "DeepSeek V4 Pro"),
    ("Minimax M3 MiniMax · MiniMax Community License", "Minimax M3"),
    ("Nemotron 3 Ultra Nvidia · OpenMDW-1.1", "Nemotron 3 Ultra"),
    ("Gemma 4 31B Google · Apache 2.0", "Gemma 4 31B"),
    ("Qwen3.7 Max Alibaba · Proprietary", "Qwen3.7 Max"),
    ("Meta Muse Spark 1.1 Meta · Proprietary", "Meta Muse Spark 1.1"),
    ("Tencent Hy3 Tencent · Apache 2.0", "Tencent Hy3"),
    ("Mimo V2.5 Pro Xiaomi · MIT", "Mimo V2.5 Pro"),
    ("Thinking Machines Inkling Thinky · Apache 2.0",
     "Thinking Machines Inkling"),
])
def test_normalize_model_name(display, expected):
    assert pull_arena.normalize_model_name(display) == expected


def test_derive_org_returns_two_tuple():
    org, kw = pull_arena.derive_org("Kimi K3")
    assert org == "Moonshot AI"
    assert kw == "kimi"


def test_derive_org_unknown():
    org, kw = pull_arena.derive_org("Some Unheard Of Model")
    assert org is None
    assert kw is None


@pytest.mark.parametrize("query,repo_id", [
    ("GLM 5.2", "zai-org/GLM-5.2"),
    ("Kimi K2.7 Code", "moonshotai/Kimi-K2.7-Code"),
    ("Minimax M3", "MiniMaxAI/MiniMax-M3"),
    ("Gemma 4 31B", "google/gemma-4-31b-it"),
    ("Kimi K2.6", "moonshotai/Kimi-K2.6"),
])
def test_score_match_high(query, repo_id):
    assert pull_arena.score_match(query, repo_id) == "high"


def test_score_match_medium_on_prefix():
    assert pull_arena.score_match("DeepSeek V4 Pro", "deepseek-ai/DeepSeek-V4") == "medium"


def test_score_match_low_on_mismatch():
    assert pull_arena.score_match("GLM 5.2", "meta-llama/Llama-3.1-8B") == "low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_arena_names.py -v`
Expected: FAIL — `AttributeError: module 'pull_arena' has no attribute 'normalize_model_name'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/pull_arena.py`, replace the `KEYWORD_MAP` block (currently lines 60-86) with:

```python
# keyword found in a model's display name -> canonical org. This is ONLY a
# search hint for HF lookup. It deliberately makes NO claim about whether the
# model is open-weight — that is decided by whether a weights repo resolves.
KEYWORD_MAP = {
    "claude": "Anthropic", "anthropic": "Anthropic",
    "gpt": "OpenAI", "openai": "OpenAI",
    "o1": "OpenAI", "o3": "OpenAI", "o4": "OpenAI",
    "gemini": "Google", "gemma": "Google",
    "llama": "Meta", "muse": "Meta", "meta": "Meta",
    "qwen": "Alibaba", "qwq": "Alibaba",
    "deepseek": "DeepSeek",
    "kimi": "Moonshot AI", "moonshot": "Moonshot AI",
    "mistral": "Mistral AI", "mixtral": "Mistral AI",
    "magistral": "Mistral AI", "devstral": "Mistral AI",
    "codestral": "Mistral AI",
    "grok": "xAI",
    "phi": "Microsoft",
    "command": "Cohere", "cohere": "Cohere", "aya": "Cohere",
    "glm": "Zhipu AI", "zhipu": "Zhipu AI",
    "yi": "01.AI",
    "nemotron": "NVIDIA",
    "falcon": "TII",
    "granite": "IBM",
    "olmo": "Ai2", "allenai": "Ai2",
    "minimax": "MiniMax",
    "ernie": "Baidu",
    "hunyuan": "Tencent", "hy3": "Tencent",
    "mimo": "Xiaomi",
    "inkling": "Thinking Machines",
}

# Org display names as they appear appended to leaderboard labels, e.g.
# "GLM 5.2 (Max) Z.ai · MIT". Stripped during normalization.
ORG_DISPLAY_ALIASES = {
    "anthropic", "openai", "google", "meta", "alibaba", "deepseek",
    "moonshot", "z.ai", "zai", "minimax", "nvidia", "xai", "microsoft",
    "cohere", "mistral", "tencent", "xiaomi", "thinky", "ibm", "baidu",
    "ai2", "01.ai", "tii", "zhipu",
}

# Repo-name suffixes that mark an instruction-tuned variant of the same model.
# Stripped before comparing an arena name to a repo name.
_VARIANT_SUFFIXES = ("instruct", "it", "chat", "base")
```

Replace `derive_org` (currently lines 104-111) and add the new helpers:

```python
def derive_org(name):
    """Return (org, matched_keyword) from a model display name.

    Makes no open-weight claim — that is decided by repo resolution.
    """
    low = name.lower()
    for kw, org in KEYWORD_MAP.items():
        if re.search(rf"\b{re.escape(kw)}", low):
            return org, kw
    return None, None


def normalize_model_name(display):
    """Strip leaderboard chrome from a display label.

    "GLM 5.2 (Max) Z.ai · MIT · SiliconFlow" -> "GLM 5.2"

    Three steps: drop everything from the first "·" separator, drop
    parenthetical effort tags like "(High)"/"(Max)", then drop a single
    trailing org display name.
    """
    name = display.split("·")[0]
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    tokens = name.split(" ")
    if len(tokens) > 1 and tokens[-1].lower() in ORG_DISPLAY_ALIASES:
        tokens = tokens[:-1]
    return " ".join(tokens).strip()


def slug(text):
    """Lowercase and strip every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def score_match(query, repo_id):
    """Rate how well an arena name matches an HF repo id: high/medium/low."""
    tail = repo_id.split("/")[-1]
    for suffix in _VARIANT_SUFFIXES:
        tail = re.sub(rf"[-_]{suffix}$", "", tail, flags=re.IGNORECASE)

    q, r = slug(query), slug(tail)
    if not q or not r:
        return "low"
    if q == r:
        return "high"
    if q.startswith(r) or r.startswith(q):
        return "medium"
    return "low"
```

Also update `looks_like_model` (line 122-123), which unpacked the old 3-tuple:

```python
def looks_like_model(text):
    return derive_org(text)[1] is not None or bool(re.search(r"[A-Za-z]{3,}", text))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_arena_names.py -v`
Expected: PASS — 23 passed (14 normalization + 2 derive_org + 5 high-match + 2 scoring)

- [ ] **Step 5: Commit**

```bash
git add scripts/pull_arena.py tests/test_arena_names.py
git commit -m "feat: arena name normalization and HF match scoring"
```

---

### Task 3: Leaderboard parsing against a fixture

**Files:**
- Modify: `scripts/pull_arena.py` (`ROW_SCHEMA` at :90-102, `parse_leaderboard` at :126-219)
- Create: `tests/fixtures/arena_sample.html`
- Create: `tests/test_arena_parse.py`

**Interfaces:**
- Consumes: `derive_org` (2-tuple form) from Task 2.
- Produces: `parse_leaderboard(html: str) -> list[dict]`. Each row dict has keys `rank`, `model`, `org`, `matched_keyword`, `net_improvement_pct`, `net_improvement_ci`, `arena_license_label`, `raw`, `by_header`. **`open_weight` is no longer set here** — Task 4 sets it from resolution.

- [ ] **Step 1: Write the failing test**

Create `tests/fixtures/arena_sample.html`. This mirrors the real page's structure (server-rendered `<table>`, the nine columns observed on 2026-07-21):

```html
<html><body>
<table>
  <thead>
    <tr>
      <th>Rank</th><th>Model</th><th>Net Improvement</th>
      <th>Confirmed Success</th><th>Praise vs Complaint</th>
      <th>Steerability</th><th>Bash Recovery</th>
      <th>Tool Hallucination</th><th>Sessions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Anthropic Claude Fable 5 (High) Anthropic · Proprietary</td>
      <td>12.72 % ±2.00%</td><td>10.67 % ±3.84%</td><td>23.94 % ±7.42%</td>
      <td>14.62 % ±3.80%</td><td>12.97 % ±1.30%</td><td>1.39 % ±0.17%</td>
      <td>23,549</td>
    </tr>
    <tr>
      <td>10</td>
      <td>GLM 5.2 (Max) Z.ai · MIT · SiliconFlow</td>
      <td>6.50 % ±1.20%</td><td>5.10 % ±2.00%</td><td>12.00 % ±4.00%</td>
      <td>7.00 % ±2.10%</td><td>6.20 % ±1.10%</td><td>2.10 % ±0.30%</td>
      <td>9,120</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Meta Muse Spark 1.1 Meta · Proprietary</td>
      <td>0.67 % ±0.90%</td><td>0.40 % ±1.10%</td><td>2.00 % ±3.00%</td>
      <td>1.10 % ±1.40%</td><td>0.80 % ±0.60%</td><td>3.40 % ±0.50%</td>
      <td>4,201</td>
    </tr>
    <tr>
      <td>37</td>
      <td>Gemma 4 31B Google · Apache 2.0</td>
      <td>14.51 % ±3.10%</td><td>11.20 % ±4.20%</td><td>25.00 % ±8.00%</td>
      <td>15.30 % ±4.00%</td><td>13.10 % ±1.90%</td><td>1.80 % ±0.40%</td>
      <td>2,880</td>
    </tr>
  </tbody>
</table>
</body></html>
```

Create `tests/test_arena_parse.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pull_arena

FIXTURE = Path(__file__).parent / "fixtures" / "arena_sample.html"


@pytest.fixture
def rows():
    return pull_arena.parse_leaderboard(FIXTURE.read_text())


def test_parses_all_rows(rows):
    assert len(rows) == 4


def test_ranks_in_order(rows):
    assert [r["rank"] for r in rows] == [1, 10, 18, 37]


def test_extracts_score_and_ci(rows):
    glm = next(r for r in rows if r["rank"] == 10)
    assert glm["net_improvement_pct"] == 6.50
    assert glm["net_improvement_ci"] == 1.20


def test_maps_org_from_keyword(rows):
    glm = next(r for r in rows if r["rank"] == 10)
    assert glm["org"] == "Zhipu AI"
    assert glm["matched_keyword"] == "glm"


def test_does_not_set_open_weight(rows):
    """open_weight is decided by repo resolution (Task 4), never by parsing."""
    assert all("open_weight" not in r for r in rows)


def test_keeps_raw_cells(rows):
    first = rows[0]
    assert first["raw"][0] == "1"
    assert "Claude Fable 5" in first["raw"][1]


def test_by_header_populated(rows):
    first = rows[0]
    assert first["by_header"]["Sessions"] == "23,549"


def test_empty_html_returns_no_rows():
    assert pull_arena.parse_leaderboard("<html><body></body></html>") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_arena_parse.py -v`
Expected: FAIL — `test_does_not_set_open_weight` fails (key still present) and `test_maps_org_from_keyword` errors on the 3-tuple unpack in `parse_leaderboard`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/pull_arena.py`, update `ROW_SCHEMA` (lines 90-102) to drop `open_weight`:

```python
ROW_SCHEMA = {
    "type": "object",
    "required": ["rank", "model", "org"],
    "properties": {
        "rank": {"type": "integer", "minimum": 1},
        "model": {"type": "string", "minLength": 1},
        "org": {"type": ["string", "null"]},
        "net_improvement_pct": {"type": ["number", "null"]},
        "net_improvement_ci": {"type": ["number", "null"]},
    },
}
```

Inside `parse_leaderboard`, change the model-detection line (currently line 174) from `if derive_org(c)[2] is not None:` to index `[1]`:

```python
        model = None
        for c in cells:
            if derive_org(c)[1] is not None:
                model = c
                break
```

Change the org unpack (currently line 189) and the appended row (lines 197-208):

```python
        org, matched = derive_org(model)

        # capture arena's own license label if any cell says so (informational)
        arena_label = next((c for c in cells
                            if re.search(r"open|proprietary|closed|source", c, re.I)), None)

        row = dict(zip(headers, cells)) if headers and len(headers) == len(cells) else {}
        rows.append({
            "rank": rank,
            "model": model,
            "org": org,
            "matched_keyword": matched,
            "net_improvement_pct": pct,
            "net_improvement_ci": ci,
            "arena_license_label": arena_label,
            "raw": cells,
            "by_header": row or None,
        })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_arena_parse.py -v`
Expected: PASS — 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/pull_arena.py tests/fixtures/arena_sample.html tests/test_arena_parse.py
git commit -m "feat: parse leaderboard without asserting open-weight status"
```

---

### Task 4: Resolve arena names to HF repos

**Files:**
- Modify: `scripts/pull_arena.py` (add `resolve_row`/`resolve_all`, rewrite `main` at :236-283, update module docstring at :2-30)
- Create: `tests/test_arena_resolve.py`

**Interfaces:**
- Consumes: `normalize_model_name`, `derive_org`, `score_match` (Task 2); `parse_leaderboard` (Task 3).
- Produces:
  - `HF_AUTHOR_HINTS: dict[str, str]` — org name → HF author namespace
  - `resolve_row(row: dict, search_fn) -> dict` — mutates and returns `row`, adding `resolved_repo: str | None`, `resolution_confidence: "high"|"medium"|"low"|None`, `open_weight: bool`, `needs_hf_repo: bool`
  - `resolve_all(rows: list[dict], search_fn) -> list[dict]`
  - `search_fn` contract: `search_fn(query: str, author: str | None) -> list[str]` returning candidate repo ids, best first.

- [ ] **Step 1: Write the failing test**

Create `tests/test_arena_resolve.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pull_arena


def fake_search(index):
    """Build a search_fn backed by a dict of {author_or_None: [repo_ids]}."""
    def _search(query, author):
        return index.get(author, index.get(None, []))
    return _search


def test_resolves_exact_match_high_confidence():
    row = {"model": "GLM 5.2 (Max) Z.ai · MIT · SiliconFlow",
           "org": "Zhipu AI", "matched_keyword": "glm"}
    search = fake_search({"zai-org": ["zai-org/GLM-5.2", "zai-org/GLM-5.1"]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "zai-org/GLM-5.2"
    assert out["resolution_confidence"] == "high"
    assert out["open_weight"] is True
    assert out["needs_hf_repo"] is False


def test_unresolvable_model_is_not_open_weight():
    """Meta Muse Spark is proprietary — no weights repo exists."""
    row = {"model": "Meta Muse Spark 1.1 Meta · Proprietary",
           "org": "Meta", "matched_keyword": "muse"}
    search = fake_search({"meta-llama": []})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] is None
    assert out["open_weight"] is False
    assert out["resolution_confidence"] is None


def test_low_confidence_flags_for_human():
    row = {"model": "Kimi K3 Moonshot · Proprietary",
           "org": "Moonshot AI", "matched_keyword": "kimi"}
    search = fake_search({"moonshotai": ["moonshotai/Kimi-Linear-48B-A3B-Base"]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolution_confidence"] == "low"
    assert out["needs_hf_repo"] is True
    assert out["open_weight"] is False


def test_medium_confidence_resolves_and_flags():
    row = {"model": "DeepSeek V4 Pro DeepSeek · MIT",
           "org": "DeepSeek", "matched_keyword": "deepseek"}
    search = fake_search({"deepseek-ai": ["deepseek-ai/DeepSeek-V4"]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "deepseek-ai/DeepSeek-V4"
    assert out["resolution_confidence"] == "medium"
    assert out["open_weight"] is True
    assert out["needs_hf_repo"] is True


def test_prefers_highest_scoring_candidate():
    row = {"model": "Kimi K2.6 Moonshot · Modified MIT",
           "org": "Moonshot AI", "matched_keyword": "kimi"}
    search = fake_search({"moonshotai": [
        "moonshotai/Kimi-K2-Thinking",   # low
        "moonshotai/Kimi-K2.6",          # high, listed second
    ]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "moonshotai/Kimi-K2.6"
    assert out["resolution_confidence"] == "high"


def test_unknown_org_searches_without_author():
    row = {"model": "Tencent Hy3 Tencent · Apache 2.0",
           "org": "Tencent", "matched_keyword": "hy3"}
    search = fake_search({None: ["tencent/Tencent-Hy3"]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "tencent/Tencent-Hy3"
    assert out["open_weight"] is True


def test_search_failure_is_not_fatal():
    def boom(query, author):
        raise RuntimeError("HF rate limited")

    row = {"model": "GLM 5.2 Z.ai · MIT", "org": "Zhipu AI",
           "matched_keyword": "glm"}

    out = pull_arena.resolve_row(row, boom)

    assert out["resolved_repo"] is None
    assert out["open_weight"] is False
    assert out["needs_hf_repo"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_arena_resolve.py -v`
Expected: FAIL — `AttributeError: module 'pull_arena' has no attribute 'resolve_row'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/pull_arena.py`, after `score_match`:

```python
# Canonical org name -> HF author namespace, used to scope the repo search.
HF_AUTHOR_HINTS = {
    "Google": "google", "Meta": "meta-llama", "Alibaba": "Qwen",
    "DeepSeek": "deepseek-ai", "Moonshot AI": "moonshotai",
    "Zhipu AI": "zai-org", "MiniMax": "MiniMaxAI", "NVIDIA": "nvidia",
    "Microsoft": "microsoft", "Cohere": "CohereLabs", "Mistral AI": "mistralai",
    "01.AI": "01-ai", "TII": "tiiuae", "IBM": "ibm-granite", "Ai2": "allenai",
    "Baidu": "baidu", "Xiaomi": "XiaomiMiMo",
    "Thinking Machines": "thinkingmachines",
}


def resolve_row(row, search_fn):
    """Resolve an arena row to an HF repo. Mutates and returns the row.

    Open-weight status is decided HERE, and only here: a model is open-weight
    if and only if public weights were found. arena's own license label and
    KEYWORD_MAP are never used to make that call.
    """
    query = normalize_model_name(row["model"])
    author = HF_AUTHOR_HINTS.get(row.get("org"))

    try:
        repo_ids = search_fn(query, author) or []
    except Exception as exc:      # rate limit, network, API change
        print(f"  ! search failed for {query!r}: {exc}")
        repo_ids = []

    best, best_conf = None, "low"
    _ORDER = {"high": 3, "medium": 2, "low": 1}
    for repo_id in repo_ids:
        conf = score_match(query, repo_id)
        if best is None or _ORDER[conf] > _ORDER[best_conf]:
            best, best_conf = repo_id, conf
        if best_conf == "high":
            break

    if best is None or best_conf == "low":
        row["resolved_repo"] = None
        row["resolution_confidence"] = best_conf if best is not None else None
        row["open_weight"] = False
        row["needs_hf_repo"] = True
    else:
        row["resolved_repo"] = best
        row["resolution_confidence"] = best_conf
        row["open_weight"] = True
        row["needs_hf_repo"] = best_conf != "high"
    return row


def resolve_all(rows, search_fn):
    for row in rows:
        resolve_row(row, search_fn)
    return rows


def hf_search_fn(api):
    """Adapt HfApi into the search_fn contract: (query, author) -> [repo_id]."""
    def _search(query, author):
        kwargs = {"search": query, "limit": 10}
        if author:
            kwargs["author"] = author
        return [m.id for m in api.list_models(**kwargs)]
    return _search
```

Replace `main` (currently lines 236-283) with:

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--open-weight-only", action="store_true",
                    help="write only rows that resolved to a weights repo")
    ap.add_argument("--html", help="parse a local HTML file instead of fetching "
                                   "(useful for testing/offline)")
    ap.add_argument("--no-resolve", action="store_true",
                    help="skip HF resolution (parse only)")
    args = ap.parse_args()

    if args.html:
        html = Path(args.html).read_text()
    else:
        print(f"Fetching {args.url}")
        resp = requests.get(args.url, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0 (owlt-arena/1.0)"})
        resp.raise_for_status()
        html = resp.text

    rows = parse_leaderboard(html)
    if not rows:
        sys.exit("No rows parsed — the page markup may have changed, or the table "
                 "loaded via JS. Inspect the HTML and adjust parse_leaderboard().")

    validate_rows(rows)

    if not args.no_resolve:
        from huggingface_hub import HfApi
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        print(f"\nResolving {len(rows)} models against Hugging Face...")
        resolve_all(rows, hf_search_fn(HfApi(token=token)))

    ow = [r for r in rows if r.get("open_weight")]
    print(f"\nParsed {len(rows)} models; {len(ow)} resolved to a weights repo.")
    for r in ow:
        s = f"{r['net_improvement_pct']}%" if r["net_improvement_pct"] is not None else "?"
        flag = "  [VERIFY]" if r.get("needs_hf_repo") else ""
        print(f"  #{r['rank']:>2}  {r['model'][:45]:<45} -> {r['resolved_repo']} ({s}){flag}")

    new_orgs = sorted({r["org"] for r in ow
                       if r.get("org") and r["org"] not in HF_AUTHOR_HINTS})
    if new_orgs:
        print(f"\nOrgs seen on the leaderboard with no HF namespace mapping "
              f"(add to HF_AUTHOR_HINTS / ORG_ALLOWLIST): {', '.join(new_orgs)}")

    out_rows = ow if args.open_weight_only else rows
    header = ("# AUTO-SCRAPED from arena.ai Agent Arena by scripts/pull_arena.py\n"
              "# open_weight means: a public weights repo was FOUND on Hugging Face.\n"
              "# It is not arena's license label and not an org guess.\n"
              "# needs_hf_repo=true means the match was inexact — verify by hand.\n")
    OUT.write_text(header + yaml.safe_dump(
        {"arena_agent": out_rows, "new_orgs": new_orgs},
        sort_keys=False, allow_unicode=True, width=100))
    print(f"\nWrote {len(out_rows)} rows to {OUT.name}")
```

Add `import os` to the imports at the top of the file (after `import argparse`).

Replace the module docstring (lines 2-30) with:

```python
"""
Scrape the Arena Intelligence "Agent Arena" leaderboard (https://arena.ai)
and resolve each ranked model to a Hugging Face weights repo.

WHAT THIS IS FOR:
    arena ranks the models people actually use. models.yaml is populated from
    Hugging Face. This script is the join between them: it turns a leaderboard
    display name into an HF repo id, so discover.py can prioritize its review
    queue by real-world usage rather than by upload recency.

HOW OPEN-WEIGHT IS DECIDED:
    A model is open-weight if and only if a public weights repo resolves for
    it on Hugging Face. We do NOT trust arena's license label, and we do NOT
    infer from the org (an earlier version did, and flagged proprietary models
    like "Meta Muse Spark · Proprietary" as open because the name said "Meta").
    KEYWORD_MAP survives only as a search hint.

HOW IT PARSES (resilient by design):
    The leaderboard is a server-rendered <table>, so requests + BeautifulSoup
    is enough (no headless browser). We do NOT pin to CSS class names (they
    change). Instead we read the <table>, then extract each row heuristically:
      - rank  = first pure-integer cell (fallback: row position)
      - model = the cell containing a known vendor/product keyword
      - score = first cell matching  NN.NN% ± N.NN%
    Every raw cell is kept under `raw` so nothing is silently lost.

Output: arena_agent_rankings.yaml

Usage:
  pip install -r requirements.txt
  python scripts/pull_arena.py                     # scrape + resolve
  python scripts/pull_arena.py --no-resolve        # parse only, no HF calls
  python scripts/pull_arena.py --open-weight-only  # only resolved models
  python scripts/pull_arena.py --html page.html    # offline, from a saved page

Note: arena.ai is unreachable from some sandboxes; run on your machine / CI.
There is no official API — this is scraping, so check arena.ai's ToS before
republishing their numbers.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_arena_resolve.py -v`
Expected: PASS — 7 passed

Then confirm nothing regressed:

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS — 58 passed (20 hf_meta + 23 arena_names + 8 arena_parse + 7 arena_resolve)

- [ ] **Step 5: Commit**

```bash
git add scripts/pull_arena.py tests/test_arena_resolve.py
git commit -m "feat: resolve arena models to HF repos, derive open-weight from resolution"
```

---

### Task 5: Rewrite `discover.py` as an org sweep

**Files:**
- Modify: `scripts/discover.py` — delete lines 105-135 (`_ORG_CACHE`, `get_org_overview`) and `is_organization`; delete `build_query` (:202-218); rewrite `main` (:221+); delete the now-unused helpers moved to `hf_meta`
- Create: `tests/test_discover_sweep.py`

**Interfaces:**
- Consumes: `hf_meta.should_track`, `hf_meta.candidate_from_repo`, `hf_meta.EXPAND` (Task 1).
- Produces:
  - `ORG_ALLOWLIST: set[str]` (unchanged membership)
  - `sweep_orgs(api, orgs, min_params, known) -> tuple[list[dict], dict]` — returns `(candidates, skip_counts)`; `skip_counts` keys are `known`, `derivative`, `small`, `license`, `no_params`, `org_error`
  - `existing_repos() -> set[str]` (unchanged behavior)

- [ ] **Step 1: Write the failing test**

Create `tests/test_discover_sweep.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover
from test_hf_meta import FakeInfo


class FakeApi:
    """Returns a canned model list per author; raises for authors in `errors`."""

    def __init__(self, by_author, errors=()):
        self.by_author = by_author
        self.errors = set(errors)
        self.calls = []

    def list_models(self, **kwargs):
        author = kwargs.get("author")
        self.calls.append(author)
        if author in self.errors:
            raise RuntimeError("HF 429")
        return list(self.by_author.get(author, []))


def test_sweep_queries_each_org_once():
    api = FakeApi({"zai-org": [], "moonshotai": []})
    discover.sweep_orgs(api, ["zai-org", "moonshotai"], 3.0, set())
    assert sorted(api.calls) == ["moonshotai", "zai-org"]


def test_sweep_collects_real_models():
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(api, ["zai-org"], 3.0, set())

    assert len(candidates) == 1
    assert candidates[0]["hf_repo"] == "zai-org/GLM-5.2"
    assert candidates[0]["discovered_via"] == ["org-sweep"]


def test_sweep_drops_quantizations():
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
        FakeInfo("zai-org/GLM-5.2-FP8", total=753_400_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(api, ["zai-org"], 3.0, set())

    assert [c["hf_repo"] for c in candidates] == ["zai-org/GLM-5.2"]
    assert skips["derivative"] == 1


def test_sweep_skips_already_known_repos():
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(
        api, ["zai-org"], 3.0, {"zai-org/glm-5.2"})

    assert candidates == []
    assert skips["known"] == 1


def test_one_failing_org_does_not_abort_sweep():
    api = FakeApi(
        by_author={"moonshotai": [
            FakeInfo("moonshotai/Kimi-K2.6", total=1_058_600_000_000, license="mit"),
        ]},
        errors=["zai-org"],
    )
    candidates, skips = discover.sweep_orgs(
        api, ["zai-org", "moonshotai"], 3.0, set())

    assert len(candidates) == 1
    assert skips["org_error"] == 1


def test_dead_code_is_gone():
    """The follower-probe machinery existed only for the unbounded query."""
    for name in ("get_org_overview", "is_organization", "build_query",
                 "AUTHOR_BLOCKLIST"):
        assert not hasattr(discover, name), f"{name} should have been deleted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_discover_sweep.py -v`
Expected: FAIL — `AttributeError: module 'discover' has no attribute 'sweep_orgs'`, and `test_dead_code_is_gone` fails on `get_org_overview`.

- [ ] **Step 3: Write minimal implementation**

Rewrite `scripts/discover.py` entirely:

```python
#!/usr/bin/env python3
"""
Discover NEW open-weight LLMs and stage them as candidates for review. This
does NOT touch models.yaml directly — it writes candidates.yaml so a human
approves them (via PR) before they land.

WHY AN ORG SWEEP, NOT A GLOBAL SCAN:
    This script used to sort ALL of Hugging Face by created_at and take the
    newest N. That population is dominated by finetunes and quantizations, so
    a frontier release essentially never appears in it. A run on 2026-07-21
    scanned 300 models and produced zero candidates (110 derivative, 189
    non-org, 1 blocked — every single one filtered).

    Now we issue one list_models(author=org) call per allowlisted org. That
    surfaced GLM-5.2, Kimi-K2.7, MiniMax-M3 and Gemma 4 immediately.

    Consequence: the org-follower probe that policed the unbounded query is
    gone. The allowlist IS the query.

WHERE NEW ORGS COME FROM:
    An allowlist only finds what it already knows. scripts/pull_arena.py
    surfaces orgs seen on the arena leaderboard that we have no mapping for
    (Tencent, Xiaomi, Thinking Machines as of 2026-07-21) and they are
    reported here for a human to add.

Usage:
  pip install -r requirements.txt
  python scripts/discover.py                  # org sweep (+ arena if present)
  python scripts/discover.py --min-params 3
  python scripts/discover.py --no-arena       # skip the arena merge
  HUGGINGFACE_TOKEN=hf_xxx python scripts/discover.py   # higher rate limits

NOTES / deliberate choices (the "don'ts"):
  - We do NOT exclude gated models — Llama & Gemma are gated; excluding them
    would drop the flagships. We keep them and flag for review.
  - safetensors.total is TOTAL params. For MoE we cannot infer active params
    from the API, so params_active_b is left equal to total and flagged TODO.
  - We do NOT trust card eval results as the benchmark column — left blank.
  - license tag is uploader-supplied; commercial_use is a *guess* to be checked.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("Install deps first:  pip install -r requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_meta

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
CANDIDATES = ROOT / "candidates.yaml"
ARENA = ROOT / "arena_agent_rankings.yaml"

# The orgs we sweep. This list IS the query — adding an org here is how the
# tracker gains coverage. pull_arena.py reports leaderboard orgs missing here.
ORG_ALLOWLIST = [
    "meta-llama", "Qwen", "deepseek-ai", "mistralai", "google", "microsoft",
    "CohereForAI", "CohereLabs", "ai21labs", "allenai", "nvidia", "01-ai",
    "tiiuae", "databricks", "HuggingFaceTB", "ibm-granite", "internlm",
    "THUDM", "zai-org", "moonshotai", "openai", "xai-org", "stabilityai",
    "MiniMaxAI", "XiaomiMiMo", "poolside", "thinkingmachines", "baidu",
]


def existing_repos():
    """Every hf_repo already tracked or already staged, lowercased."""
    repos = set()
    for path in (DATA, CANDIDATES):
        if not path.exists():
            continue
        doc = yaml.safe_load(path.read_text()) or {}
        for m in doc.get("models", []) or []:
            if m.get("hf_repo"):
                repos.add(m["hf_repo"].lower())
    return repos


def sweep_orgs(api, orgs, min_params, known):
    """One list_models call per org. Returns (candidates, skip_counts).

    A failure on one org is logged and skipped — it never aborts the sweep.
    """
    candidates = []
    skips = {"known": 0, "derivative": 0, "small": 0, "license": 0,
             "no_params": 0, "org_error": 0}
    seen = set(known)

    for org in orgs:
        try:
            models = api.list_models(
                author=org,
                pipeline_tag="text-generation",
                sort="created_at",
                limit=50,
                expand=hf_meta.EXPAND,
            )
        except Exception as exc:
            print(f"  ! {org}: {exc}")
            skips["org_error"] += 1
            continue

        for info in models:
            if info.id.lower() in seen:
                skips["known"] += 1
                continue
            keep, reason = hf_meta.should_track(info, min_params)
            if not keep:
                skips[reason] += 1
                continue
            candidates.append(
                hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"]))
            seen.add(info.id.lower())

    return candidates, skips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-params", type=float, default=3.0,
                    help="minimum total params in billions")
    ap.add_argument("--no-arena", action="store_true",
                    help="skip merging arena_agent_rankings.yaml")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    known = existing_repos()

    print(f"Sweeping {len(ORG_ALLOWLIST)} orgs (min {args.min_params}B params)")
    candidates, skips = sweep_orgs(api, ORG_ALLOWLIST, args.min_params, known)
    print(f"  org sweep found {len(candidates)} candidate(s); skipped: {skips}")

    candidates.sort(key=lambda c: c["release_date"], reverse=True)

    header = (
        "# AUTO-GENERATED candidate models from scripts/discover.py\n"
        "# Review each entry, fix the TODO fields (active params, architecture,\n"
        "# context, benchmark, commercial_use), then move approved rows into\n"
        "# models.yaml and delete them here.\n"
    )
    CANDIDATES.write_text(header + yaml.safe_dump({"models": candidates},
                                                  sort_keys=False,
                                                  allow_unicode=True, width=100))
    print(f"\nWrote {len(candidates)} candidate(s) to {CANDIDATES.name}")
    # exit 0 always; the Action decides whether the diff is non-empty


if __name__ == "__main__":
    main()
```

> The arena merge is deliberately absent here — `--no-arena` is parsed but unused until Task 6 wires it. Task 6 adds the merge in one reviewable change.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_discover_sweep.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/discover.py tests/test_discover_sweep.py
git commit -m "feat: replace global HF scan with org-scoped sweep"
```

---

### Task 6: Merge arena candidates into the queue

**Files:**
- Modify: `scripts/discover.py` (add `load_arena`, `merge_candidates`; wire into `main`)
- Create: `tests/test_discover_merge.py`

**Interfaces:**
- Consumes: `hf_meta.candidate_from_repo`, `hf_meta.should_track` (Task 1); `sweep_orgs` (Task 5); the `arena_agent_rankings.yaml` shape from Task 4 (`resolved_repo`, `rank`, `open_weight`).
- Produces:
  - `load_arena(path) -> tuple[list[dict], list[str]]` — `(resolved_rows, new_orgs)`; returns `([], [])` when the file is missing or unparseable
  - `arena_candidates(api, rows, min_params, known) -> list[dict]`
  - `merge_candidates(org_rows: list[dict], arena_rows: list[dict]) -> list[dict]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_discover_merge.py`:

```python
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover


def cand(repo, via, rank=None, released="2026-01-01"):
    y, m, d = (int(x) for x in released.split("-"))
    c = {"hf_repo": repo, "name": repo.split("/")[-1],
         "discovered_via": list(via), "release_date": date(y, m, d)}
    if rank is not None:
        c["arena_rank"] = rank
    return c


def test_merge_dedups_on_repo_case_insensitively():
    org_rows = [cand("zai-org/GLM-5.2", ["org-sweep"])]
    arena_rows = [cand("ZAI-ORG/glm-5.2", ["arena"], rank=10)]

    merged = discover.merge_candidates(org_rows, arena_rows)

    assert len(merged) == 1
    assert sorted(merged[0]["discovered_via"]) == ["arena", "org-sweep"]
    assert merged[0]["arena_rank"] == 10


def test_arena_ranked_models_sort_first():
    org_rows = [cand("org/New-Model", ["org-sweep"], released="2026-07-01")]
    arena_rows = [cand("zai-org/GLM-5.2", ["arena"], rank=10,
                       released="2026-06-16")]

    merged = discover.merge_candidates(org_rows, arena_rows)

    assert [c["hf_repo"] for c in merged] == [
        "zai-org/GLM-5.2", "org/New-Model"]


def test_ranked_models_sort_by_rank():
    arena_rows = [
        cand("a/Model-B", ["arena"], rank=27),
        cand("a/Model-A", ["arena"], rank=10),
    ]
    merged = discover.merge_candidates([], arena_rows)
    assert [c["arena_rank"] for c in merged] == [10, 27]


def test_unranked_models_sort_by_recency():
    org_rows = [
        cand("a/Older", ["org-sweep"], released="2026-01-01"),
        cand("a/Newer", ["org-sweep"], released="2026-07-01"),
    ]
    merged = discover.merge_candidates(org_rows, [])
    assert [c["name"] for c in merged] == ["Newer", "Older"]


def test_load_arena_missing_file_is_not_fatal(tmp_path):
    rows, new_orgs = discover.load_arena(tmp_path / "nope.yaml")
    assert rows == []
    assert new_orgs == []


def test_load_arena_malformed_file_is_not_fatal(tmp_path):
    bad = tmp_path / "arena.yaml"
    bad.write_text("{{{ not yaml at all")
    rows, new_orgs = discover.load_arena(bad)
    assert rows == []
    assert new_orgs == []


def test_load_arena_returns_only_resolved_rows(tmp_path):
    f = tmp_path / "arena.yaml"
    f.write_text(
        "arena_agent:\n"
        "- rank: 10\n"
        "  model: GLM 5.2\n"
        "  resolved_repo: zai-org/GLM-5.2\n"
        "  open_weight: true\n"
        "- rank: 18\n"
        "  model: Meta Muse Spark 1.1\n"
        "  resolved_repo: null\n"
        "  open_weight: false\n"
        "new_orgs: [Tencent, Xiaomi]\n"
    )
    rows, new_orgs = discover.load_arena(f)

    assert [r["resolved_repo"] for r in rows] == ["zai-org/GLM-5.2"]
    assert new_orgs == ["Tencent", "Xiaomi"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_discover_merge.py -v`
Expected: FAIL — `AttributeError: module 'discover' has no attribute 'merge_candidates'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/discover.py`, after `sweep_orgs`:

```python
def load_arena(path=ARENA):
    """Read arena_agent_rankings.yaml. Returns (resolved_rows, new_orgs).

    Arena is never load-bearing: a missing, empty, or malformed file yields
    ([], []) so the org sweep still produces candidates.
    """
    path = Path(path)
    if not path.exists():
        return [], []
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        print(f"  ! arena file unreadable ({exc}); continuing without it")
        return [], []

    rows = [r for r in (doc.get("arena_agent") or [])
            if r.get("resolved_repo")]
    return rows, list(doc.get("new_orgs") or [])


def arena_candidates(api, rows, min_params, known):
    """Build candidates from arena-resolved repos via the shared hf_meta path."""
    out = []
    for row in rows:
        repo = row["resolved_repo"]
        if repo.lower() in known:
            continue
        try:
            info = api.model_info(repo, expand=hf_meta.EXPAND)
        except Exception as exc:
            print(f"  ! {repo}: {exc}")
            continue
        keep, reason = hf_meta.should_track(info, min_params)
        if not keep:
            print(f"  - {repo} skipped ({reason})")
            continue
        out.append(hf_meta.candidate_from_repo(
            info, discovered_via=["arena"], arena_rank=row.get("rank")))
    return out


def merge_candidates(org_rows, arena_rows):
    """Dedup on lowercased hf_repo, then sort by arena rank, then recency.

    A model found by both sources keeps both tags and the arena rank, so the
    review queue leads with models people actually use.
    """
    by_repo = {}
    for c in list(org_rows) + list(arena_rows):
        key = c["hf_repo"].lower()
        if key not in by_repo:
            by_repo[key] = dict(c)
            continue
        merged = by_repo[key]
        merged["discovered_via"] = sorted(
            set(merged["discovered_via"]) | set(c["discovered_via"]))
        if c.get("arena_rank") is not None:
            merged["arena_rank"] = c["arena_rank"]

    # ranked first by rank ascending; unranked after, newest first
    return sorted(by_repo.values(),
                  key=lambda c: (c.get("arena_rank") is None,
                                 c.get("arena_rank") or 0,
                                 -c["release_date"].toordinal()))
```

No new imports are needed in `discover.py` for this task.

Then replace the body of `main` between the sweep and the write with:

```python
    print(f"Sweeping {len(ORG_ALLOWLIST)} orgs (min {args.min_params}B params)")
    org_rows, skips = sweep_orgs(api, ORG_ALLOWLIST, args.min_params, known)
    print(f"  org sweep found {len(org_rows)} candidate(s); skipped: {skips}")

    arena_rows, new_orgs = ([], [])
    if not args.no_arena:
        resolved, new_orgs = load_arena()
        if resolved:
            print(f"  arena contributed {len(resolved)} resolved repo(s)")
            arena_rows = arena_candidates(api, resolved, args.min_params, known)
        else:
            print("  no arena data (run scripts/pull_arena.py first)")

    candidates = merge_candidates(org_rows, arena_rows)

    if new_orgs:
        print(f"\nNEW ORGS seen on the leaderboard but not in ORG_ALLOWLIST: "
              f"{', '.join(new_orgs)}")
        print("Add them to ORG_ALLOWLIST in this file to widen coverage.")
```

And delete the now-redundant `candidates.sort(...)` line that followed the old sweep — `merge_candidates` owns ordering.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_discover_merge.py -v`
Expected: PASS — 7 passed

Run the full suite:

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS — 71 passed (58 from Task 4 + 6 sweep + 7 merge)

- [ ] **Step 5: Commit**

```bash
git add scripts/discover.py tests/test_discover_merge.py
git commit -m "feat: merge arena-resolved candidates and prioritize by rank"
```

---

### Task 7: Wire tests into CI and document the new fields

**Files:**
- Modify: `requirements.txt` (add `pytest`)
- Modify: `.github/workflows/validate.yml` (add pytest step after checkout/install)
- Modify: `.github/workflows/discover.yml` (run `pull_arena.py` before `discover.py`)
- Modify: `SCHEMA.md` (append a `candidates.yaml` section)

**Interfaces:**
- Consumes: the full test suite from Tasks 1-6.
- Produces: no new code interfaces.

- [ ] **Step 1: Verify the full suite passes locally first**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS — 71 passed

- [ ] **Step 2: Add pytest to requirements**

`requirements.txt` becomes:

```
PyYAML>=6.0
huggingface_hub>=0.25
requests>=2.31
beautifulsoup4>=4.12
jsonschema>=4.0
pytest>=8.0
```

- [ ] **Step 3: Add the test step to the validate workflow**

In `.github/workflows/validate.yml`, insert a step between `pip install -r requirements.txt` and `Validate models.yaml`:

```yaml
      - name: Run tests
        run: python -m pytest tests/ -v
```

- [ ] **Step 4: Run arena before discovery in the discover workflow**

In `.github/workflows/discover.yml`, replace the `Discover new models` step with:

```yaml
      - name: Pull arena leaderboard
        continue-on-error: true    # arena must never block discovery
        env:
          HUGGINGFACE_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
        run: python scripts/pull_arena.py
      - name: Discover new models
        env:
          HUGGINGFACE_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
        run: python scripts/discover.py --min-params 3
```

Also widen `add-paths` so the arena file travels with the PR:

```yaml
          add-paths: |
            candidates.yaml
            arena_agent_rankings.yaml
```

- [ ] **Step 5: Document the candidates-only fields**

Append to `SCHEMA.md`:

```markdown
## `candidates.yaml` (staging only)

`candidates.yaml` is written by `scripts/discover.py` and holds *unreviewed*
models. Rows use the `models.yaml` fields above plus the discovery-only fields
below. **Strip these three fields when promoting a row into `models.yaml`** —
`validate.py` checks `models.yaml` only, so they would otherwise leak through.

| Field | Type | Notes |
|-------|------|-------|
| `discovered_via` | list | `org-sweep`, `arena`, or both — which source found it |
| `arena_rank` | integer | Agent Arena rank, present only if arena resolved it. Sorts the review queue. |
| `downloads` | integer | HF download count at discovery time; a rough popularity signal |

Candidates are ordered arena-ranked first (ascending), then unranked by release
date descending — so the models people actually use lead the review queue.

### Open-weight status

`arena_agent_rankings.yaml` sets `open_weight: true` if and only if a public
weights repo resolved on Hugging Face. It is **not** arena's license label and
**not** an org guess. `needs_hf_repo: true` marks an inexact name match that a
human should verify before promotion.
```

- [ ] **Step 6: Verify and commit**

Run: `.venv/bin/python -m pytest tests/ -v && .venv/bin/python scripts/validate.py`
Expected: PASS — 71 passed, then `OK — 16 models, no problems found.`

```bash
git add requirements.txt .github/workflows/validate.yml \
        .github/workflows/discover.yml SCHEMA.md
git commit -m "ci: run tests in validate, pull arena before discovery"
```

---

## Manual step (repo owner, not automatable)

`discover-models` cannot open its PR until this is set:

**Settings → Actions → General → Workflow permissions → "Allow GitHub Actions to create and approve pull requests"**

Verify after Task 7 lands:

```bash
git push open-weight-llm-tracker main
gh workflow run discover-models --repo MakanFar/open-weight-llm-tracker
gh run watch --repo MakanFar/open-weight-llm-tracker --exit-status
```

Expected: the run completes and opens a PR on branch `auto/model-candidates` containing a non-empty `candidates.yaml`. If the final step still reports `GitHub Actions is not permitted to create or approve pull requests`, the setting has not been applied.

## Deferred (explicitly not in this plan)

- Any change to `models.yaml`'s benchmark schema, or adding an arena score column. Revisit once candidates have been promoted and there are rows to populate.
- Backfilling the 15 months of missing models. This plan makes them discoverable; promotion stays a human PR review.
- Replacing the MMLU anchor column.
