# AA / Arena False-Blank Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the README printing `—` in the Arena and AA Index columns for models that upstream actually rates.

**Architecture:** Consolidate the name/repo normalization vocabulary — currently duplicated and drifted between `scripts/pull_arena.py` and `scripts/render_readme.py` — into `scripts/names.py`, and add two new join keys there: `repo_identity()` (repo ids) and `display_identity()` (leaderboard display names). `render_readme.py` gains an identity fallback behind its existing exact-repo match, guarded so an ambiguous identity renders `—` rather than a wrong number. `pull_arena.py` learns to prefer a vendor's primary release repo over a quantized mirror.

**Tech Stack:** Python 3.9, PyYAML, BeautifulSoup4, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-04-aa-arena-false-blanks-design.md` (read the "Amended" section on BF16 before starting Task 3).

## Global Constraints

- **`scripts/names.py` must import only `re`.** `render_readme.py` imports it to stay offline and dependency-light; adding any other import breaks that guarantee. There is a comment on the import in `render_readme.py:21` saying so.
- **`scripts/pull_arena.py` must not import `scripts/discover.py`.** `discover.py` imports `huggingface_hub` at module scope, which would break `pull_arena.py --no-resolve` (the offline parse-only path). Importing `hf_meta` is fine — it is stdlib-only.
- **The README must not change.** `.github/workflows/validate.yml` re-renders `README.md` and fails on any diff. Verified: with every change in this plan applied, 0 of 22 rows change any cell. If `git diff README.md` is non-empty after Task 4, something is wrong — do not commit the regenerated README, debug the join instead.
- **Size tokens are identity.** `repo_identity` must NOT strip `405B`, `7B`, `A22B`, etc. `models.yaml` tracks distinct sizes as separate rows (CLAUDE.md: "one row per model — a family flagship or distinct sizes"). `tests/test_pull_aa.py::test_best_by_slug_keeps_different_sizes_apart` already documents this convention; follow it.
- **Run tests with the repo venv:** `.venv/bin/python -m pytest`. `pytest` is not installed on the base interpreter.
- **Never hand-edit** `models.yaml` (human-only), `README.md` (generated), `aa_scores.yaml`, `arena_agent_rankings.yaml`, or `candidates.yaml` (all AUTO-SCRAPED).

---

### Task 1: `names.py` gains the shared vocabulary and two join keys

Move the normalization vocabulary out of `pull_arena.py` into `names.py` and add
the two new join-key functions. `pull_arena.py` keeps its own copies for now —
Task 2 removes them. This temporary duplication keeps Task 1's test run clean.

**Files:**
- Modify: `scripts/names.py` (append after line 30)
- Test: `tests/test_names.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces, all consumed by Tasks 2–5:
  - `names.ORG_DISPLAY_ALIASES: set[str]`
  - `names.LEADING_ORG_PHRASES: tuple[str, ...]`
  - `names.NATIVE_FORMATS: set[str]`
  - `names.QUANT_FORMATS: set[str]`
  - `names.PRECISION_TOKENS: set[str]` (`NATIVE_FORMATS | QUANT_FORMATS`)
  - `names.SIZE_TOKEN: re.Pattern`
  - `names.DATE_TOKEN: re.Pattern`
  - `names.normalize_display(display: str) -> str`
  - `names.without_leading_vendor(name: str) -> str`
  - `names.strip_decorations(parts: list[str]) -> list[str]` (mutates and returns)
  - `names.strip_repo_decorations(repo_id: str) -> str` (strips size — for `score_match`)
  - `names.repo_identity(repo_id: str) -> str` (keeps size — for joins)
  - `names.display_identity(display: str) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_names.py`:

```python
# --- repo_identity: the render-time join key ---------------------------------

def test_repo_identity_keeps_size_tokens():
    """Size is identity: 405B and 8B are different models, not variants.

    This is the join key, used for an unguarded equality match across a whole
    file. strip_repo_decorations drops size (right for its pairwise ratio
    matching) and is fatally wrong here — both would key to 'llama31'.
    Mirrors tests/test_pull_aa.py::test_best_by_slug_keeps_different_sizes_apart.
    """
    assert names.repo_identity("meta-llama/Llama-3.1-405B-Instruct") == "llama31405b"
    assert names.repo_identity("meta-llama/Llama-3.1-8B-Instruct") == "llama318b"
    assert names.repo_identity("openai/gpt-oss-120b") == "gptoss120b"
    assert names.repo_identity("openai/gpt-oss-20b") == "gptoss20b"
    assert names.repo_identity("ibm-granite/granite-4.1-30b-base") == "granite4130b"
    assert names.repo_identity("ibm-granite/granite-4.1-3b-base") == "granite413b"


def test_repo_identity_drops_a_dated_snapshot_suffix():
    """The DeepSeek V4 Flash split: arena resolved the bare repo, AA the dated one."""
    assert names.repo_identity("deepseek-ai/DeepSeek-V4-Flash-0731") == "deepseekv4flash"
    assert names.repo_identity("deepseek-ai/DeepSeek-V4-Flash") == "deepseekv4flash"


def test_repo_identity_keeps_a_meaningful_suffix():
    """DSpark is a distinct model, not a decoration."""
    assert names.repo_identity(
        "deepseek-ai/DeepSeek-V4-Flash-DSpark") == "deepseekv4flashdspark"


def test_repo_identity_drops_precision_but_not_size():
    """The Nemotron case: NVFP4 and BF16 are the same weights, 550B/A55B are not."""
    assert names.repo_identity(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4") == "nemotron3ultra550ba55b"
    assert names.repo_identity(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16") == "nemotron3ultra550ba55b"


def test_repo_identity_drops_a_duplicated_vendor_prefix():
    assert names.repo_identity("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16") \
        == "nemotron3super120ba12b"


def test_repo_identity_never_empties_a_name():
    """A single-digit tail is not a date; gemma-2 must stay 'gemma2'."""
    assert names.repo_identity("google/gemma-2-27b-it") == "gemma227b"
    assert names.repo_identity("microsoft/phi-4") == "phi4"


def test_repo_identity_strips_a_two_token_date():
    assert names.repo_identity(
        "CohereForAI/c4ai-command-r-plus-08-2024") == "c4aicommandrplus"


# --- display_identity: ordered candidate keys --------------------------------

def test_display_identity_strips_leaderboard_chrome():
    assert names.display_identity("GLM 5.2 (Max) Z.ai · MIT") == ("glm52",)


def test_display_identity_offers_a_vendor_stripped_fallback():
    """Arena writes 'Tencent Hy3'; models.yaml calls it 'Hy3'.

    Both keys are returned, full name FIRST. The full name must win when it
    matches, because the vendor-stripped form is lossy — 'Mistral Small 3'
    reduces to 'small3', which could collide with another vendor's Small 3.
    """
    assert names.display_identity("Tencent Hy3 Tencent · Apache 2.0") \
        == ("tencenthy3", "hy3")
    assert names.display_identity("Thinking Machines Inkling Thinky · Apache 2.0") \
        == ("thinkingmachinesinkling", "inkling")
    assert names.display_identity("Mistral Small 3") == ("mistralsmall3", "small3")


def test_display_identity_returns_one_key_when_there_is_no_leading_vendor():
    assert names.display_identity("Kimi K3") == ("kimik3",)
    assert names.display_identity("Nemotron 3 Ultra Nvidia · OpenMDW-1.1") \
        == ("nemotron3ultra",)


def test_display_identity_drops_a_variant_suffix():
    assert names.display_identity("Llama 3.3 70B Instruct") == ("llama3370b",)


def test_precision_tokens_is_the_union_of_native_and_quant():
    assert names.PRECISION_TOKENS == names.NATIVE_FORMATS | names.QUANT_FORMATS
    assert "bf16" in names.NATIVE_FORMATS
    assert "nvfp4" in names.QUANT_FORMATS
    assert not names.NATIVE_FORMATS & names.QUANT_FORMATS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_names.py -v`

Expected: FAIL, `AttributeError: module 'names' has no attribute 'repo_identity'`.

- [ ] **Step 3: Implement in `scripts/names.py`**

Append to `scripts/names.py`, after the existing `strip_variant_suffix`:

```python
# --- leaderboard display vocabulary ------------------------------------------
# Moved here from pull_arena.py so render_readme.py can share one set of rules.
# render_readme.py had a drifted copy that stripped a TRAILING org token but not
# a LEADING one, so arena's "Tencent Hy3" never matched models.yaml's "Hy3".

# Org display names as they appear appended to leaderboard labels, e.g.
# "GLM 5.2 (Max) Z.ai · MIT".
ORG_DISPLAY_ALIASES = {
    "anthropic", "openai", "google", "meta", "alibaba", "deepseek",
    "moonshot", "z.ai", "zai", "minimax", "nvidia", "xai", "microsoft",
    "cohere", "mistral", "tencent", "xiaomi", "thinky", "ibm", "baidu",
    "ai2", "01.ai", "tii", "zhipu",
}

# Vendor names that appear at the START of a display label. Sorted longest-first
# at use so a multi-word vendor beats its first word.
LEADING_ORG_PHRASES = (
    "thinking machines", "meta", "anthropic", "google", "openai", "nvidia",
    "microsoft", "alibaba", "tencent", "xiaomi", "baidu", "mistral", "cohere",
)

# --- repo-slug decorations ---------------------------------------------------

# A parameter count (550B) or an active-parameter count (A55B).
SIZE_TOKEN = re.compile(r"^[aA]?\d+(?:\.\d+)?[bBmM]$")

# A vendor's PRIMARY release format. These are name decorations, but they are
# deliberately NOT in hf_meta.EXCLUDE_PATTERNS: NVIDIA ships Nemotron 3 Ultra
# only as -BF16 repos, so excluding them would delete the model from the
# tracker. See tests/test_hf_meta.py.
NATIVE_FORMATS = {"bf16", "fp16", "fp32"}

# Quantized re-releases. Every one of these MUST also be caught by
# hf_meta.EXCLUDE_PATTERNS — a test asserts it.
QUANT_FORMATS = {
    "fp8", "int4", "int8", "nvfp4", "mxfp8", "w4a16", "w8a8",
    "4bit", "8bit", "gguf", "awq", "gptq",
}

# Both kinds are decorations when comparing two names.
PRECISION_TOKENS = NATIVE_FORMATS | QUANT_FORMATS

# A dated snapshot suffix: "0731", "2501", "20240730", or the "2024" of
# "...-08-2024". Two, four, six or eight digits. Only ever stripped from the
# tail, and never if it is the only token left.
DATE_TOKEN = re.compile(r"^\d{2}(?:\d{2}){0,3}$")


def normalize_display(display):
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


def without_leading_vendor(name):
    """Drop a leading vendor phrase: 'Thinking Machines Inkling' -> 'Inkling'.

    Lossy on purpose-built names — "Mistral Small 3" becomes "Small 3" — so
    callers must treat this as a FALLBACK behind the full name, never as a
    replacement for it. Longest phrase first, so "thinking machines" beats
    "thinking".
    """
    tokens = name.split(" ")
    for phrase in sorted(LEADING_ORG_PHRASES, key=len, reverse=True):
        parts = phrase.split(" ")
        if len(tokens) > len(parts) and \
                [t.lower() for t in tokens[:len(parts)]] == parts:
            return " ".join(tokens[len(parts):]).strip()
    return name


def strip_decorations(parts):
    """Drop trailing size/precision/variant tokens from a token list.

    Mutates and returns `parts`. Strips SIZE — used by pull_arena's pairwise
    matching, where the arena display name may omit it. Not for join keys;
    see repo_identity.
    """
    while parts:
        last = parts[-1].lower()
        if last in PRECISION_TOKENS or last in VARIANT_SUFFIXES \
                or SIZE_TOKEN.match(parts[-1]):
            parts.pop()
            continue
        break
    return parts


def strip_repo_decorations(repo_id):
    """Reduce a repo id to its identifying name, size included.

    Drops the author, a duplicated vendor prefix ("nvidia/NVIDIA-Nemotron-…"),
    then trailing size/precision/variant tokens. Only decorations go: a token
    like "Nano" or "Ultra" distinguishes two real models and is kept, so
    Nemotron-3-Ultra and Nemotron-3-Nano do not collapse together.
    """
    author, _, tail = repo_id.rpartition("/")
    parts = [p for p in re.split(r"[-_.]", tail) if p]

    if author and parts and parts[0].lower() == author.split("/")[-1].lower():
        parts = parts[1:]

    return "-".join(strip_decorations(parts))


def repo_identity(repo_id):
    """A join key for an HF repo id. KEEPS size tokens.

    Distinct from strip_repo_decorations, which drops them. That function
    compares one arena display name against a small, scoped candidate set and
    is guarded by a confidence rating, so losing size is safe there. This one
    is an UNGUARDED equality join across a whole file, and size is part of a
    model's identity in models.yaml ("one row per model — a family flagship or
    distinct sizes"). Stripping it collapses Llama-3.1-405B onto Llama-3.1-8B.

    Drops: the author, a duplicated vendor prefix, and trailing precision,
    variant and dated-snapshot tokens.
    """
    author, _, tail = repo_id.rpartition("/")
    parts = [p for p in re.split(r"[-_.]", tail) if p]

    if author and parts and parts[0].lower() == author.split("/")[-1].lower():
        parts = parts[1:]

    while len(parts) > 1 and (parts[-1].lower() in PRECISION_TOKENS
                              or parts[-1].lower() in VARIANT_SUFFIXES
                              or DATE_TOKEN.match(parts[-1])):
        parts.pop()

    return slug("-".join(parts))


def display_identity(display):
    """Ordered join keys for a leaderboard display name, best first.

    Returns the full normalized name, then the vendor-stripped form. A caller
    must try them in order and stop at the first hit: the vendor-stripped key
    is lossy ("Mistral Small 3" -> "small3") and only safe as a fallback.

    A tuple, not a set — iterating a set is hash-randomized, so if two keys
    ever resolved to different entries, which one won would vary between runs
    on identical input.
    """
    base = normalize_display(display)
    keys = []
    for candidate in (base, without_leading_vendor(base)):
        key = slug(strip_variant_suffix(candidate))
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_names.py -v`

Expected: PASS, all tests (4 pre-existing + 12 new).

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: PASS. `pull_arena.py` still has its own copies, so nothing else changed yet.

- [ ] **Step 6: Commit**

```bash
git add scripts/names.py tests/test_names.py
git commit -m "feat(names): add repo_identity and display_identity join keys

repo_identity KEEPS size tokens, unlike strip_repo_decorations. Size is
identity in models.yaml, and the join is an unguarded equality match, so
stripping it would collapse Llama-3.1-405B onto Llama-3.1-8B.

display_identity returns an ordered tuple, full name before the
vendor-stripped fallback, because the latter is lossy: Mistral Small 3
reduces to 'small3'.

PRECISION_TOKENS splits into NATIVE_FORMATS and QUANT_FORMATS. BF16 is a
name decoration but not a reason to reject a repo — NVIDIA ships Nemotron
3 Ultra only as -BF16."
```

---

### Task 2: `pull_arena.py` consumes the shared vocabulary

Delete `pull_arena.py`'s now-duplicated definitions and import them from
`names`. Behaviour must not change — the existing arena tests are the proof.

**Files:**
- Modify: `scripts/pull_arena.py` (delete lines ~97–125 and ~172–215; see steps)
- Test: `tests/test_arena_names.py`, `tests/test_arena_resolve.py` (unchanged, act as the regression gate)
- Test: `tests/test_hf_meta.py` (add the vocabulary-consistency tests)

**Interfaces:**
- Consumes: everything Task 1 produced.
- Produces: `pull_arena.normalize_model_name` is retained as a module-level alias of `names.normalize_display`. Do NOT delete it — `tests/test_arena_names.py:30` and `tests/test_arena_resolve.py:307` call it, and so do `resolve_row` and `carry_forward_resolutions`.

- [ ] **Step 1: Write the failing vocabulary-consistency tests**

In `tests/test_hf_meta.py`, add `import names` directly below the existing
`import hf_meta` (line 9), then append:

```python
def test_every_quant_format_is_excluded_by_hf_meta():
    """Two overlapping vocabularies, held together by a test not by coupling.

    names.QUANT_FORMATS says "this token is a quantized re-release";
    hf_meta.EXCLUDE_PATTERNS says "reject this repo". Every quant format must
    appear in both, or pull_arena would de-prioritise a repo that discover.py
    then happily stages. Same technique as
    test_author_hints_stay_in_step_with_org_allowlist.

    Tokens are probed as "model-<token>" because EXCLUDE_PATTERNS anchors
    several of them on a leading hyphen (-int4, -fp8, -4bit).
    """
    missing = sorted(t for t in names.QUANT_FORMATS
                     if not hf_meta.EXCLUDE_PATTERNS.search(f"model-{t}"))
    assert not missing, (
        f"names.QUANT_FORMATS entries not caught by EXCLUDE_PATTERNS: {missing}")


def test_native_formats_are_deliberately_not_excluded():
    """BF16 is a release format, not a quantization. Do not 'fix' this gap.

    NVIDIA publishes Nemotron 3 Ultra ONLY as -BF16 repos — there is no bare
    nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B. Adding bf16 to EXCLUDE_PATTERNS
    would delete the model from the tracker entirely. The token still belongs
    in names.PRECISION_TOKENS, because it IS noise when comparing two names.
    """
    for fmt in names.NATIVE_FORMATS:
        assert not hf_meta.EXCLUDE_PATTERNS.search(f"model-{fmt}"), (
            f"{fmt} is a primary release format and must stay excludable-free")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hf_meta.py -v -k "quant_format or native_formats"`

Expected: **PASS**, both tests. This is deliberate — Task 1 already defined
`QUANT_FORMATS` and `NATIVE_FORMATS`, and these two are regression guards
against a future contributor closing the BF16 gap, not red-green drivers. If
either FAILS, Task 1 is wrong: `test_every_quant_format_is_excluded_by_hf_meta`
failing means a token is in `QUANT_FORMATS` that `hf_meta` does not reject, and
`test_native_formats_are_deliberately_not_excluded` failing means a native
format was wrongly added to `EXCLUDE_PATTERNS`.

- [ ] **Step 3: Delete the duplicated definitions from `scripts/pull_arena.py`**

Delete these blocks entirely (they now live in `names.py`):

- `ORG_DISPLAY_ALIASES = {...}` and its comment
- `_LEADING_ORG_PHRASES = (...)` and its comment
- `_SIZE_TOKEN = ...` and `_PRECISION_TOKENS = {...}` and their comment
- `def normalize_model_name(display):` (whole function)
- `def without_leading_vendor(name):` (whole function)
- `def strip_repo_decorations(repo_id):` (whole function)
- `def _strip_decorations(parts):` (whole function)

Then replace the existing import block:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
import names
slug = names.slug
_VARIANT_SUFFIXES = names.VARIANT_SUFFIXES
```

with:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
import names

# The name/repo normalisation vocabulary lives in names.py so render_readme.py
# can share it — an earlier drifted copy there is what broke the arena name
# fallback for org-prefixed labels like "Tencent Hy3".
slug = names.slug

# Public alias: this is the name the tests and the rest of this module use.
normalize_model_name = names.normalize_display
```

- [ ] **Step 4: Repoint `score_match` at `names`**

Replace the body of `score_match` with:

```python
def score_match(query, repo_id):
    """Rate how well an arena name matches an HF repo id: high/medium/low.

    Decorations are stripped from BOTH sides. Stripping only the repo breaks
    the symmetric case: "Gemma 4 31B" vs "google/gemma-4-31b-it" would compare
    "gemma431b" against "gemma4" and score low.
    """
    r = slug(names.strip_repo_decorations(repo_id))
    for candidate in (query, names.without_leading_vendor(query)):
        q = slug("-".join(names.strip_decorations(
            [p for p in re.split(r"[-_.\s]", candidate) if p])))
        conf = _rate(q, r)
        if conf != "low":
            return conf
    return "low"
```

- [ ] **Step 5: Run the arena tests to verify behaviour is unchanged**

Run: `.venv/bin/python -m pytest tests/test_arena_names.py tests/test_arena_resolve.py tests/test_hf_meta.py -v`

Expected: PASS, every test. Any failure means the move changed behaviour — diff the moved function against the original rather than adjusting the test.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/pull_arena.py tests/test_hf_meta.py
git commit -m "refactor(arena): source normalisation from names.py

Deletes pull_arena's copies of the display/repo vocabulary now that
names.py owns them. No behaviour change — the existing arena tests are
the gate. normalize_model_name stays as an alias; tests and resolve_row
call it by that name.

Adds the two tests that hold names.QUANT_FORMATS and
hf_meta.EXCLUDE_PATTERNS in step, and pins NATIVE_FORMATS as
deliberately not excluded so nobody closes that gap and deletes
Nemotron 3 Ultra."
```

---

### Task 3: `pull_arena.py` prefers a vendor's primary release repo

Arena resolved Nemotron 3 Ultra to the `-NVFP4` repo, which
`hf_meta.EXCLUDE_PATTERNS` then rejects, so rank 42 never reaches the review
queue. Rank the search hits by format after confidence.

**Files:**
- Modify: `scripts/pull_arena.py` (`resolve_row`, the `best`/`best_conf` loop)
- Test: `tests/test_arena_resolve.py`

**Interfaces:**
- Consumes: `names.QUANT_FORMATS`, `names.NATIVE_FORMATS` from Task 1.
- Produces: `pull_arena.format_rank(repo_id) -> int` — `2` clean, `1` native-only, `0` any quantized token. Used only inside `resolve_row`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arena_resolve.py`:

```python
# --- format preference -------------------------------------------------------

def test_format_rank_orders_clean_above_native_above_quantized():
    assert pa.format_rank("zai-org/GLM-5.2") == 2
    assert pa.format_rank("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16") == 1
    assert pa.format_rank("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4") == 0


def test_prefers_the_native_repo_over_the_quantized_one():
    """The Nemotron case. NVFP4 is in hf_meta.EXCLUDE_PATTERNS, so resolving to
    it means discover.py drops the row and arena rank 42 never reaches the
    review queue. BF16 is not excluded and must win.
    """
    row = {"model": "Nemotron 3 Ultra Nvidia · OpenMDW-1.1",
           "org": "NVIDIA", "matched_keyword": "nemotron"}
    search = fake_search({"nvidia": [
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",   # listed first
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    ]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    assert out["open_weight"] is True


def test_a_high_confidence_quant_does_not_short_circuit_the_search():
    """The old loop broke on the first 'high' hit, so a clean repo later in the
    result list was never examined."""
    row = {"model": "GLM 5.2 Z.ai · MIT", "org": "Zhipu AI",
           "matched_keyword": "glm"}
    search = fake_search({"zai-org": [
        "zai-org/GLM-5.2-GPTQ",   # high confidence, quantized, listed first
        "zai-org/GLM-5.2",        # high confidence, clean
    ]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "zai-org/GLM-5.2"
    assert out["resolution_confidence"] == "high"


def test_format_preference_never_beats_confidence():
    """A clean but low-confidence repo must not outrank a high-confidence one."""
    row = {"model": "Kimi K2.6 Moonshot · Modified MIT",
           "org": "Moonshot AI", "matched_keyword": "kimi"}
    search = fake_search({"moonshotai": [
        "moonshotai/Kimi-K2-Thinking",   # clean, but low confidence
        "moonshotai/Kimi-K2.6-BF16",     # high confidence, native format
    ]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "moonshotai/Kimi-K2.6-BF16"
    assert out["resolution_confidence"] == "high"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_arena_resolve.py -v -k "format_rank or native_repo or short_circuit or never_beats"`

Expected: FAIL, `AttributeError: module 'pull_arena' has no attribute 'format_rank'`.

- [ ] **Step 3: Add `format_rank` to `scripts/pull_arena.py`**

Insert immediately above `resolve_row`:

```python
def format_rank(repo_id):
    """Prefer a vendor's primary release repo over a quantized mirror.

    2 = no precision token, 1 = a native format only (BF16/FP16/FP32),
    0 = carries a quantization token.

    Not a boolean, because "is it clean?" cannot choose when every candidate
    carries a token — NVIDIA publishes Nemotron 3 Ultra only as -BF16 and
    -NVFP4, and resolving to the NVFP4 repo means hf_meta.EXCLUDE_PATTERNS
    rejects it and the arena rank never reaches the review queue.
    """
    tokens = {p.lower() for p in re.split(r"[-_./]", repo_id) if p}
    if tokens & names.QUANT_FORMATS:
        return 0
    if tokens & names.NATIVE_FORMATS:
        return 1
    return 2
```

- [ ] **Step 4: Replace the selection loop in `resolve_row`**

Find this block in `resolve_row`:

```python
    best, best_conf = None, "low"
    _ORDER = {"high": 3, "medium": 2, "low": 1}
    for repo_id in repo_ids:
        conf = score_match(query, repo_id)
        if best is None or _ORDER[conf] > _ORDER[best_conf]:
            best, best_conf = repo_id, conf
        if best_conf == "high":
            break
```

Replace it with:

```python
    # Every hit is scored — no early break. An earlier version stopped at the
    # first "high" match, so a better-format repo further down the result list
    # was never examined. Confidence dominates; format only breaks ties.
    _ORDER = {"high": 3, "medium": 2, "low": 1}
    scored = [(repo_id, score_match(query, repo_id)) for repo_id in repo_ids]
    best, best_conf = None, "low"
    if scored:
        best, best_conf = max(
            scored, key=lambda pair: (_ORDER[pair[1]], format_rank(pair[0])))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_arena_resolve.py -v`

Expected: PASS, all tests including the pre-existing
`test_prefers_highest_scoring_candidate` and `test_unresolvable_model_is_not_open_weight`
(the `scored == []` case must still leave `best is None`).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/pull_arena.py tests/test_arena_resolve.py
git commit -m "fix(arena): prefer a primary release repo over a quantized mirror

Arena resolved Nemotron 3 Ultra to the -NVFP4 repo, which
hf_meta.EXCLUDE_PATTERNS then rejects, so rank 42 never reached the
review queue — the only one of 14 resolved repos missing from
candidates.yaml.

format_rank is three-level, not boolean: NVIDIA ships this model only as
-BF16 and -NVFP4, so 'is it clean?' cannot choose between them.
Confidence still dominates; format only breaks ties.

Also drops the early break on the first 'high' hit, which stopped the
search before a better-format repo later in the list was examined."
```

---

### Task 4: `render_readme.py` identity fallback with a collision guard

Both joins key on an exact lowercased `hf_repo`, so the DeepSeek V4 Flash split
prints `—` for a model both sources rate. And the renderer's private copy of the
display normalizer strips a trailing org token but not a leading one, so
`"Tencent Hy3"` never matches `"Hy3"`.

**Files:**
- Modify: `scripts/render_readme.py` (delete `_ORG_TAIL_WORDS`, `_slug`, `_arena_display_name`; rewrite `load_arena_ranks`, `load_aa_scores`, `aa_cell`, `arena_cell`)
- Test: `tests/test_render_readme.py`

**Interfaces:**
- Consumes: `names.repo_identity`, `names.display_identity` from Task 1.
- Produces: `load_aa_scores()` now returns `{"repos": {...}, "identities": {...}}` instead of a flat dict, and `load_arena_ranks()` gains an `"identities"` key. **This is a breaking shape change** — `aa_cell(model, aa)` and `arena_cell(model, ranks)` must be updated together with them, and the pre-existing tests that construct these dicts by hand must be updated too (listed in Step 1).

- [ ] **Step 1: Update the existing tests and write the failing new ones**

In `tests/test_render_readme.py`, first update the three pre-existing tests that
build an AA dict by hand, because the shape changes:

```python
def test_aa_cell_shows_the_index():
    m = _model(hf_repo="moonshotai/Kimi-K3")
    aa = {"repos": {"moonshotai/kimi-k3": {"index": 57, "variant": "max"}},
          "identities": {}}
    assert rr.aa_cell(m, aa) == "57"


def test_aa_cell_dashes_when_aa_does_not_rate_the_model():
    """There is no fallback to a manual figure: models.yaml carries no score."""
    empty = {"repos": {}, "identities": {}}
    assert rr.aa_cell(_model(hf_repo="org/m"), empty) == "—"
```

Update `test_load_aa_scores_parses_the_sidecar` to assert against
`scores["repos"]` rather than `scores`. Update
`test_arena_cell_matches_by_resolved_repo`,
`test_arena_cell_falls_back_to_name_when_the_repo_never_resolved` and
`test_arena_cell_dash_when_the_model_is_not_ranked` to include
`"identities": {}` in the dicts they build.

Then append the new tests:

```python
# --- identity fallback -------------------------------------------------------

def test_aa_cell_falls_back_to_repo_identity():
    """The DeepSeek V4 Flash split: arena resolved the bare repo, AA scored the
    dated snapshot. Both name the same weights."""
    m = _model(hf_repo="deepseek-ai/DeepSeek-V4-Flash")
    aa = {"repos": {"deepseek-ai/deepseek-v4-flash-0731":
                    {"index": 50, "variant": "max"}},
          "identities": {"deepseekv4flash": {"index": 50, "variant": "max"}}}
    assert rr.aa_cell(m, aa) == "50"


def test_aa_cell_prefers_an_exact_repo_match_over_an_identity_match():
    m = _model(hf_repo="deepseek-ai/DeepSeek-V4-Flash")
    aa = {"repos": {"deepseek-ai/deepseek-v4-flash": {"index": 44, "variant": "default"}},
          "identities": {"deepseekv4flash": {"index": 50, "variant": "max"}}}
    assert rr.aa_cell(m, aa) == "44"


def test_arena_cell_falls_back_to_repo_identity():
    """Arena resolved the NVFP4 mirror; the tracked row is the BF16 release."""
    m = _model(hf_repo="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
               name="Nemotron 3 Ultra")
    ranks = {"repos": {}, "names": {}, "identities": {"nemotron3ultra550ba55b": 42}}
    assert rr.arena_cell(m, ranks) == "42"


def test_arena_cell_falls_back_to_a_vendor_prefixed_display_name():
    """Arena writes 'Tencent Hy3'; models.yaml calls the model 'Hy3'.

    The renderer's old private normalizer stripped a TRAILING org token but not
    a LEADING one, so this never matched.
    """
    m = _model(hf_repo="tencent/Hy3", name="Hy3")
    ranks = rr.load_arena_ranks_from_rows(
        [{"rank": 27, "model": "Tencent Hy3 Tencent · Apache 2.0",
          "resolved_repo": None}])
    assert rr.arena_cell(m, ranks) == "27"


def test_arena_cell_matches_thinking_machines_inkling():
    m = _model(hf_repo="thinkingmachines/Inkling", name="Inkling")
    ranks = rr.load_arena_ranks_from_rows(
        [{"rank": 36, "model": "Thinking Machines Inkling Thinky · Apache 2.0",
          "resolved_repo": None}])
    assert rr.arena_cell(m, ranks) == "36"


# --- collision guard ---------------------------------------------------------

def test_ambiguous_identity_is_dropped_rather_than_guessed(capsys):
    """Two sidecar entries sharing an identity means we cannot know which one
    the tracked row refers to. A wrong number is worse than no number."""
    m = _model(hf_repo="org/Model-C")
    aa = rr.load_aa_scores_from_dict({
        "org/Model-A": {"intelligence_index": 10},
        "org/Model-B": {"intelligence_index": 20},
    }, identity_of=lambda repo: "collide")
    assert rr.aa_cell(m, aa) == "—"
    assert "collide" in capsys.readouterr().out


def test_distinct_sizes_do_not_collide():
    """405B and 8B are different models. If repo_identity ever stripped size,
    this pair would share an identity and BOTH would be dropped by the guard."""
    aa = rr.load_aa_scores_from_dict({
        "meta-llama/Llama-3.1-405B-Instruct": {"intelligence_index": 30},
        "meta-llama/Llama-3.1-8B-Instruct": {"intelligence_index": 12},
    })
    assert aa["identities"]["llama31405b"]["index"] == 30
    assert aa["identities"]["llama318b"]["index"] == 12
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_render_readme.py -v`

Expected: FAIL — `AttributeError: module 'render_readme' has no attribute 'load_arena_ranks_from_rows'`, plus KeyErrors from the reshaped dicts.

- [ ] **Step 3: Rewrite the loaders and cells in `scripts/render_readme.py`**

Delete `_ORG_TAIL_WORDS`, `_slug` and `_arena_display_name` entirely — `names`
now owns all of it.

Replace `load_arena_ranks`, `load_aa_scores`, `aa_cell` and `arena_cell` with:

```python
def _index_by_identity(pairs, identity_of=names.repo_identity):
    """{identity: value} from (repo_id, value) pairs, ambiguous keys dropped.

    An identity claimed by two entries with DIFFERENT values cannot be resolved
    — we would be guessing which one the tracked row means — so it is dropped
    and reported. Identical values are harmless and kept: two spellings of the
    same repo naturally agree.

    Values are compared with != rather than collected into a set, because an AA
    value is a dict and dicts are unhashable.

    identity_of is a seam for testing the guard; production always uses
    names.repo_identity.
    """
    grouped = {}
    for repo_id, value in pairs:
        grouped.setdefault(identity_of(repo_id), []).append((repo_id, value))

    out = {}
    for identity, claims in grouped.items():
        if not identity:
            continue
        first = claims[0][1]
        if any(value != first for _, value in claims[1:]):
            print(f"  ! identity {identity!r} claimed by "
                  f"{sorted(r for r, _ in claims)}; not joining")
            continue
        out[identity] = first
    return out


def load_arena_ranks_from_rows(rows):
    """Build the rank indexes from already-parsed arena rows.

    Split out from load_arena_ranks so tests can exercise the indexing without
    a file on disk.
    """
    repos, name_index, identity_pairs = {}, {}, []
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("rank"), int):
            continue
        if r.get("resolved_repo"):
            repos[str(r["resolved_repo"]).lower()] = r["rank"]
            identity_pairs.append((str(r["resolved_repo"]), r["rank"]))
        if r.get("model"):
            # Full name first, vendor-stripped second. setdefault keeps the
            # BEST rank: rows arrive rank-ordered, and the same model listed at
            # several reasoning efforts legitimately repeats a display name.
            for key in names.display_identity(str(r["model"])):
                name_index.setdefault(key, r["rank"])
    return {"repos": repos, "names": name_index,
            "identities": _index_by_identity(identity_pairs)}


def load_arena_ranks(path=ARENA):
    """Rank indexes from arena_agent_rankings.yaml.

    Three indexes, because a rank and a weights repo are separate facts. HF
    resolution fails transiently — one rate-limited search writes
    resolved_repo: null — and a rank already scraped should not vanish from the
    table because of it. Open-weight status still comes only from resolution;
    these indexes decide where a number is printed, nothing more.
    """
    empty = {"repos": {}, "names": {}, "identities": {}}
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return empty
    rows = doc.get("arena_agent") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return empty
    return load_arena_ranks_from_rows(rows)


def load_aa_scores_from_dict(scores, identity_of=names.repo_identity):
    """Build the AA indexes from an already-parsed `scores:` mapping.

    Split out from load_aa_scores so tests can exercise the indexing and the
    collision guard without a file on disk.
    """
    repos, pairs = {}, []
    if isinstance(scores, dict):
        for repo, entry in scores.items():
            if not isinstance(entry, dict):
                continue
            idx = entry.get("intelligence_index")
            if isinstance(idx, int) and not isinstance(idx, bool):
                value = {"index": idx, "variant": entry.get("variant")}
                repos[str(repo).lower()] = value
                pairs.append((str(repo), value))

    return {"repos": repos,
            "identities": _index_by_identity(pairs, identity_of)}


def load_aa_scores(path=AA):
    """AA indexes keyed by lowercased repo and by repo identity."""
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {"repos": {}, "identities": {}}
    scores = doc.get("scores") if isinstance(doc, dict) else None
    return load_aa_scores_from_dict(scores)


def aa_cell(model, aa):
    """Artificial Analysis Intelligence Index, or — when AA does not rate it.

    Exact repo first, then repo identity: AA and arena disagree about which
    repo string names a model (AA scored DeepSeek-V4-Flash-0731, arena resolved
    DeepSeek-V4-Flash), and an exact-only join prints — for a model both rate.

    There is deliberately no fallback to a manual figure: models.yaml no longer
    carries one, because MMLU (~86) and the AA index (~10-57) are different
    scales and sharing a column invited a comparison that does not exist.
    """
    repo = model.get("hf_repo") or ""
    entry = aa["repos"].get(repo.lower())
    if entry is None and repo:
        entry = aa["identities"].get(names.repo_identity(repo))
    return str(entry["index"]) if entry else "—"


def arena_cell(model, ranks):
    """Rank by resolved repo, then repo identity, then display name, else '—'."""
    repo = model.get("hf_repo") or ""
    rank = ranks["repos"].get(repo.lower())
    if rank is None and repo:
        rank = ranks["identities"].get(names.repo_identity(repo))
    if rank is None:
        for key in names.display_identity(str(model.get("name") or "")):
            rank = ranks["names"].get(key)
            if rank is not None:
                break
    return str(rank) if rank is not None else "—"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_render_readme.py -v`

Expected: PASS, all tests.

- [ ] **Step 5: Verify the README is unchanged — the honesty gate**

Run:

```bash
.venv/bin/python scripts/render_readme.py && git diff --stat README.md
```

Expected: `Rendered README.md with 22 models.` and **empty** `git diff` output.

A non-empty diff means the identity fallback filled a cell that was previously
blank. That is a bug in this change, not a discovery: every currently-blank row
belongs to a model neither source rates. Stop and debug — do not commit a
changed README.

- [ ] **Step 6: Run the full suite and the validator**

Run:

```bash
.venv/bin/python -m pytest tests/ -q && .venv/bin/python scripts/validate.py
```

Expected: PASS, then `OK — 22 models, no problems found.`

- [ ] **Step 7: Commit**

```bash
git add scripts/render_readme.py tests/test_render_readme.py
git commit -m "fix(render): join AA and arena by repo identity, not just exact repo

Both joins keyed on an exact lowercased hf_repo, so the DeepSeek V4
Flash split — arena resolved the bare repo, AA scored the -0731
snapshot — printed a dash for a model both sources rate.

Deletes the renderer's private _arena_display_name and _ORG_TAIL_WORDS,
a drifted copy of pull_arena's normalizer that stripped a trailing org
token but not a leading one, so 'Tencent Hy3' never matched 'Hy3'.
names.display_identity now serves both.

An identity claimed by two entries with different values is dropped and
reported rather than guessed. README output is byte-identical."
```

---

### Task 5: `validate.py` rejects duplicate model identities

The render-time guard covers the sidecar side. Two `models.yaml` rows sharing an
identity would both claim the same sidecar entry — a duplicate-row bug, and a
violation of CLAUDE.md's "one row per model".

**Files:**
- Modify: `scripts/validate.py`
- Test: `tests/test_validate.py` (create)

**Interfaces:**
- Consumes: `names.repo_identity` from Task 1.
- Produces: `validate.identity_errors(models) -> list[str]`, called from `main()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import validate


def test_identity_errors_flags_two_rows_naming_the_same_weights():
    """A dated snapshot and its bare repo are one model, so one row.

    CLAUDE.md: 'one row per model — a family flagship or distinct sizes, never
    both.' Two rows sharing an identity would also both claim the same AA
    sidecar entry at render time.
    """
    models = [
        {"name": "DeepSeek V4 Flash", "hf_repo": "deepseek-ai/DeepSeek-V4-Flash"},
        {"name": "DeepSeek V4 Flash 0731",
         "hf_repo": "deepseek-ai/DeepSeek-V4-Flash-0731"},
    ]
    errors = validate.identity_errors(models)
    assert len(errors) == 1
    assert "deepseekv4flash" in errors[0]


def test_identity_errors_allows_distinct_sizes():
    """405B and 8B are separate rows by design."""
    models = [
        {"name": "Llama 3.1 405B", "hf_repo": "meta-llama/Llama-3.1-405B-Instruct"},
        {"name": "Llama 3.1 8B", "hf_repo": "meta-llama/Llama-3.1-8B-Instruct"},
    ]
    assert validate.identity_errors(models) == []


def test_identity_errors_ignores_rows_without_a_repo():
    assert validate.identity_errors([{"name": "M"}, {"name": "N", "hf_repo": ""}]) == []


def test_the_committed_models_file_has_no_identity_collisions():
    import yaml
    doc = yaml.safe_load(validate.DATA.read_text())
    assert validate.identity_errors(doc["models"]) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_validate.py -v`

Expected: FAIL, `AttributeError: module 'validate' has no attribute 'identity_errors'`.

- [ ] **Step 3: Implement in `scripts/validate.py`**

`import sys` and `from pathlib import Path` are already present. Add the `names`
import directly below the existing `import yaml`:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
import names  # noqa: E402  (names.py imports only `re`)
```

Add the function above `main()`:

```python
def identity_errors(models):
    """Reject two rows whose hf_repos name the same weights.

    render_readme joins AA scores and arena ranks by repo identity when the
    exact repo string misses. Two rows sharing an identity would both claim the
    same sidecar entry, so the renderer's guard would drop it and BOTH rows
    would lose their number. It is also a plain duplicate: CLAUDE.md allows one
    row per model, a family flagship or distinct sizes, never both.
    """
    seen = {}
    errors = []
    for m in models:
        repo = m.get("hf_repo")
        if not repo:
            continue
        identity = names.repo_identity(repo)
        if not identity:
            continue
        if identity in seen:
            errors.append(
                f"[{m.get('name', repo)}] hf_repo '{repo}' names the same model "
                f"as '{seen[identity]}' (shared identity '{identity}') — "
                f"keep one row per model")
        else:
            seen[identity] = repo
    return errors
```

Then, inside `main()`, after the `for i, m in enumerate(models):` loop ends and
before `if errors:`, add:

```python
    errors.extend(identity_errors(models))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_validate.py -v`

Expected: PASS, all four tests.

- [ ] **Step 5: Run the validator against the real file**

Run: `.venv/bin/python scripts/validate.py`

Expected: `OK — 22 models, no problems found.`

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/validate.py tests/test_validate.py
git commit -m "feat(validate): reject two rows naming the same weights

render_readme falls back to a repo-identity join when the exact repo
string misses. Two models.yaml rows sharing an identity would both claim
the same sidecar entry, the renderer's guard would drop it, and BOTH
rows would lose their number.

Enforces CLAUDE.md's 'one row per model' mechanically. models.yaml is
clean today, so this costs nothing on landing."
```

---

### Task 6: `discover.py` reports the arena rank it is dropping

When `should_track` rejects an arena-resolved repo, the rank goes with it
silently. Task 3 handles the case where a better repo exists in the search
results; this covers the residual case where it does not.

**Files:**
- Modify: `scripts/discover.py:276`
- Test: `tests/test_discover_queue.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new — a message-text change only.

- [ ] **Step 1: Write the failing test**

`tests/test_discover_queue.py` already imports `discover` and `FakeInfo` (from
`test_hf_meta`) at lines 16–18. Append:

```python
def test_a_skipped_arena_repo_reports_the_rank_it_takes_with_it(capsys):
    """A dropped quantization also drops its leaderboard rank.

    hf_meta.EXCLUDE_PATTERNS is right to reject a quantized repo, but the rank
    is real and the model may deserve a row under its primary repo. Say so,
    rather than dropping it silently — this is how Nemotron 3 Ultra's rank 42
    vanished from the review queue.
    """
    class _Api:
        def model_info(self, repo, expand=None):
            return FakeInfo(id=repo)

    rows = [{"resolved_repo": "nvidia/Some-Model-NVFP4", "rank": 42}]
    discover.arena_candidates(_Api(), rows, min_params=0, known=set())

    out = capsys.readouterr().out
    assert "nvidia/Some-Model-NVFP4" in out
    assert "42" in out
```

`should_track` calls `is_derivative(info.id)` first, so the `-NVFP4` suffix is
rejected before any other attribute of `FakeInfo` is read and `resolve_facts` is
never reached.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_discover_queue.py -v -k "takes_with_it"`

Expected: FAIL — the message contains the repo and the skip reason but not `42`.

- [ ] **Step 3: Implement in `scripts/discover.py`**

Find, inside `arena_candidates`:

```python
        keep, reason = hf_meta.should_track(info, min_params)
        if not keep:
            print(f"  - {repo} skipped ({reason})")
            continue
```

Replace with:

```python
        keep, reason = hf_meta.should_track(info, min_params)
        if not keep:
            # Name the rank being dropped. The filter is right to reject a
            # quantization, but the rank is real and the model may deserve a
            # row under its primary repo — this is how Nemotron 3 Ultra's
            # rank 42 vanished from the queue without a trace.
            print(f"  - {repo} skipped ({reason}; arena rank "
                  f"{row.get('rank')} dropped)")
            continue
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_discover_queue.py -v`

Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/discover.py tests/test_discover_queue.py
git commit -m "feat(discover): name the arena rank a skipped repo takes with it

hf_meta.EXCLUDE_PATTERNS is right to reject a quantization, but the rank
is real. Nemotron 3 Ultra's rank 42 vanished from the review queue with
no trace beyond a generic skip line."
```

---

## Final verification

- [ ] **Run everything**

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/render_readme.py
git diff --exit-code README.md && echo "README unchanged — correct"
```

Expected: all tests pass, `OK — 22 models`, and an unchanged README. That last
line is the honesty gate: this change arms the join for models already staged in
`candidates.yaml`, and must invent no number for a model upstream does not rate.

- [ ] **Confirm the two target cases would now resolve**

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "scripts")
import names
pairs = [("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-0731"),
         ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
          "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4")]
for a, b in pairs:
    ia, ib = names.repo_identity(a), names.repo_identity(b)
    print(f"{'JOIN' if ia == ib else 'SPLIT':5} {ia:24} {a}\n      {ib:24} {b}")
PY
```

Expected: both pairs print `JOIN`.

## Notes for the implementer

- **The arena and AA sidecars will not change on this branch.** Both scrapers
  need network access that is blocked here, and all three YAML data files are
  marked AUTO-SCRAPED / do-not-hand-edit. Task 3's fix lands as code and tests;
  `arena_agent_rankings.yaml` picks up the better Nemotron repo on the next
  networked run of `pull_arena.py` (weekly CI, or a local run).
- **`pull_aa.py` is deliberately untouched.** Its join has no reported defect —
  22 scores, zero orphans, no collisions — and re-proving all 22 matches needs a
  live AA scrape. Adopting `display_identity` there is deferred to its own spec.
- **If a test in Task 2 fails**, the move changed behaviour. Diff the moved
  function against `git show HEAD~1:scripts/pull_arena.py` rather than adjusting
  the test — those tests are the only proof the refactor is safe.
