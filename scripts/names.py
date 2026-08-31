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


_PAREN_RE = re.compile(r"\(([^)]*)\)")


def split_variant(display):
    """('Kimi K3 (max)') -> ('Kimi K3', 'max'). No parenthetical -> 'default'.

    Leaderboards append the reasoning effort a score was measured at, and the
    parenthetical is not part of the model's identity. It lives here because
    both the AA scraper and the AA join have to strip it the same way: the
    join re-derives its keys from the display names the scraper wrote, so a
    disagreement about what "Kimi K3 (max)" slugs to means nothing matches.
    """
    found = _PAREN_RE.search(display)
    variant = found.group(1).strip().lower() if found else "default"
    base = _PAREN_RE.sub(" ", display)
    base = re.sub(r"\s+", " ", base).strip()
    return base, variant


def slug(text):
    """Lowercase and strip every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def strip_variant_suffix(text):
    """Drop one trailing variant suffix. 'Llama 3.3 70B Instruct' -> 'Llama 3.3 70B'.

    The separator is required, so a word that merely ends in the letters of a
    suffix ("Summit") is untouched.
    """
    return _SUFFIX_RE.sub("", text)


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


# Reasoning-effort settings arena's TEXT leaderboard (open-source view) glues
# directly onto the model slug with a hyphen/underscore, e.g. "glm-5.2-max",
# "kimi-k3-max" -- unlike the older Agent board, which wrote the same thing
# in parentheses ("GLM 5.2 (Max)"), already handled below. Both are
# inference-time settings, not different weights, so both get stripped.
#
# Deliberately narrower than it looks -- every exclusion below is a case
# where stripping would be worse than leaving the row unresolved, because a
# missing match sends a row to review while a WRONG match publishes the
# wrong weights under the model's name:
#   - "thinking" is NOT here: moonshotai/Kimi-K2-Thinking is a real, separate
#     repo. Stripping it would resolve "kimi-k2.5-thinking" to the (wrong)
#     non-thinking Kimi-K2.5 weights instead of correctly staying unresolved.
#   - "preview" is NOT here for the same reason: it can denote a genuinely
#     distinct release ("deepseek-v4-pro-high-preview" is on the board) and
#     must stay unresolved rather than be trimmed down to a guess.
#   - "instruct"/"it"/"chat"/"base" are NOT here: VARIANT_SUFFIXES already
#     handles those downstream in the matcher; duplicating the list here
#     would just be two places to keep in sync.
EFFORT_TOKENS = ("max", "high", "medium", "low", "minimal", "xhigh")

# Anchored to a HYPHEN/UNDERSCORE attachment, never a plain space. That is
# the load-bearing distinction: the new board glues the tag onto the slug
# with a hyphen ("glm-5.2-max"), while a legitimate product name can be a
# separate, space-delimited word that happens to be one of these words --
# Alibaba's "Qwen3.7 Max" is a real proprietary tier, not a reasoning knob,
# and the old board always renders it space-separated. Adding \s here would
# silently start truncating names like that (see test_normalize_model_name
# and test_space_separated_effort_word_is_a_real_product_name_not_a_tag).
_EFFORT_RE = re.compile(
    r"[-_](?:" + "|".join(EFFORT_TOKENS) + r")$", re.IGNORECASE)


def normalize_display(display):
    """Strip leaderboard chrome from a display label.

    "GLM 5.2 (Max) Z.ai · MIT · SiliconFlow" -> "GLM 5.2"
    "glm-5.2-max Z.ai · MIT" -> "glm-5.2"

    Four steps: drop everything from the first "·" separator, drop
    parenthetical effort tags like "(High)"/"(Max)", drop a single trailing
    org display name, then drop a hyphen/underscore-attached trailing effort
    token (see EFFORT_TOKENS) -- the newer slug-style board's equivalent of
    the parenthetical form.
    """
    name = display.split("·")[0]
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    tokens = name.split(" ")
    if len(tokens) > 1 and tokens[-1].lower() in ORG_DISPLAY_ALIASES:
        tokens = tokens[:-1]
    name = " ".join(tokens).strip()
    return _EFFORT_RE.sub("", name)


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


def _identity_parts(repo_id):
    """Repo tail split into tokens, author prefix and trailing noise removed.

    Shared by repo_identity and family_stem so the two keys always agree on
    what counts as noise; they differ only in whether versions survive.
    """
    author, _, tail = repo_id.rpartition("/")
    parts = [p for p in re.split(r"[-_.]", tail) if p]

    if author and parts and parts[0].lower() == author.split("/")[-1].lower():
        parts = parts[1:]

    while len(parts) > 1 and (parts[-1].lower() in PRECISION_TOKENS
                              or parts[-1].lower() in VARIANT_SUFFIXES
                              or DATE_TOKEN.match(parts[-1])):
        parts.pop()
    return parts


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
    return slug("-".join(_identity_parts(repo_id)))


# A token that is a version, not a size: an optional leading letter then digits.
# "4", "5", "K3", "V4" match; "405B", "17B", "16E", "A22B" do not,
# because they end in a letter and therefore denote a size or an expert count.
_VERSION_TOKEN = re.compile(r"^[A-Za-z]?\d+$")

# A version fused directly onto the family name with no separator to split on,
# e.g. "Qwen3" ("Qwen/Qwen3-235B-A22B" — HF writes no hyphen before the
# generation digit, unlike Llama's "Llama-4-Scout"). Requires letters BEFORE
# the digits, which is what keeps it from ever matching a size/expert token:
# those end in a letter ("405B", "16E"), never in a digit.
_FUSED_VERSION = re.compile(r"^([A-Za-z]+?)\d+$")


def family_stem(repo_id):
    """A model FAMILY key: repo_identity with version tokens removed.

    Deliberately coarser than repo_identity and deliberately finer than a bare
    vendor name. repo_identity keeps versions, so it can never detect that
    GLM-5.2 supersedes GLM-5.1. Stripping more than versions would be worse:
    dropping sizes collapses Llama-3.1-405B onto Llama-3.1-8B, and dropping
    words collapses Llama-4-Scout onto Llama-4-Maverick — all four are
    legitimately tracked as separate rows.

    So this fires on exactly one shape: a version bump at the same size, which
    is the supersede-or-coexist call a human should make. That shape includes
    a version fused onto the family name itself (Qwen2 -> Qwen3): Qwen ships
    generation bumps that way, and missing them would defeat the point of this
    key for one of the most active open-weight vendors.

    LIMITATION: Collisions in family_stem indicate that a human should review
    whether the models are truly the same family line. The heuristic cannot
    distinguish a version marker from a product-line marker — both are a single
    letter followed by digits (e.g. "V3", "R1"). DeepSeek-V3 and DeepSeek-R1
    are different product lines (base/chat vs reasoning), not versions of each
    other, yet both produce the stem "deepseek". Collision is not an error; it
    is a signal to check. While the collision exists, no DeepSeek model can
    auto-promote based on family_stem matching.
    """
    kept = []
    for part in _identity_parts(repo_id):
        if _VERSION_TOKEN.match(part):
            continue
        fused = _FUSED_VERSION.match(part)
        kept.append(fused.group(1) if fused else part)
    return slug("-".join(kept))


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
