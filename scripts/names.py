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
