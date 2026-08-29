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
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_meta  # noqa: E402  (for auth_headers — see _http_get_text)

CARD_URL = "https://huggingface.co/{repo}/resolve/main/README.md"


def _http_get_text(url):
    """Default fetcher. Injected in tests so no test touches the network.

    Shares hf_meta.auth_headers so a configured token reaches model cards
    too. Cards on gated repos happen to be public today, but that is the
    vendor's choice and not a guarantee — and one header builder means the
    two fetchers cannot drift on how a token is read.
    """
    req = urllib.request.Request(url, headers=hf_meta.auth_headers("owlt-enrich/1.0"))
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")

# Vendor phrasings observed in real cards, most specific first.
#
# Two families. _PAIRED_PATTERNS capture a total AND its matching activation
# figure stated in one breath ("284B parameters (13B activated)"); they are
# what lets a card describing a whole model family be resolved against the
# row's own size instead of abandoned as ambiguous. _ACTIVE_PATTERNS capture
# an activation figure alone, for the cards that never restate the total
# beside it. Both are searched; a paired match also counts as an active one.
#
# Every pattern ends on the whole word activ(e|ated) so the quoted source
# reads as a sentence a reviewer can check, not a fragment cut mid-word.

_NUM = r"(\d+(?:\.\d+)?)\s*([BTM])"

_PAIRED_PATTERNS = (
    # "1.6T parameters (49B activated)" / "284B\nparameters (13B active)"
    re.compile(_NUM + r"\s*param\w*\s*\(\s*~?" + _NUM + r"\s*activ(?:e|ated)\b",
               re.I | re.S),
    # "117B parameters with 5.1B active parameters"
    re.compile(_NUM + r"\s*param\w*\s+with\s+~?" + _NUM + r"\s*activ(?:e|ated)\b",
               re.I | re.S),
    # "975B total, 41B active" / "397B in total and 17B activated"
    re.compile(_NUM + r"\s*(?:in\s+)?total[,\s]+(?:and\s+)?~?" + _NUM
               + r"\s*activ(?:e|ated)\b", re.I | re.S),
)

# The vendor's A-notation, written in prose ("a 30B-A3B MoE model") or simply
# echoed as the repo id inside the card. Kept in its own tier because the
# notation ROUNDS: Gemma 4 is "26B-A4B" in its name and 3.8B in its table, and
# both strings appear on the same page. Treating the two as rival claims made
# the rounded one win the tie-break and publish 4.0 for a 3.8B model, so a
# precise figure is always resolved first and this tier is consulted only when
# that yields nothing. Units are literal, so the groups still read
# (num, unit, num, unit).
_ROUNDED_PATTERNS = (
    re.compile(r"(\d+(?:\.\d+)?)\s*(B)[-_]A(\d+(?:\.\d+)?)\s*(B)\b"),
)

_ACTIVE_PATTERNS = (
    # "1.6T parameters (49B activated)"
    re.compile(r"\(\s*~?\s*" + _NUM + r"\s*activated\s*\)", re.I),
    # "~23B activated parameters" / "21B active parameters"
    re.compile(r"~?\s*" + _NUM + r"\s*activ(?:e|ated)\s+param", re.I),
    # "Activated Parameters</td><td>104B"
    re.compile(r"Activ(?:e|ated)\s+Parameters?.{0,120}?>\s*~?" + _NUM,
               re.I | re.S),
    # "| Activated Parameters | 104B |", and the same cell wrapped in markdown
    # emphasis: "| **Active Parameters** | 3.8B |". The stars are why Gemma 4
    # fell through for so long, so both \** are deliberate, not decoration.
    re.compile(r"Activ(?:e|ated)\s+Parameters?\**\s*\|\s*\**\s*~?" + _NUM, re.I),
    # "17B activated" / "41B active" with no noun following -- the phrasing
    # every Qwen card uses ("397B in total and 17B activated") and the one
    # thinkingmachines uses ("975B total, 41B active"). Last because it is the
    # loosest; the patterns above quote more context.
    re.compile(r"~?" + _NUM + r"\s*activ(?:e|ated)\b", re.I),
)

_UNIT_TO_B = {"b": 1.0, "t": 1000.0, "m": 0.001}

# How far a card's stated total may sit from the row's measured total and
# still be taken as describing the same model. Cards round and quote the
# headline figure ("744B") while safetensors counts every tensor (753.9B), so
# the gap is normally a few percent; a different variant of the same family is
# off by an order of magnitude. 15% separates those two cases with room to
# spare and is nowhere near tight enough to need tuning per vendor.
_TOTAL_MATCH_TOLERANCE = 0.15


def _to_b(value, unit):
    return round(float(value) * _UNIT_TO_B[unit.lower()], 1)


def _matches_total(stated, total_b):
    """Is a card's stated total the same model as one measuring total_b?"""
    if not total_b or stated <= 0:
        return False
    return abs(stated - total_b) / float(total_b) <= _TOTAL_MATCH_TOLERANCE


def _paired(patterns, text):
    return [(_to_b(m.group(3), m.group(4)), _to_b(m.group(1), m.group(2)),
             " ".join(m.group(0).split()))
            for pattern in patterns for m in pattern.finditer(text)]


def _card_claims(text):
    """(precise, rounded) claim lists of (active_b, total_b_or_None, quote)."""
    precise = _paired(_PAIRED_PATTERNS, text)
    for pattern in _ACTIVE_PATTERNS:
        for m in pattern.finditer(text):
            precise.append((_to_b(m.group(1), m.group(2)), None,
                            " ".join(m.group(0).split())))
    return precise, _paired(_ROUNDED_PATTERNS, text)


def _resolve(claims, total_b):
    """The winning claim from a tier, or None if the tier is ambiguous."""
    if not claims:
        return None
    if len({active for active, _, _ in claims}) == 1:
        return claims[0]
    plausible = [c for c in claims if _matches_total(c[1] or 0, total_b)]
    if len({active for active, _, _ in plausible}) == 1:
        return plausible[0]
    return None


def active_params_from_card(text, total_b=None, notes=None):
    """(billions, quoted_source) from a model card, or None if not stated.

    None is the important case: it routes the row to human review. Returning 0
    or a computed estimate would be worse than useless, because nothing
    downstream can tell an invented number from a published one.

    A card can document several variants in one README (e.g. a "Pro" and a
    "Flash" size, each with its own activated-parameter figure). If the card
    yields more than one DISTINCT figure it is ambiguous, and the fallback is
    the row's own measured total: a claim that also states a total within
    _TOTAL_MATCH_TOLERANCE of it is describing THIS model, and if exactly one
    variant qualifies the figure is no longer a guess. Two plausible variants,
    none plausible, or no total to compare against all stay None -- guessing
    which mention belongs to the requested repo would silently attribute the
    wrong model's number.

    Repeated mentions of the SAME figure are not a conflict and must still
    resolve normally, so every pattern is searched across the WHOLE text (not
    just its first hit) before any decision is made.
    """
    if not text:
        return None
    precise, rounded = _card_claims(text)
    claim = _resolve(precise, total_b)
    if claim is None:
        claim = _resolve(rounded, total_b)
    if claim is None:
        return None
    active, stated_total, quote = claim
    _record_stated_total(notes, stated_total)
    return active, quote


def _record_stated_total(notes, stated_total):
    """Report the TOTAL the winning quote asserts, when it asserts one.

    The `notes` out-parameter is the same seam hf_meta.fetch_config and
    context_from_tokenizer already use, chosen so the return shape stays a
    plain (billions, quote) pair for every existing caller.

    This exists because a quote like "284B parameters (13B activated)" is
    evidence about two numbers, and storing it beside a params_total_b of
    290.9 made the record contradict itself in 13 published rows. Recorded
    only when the quote actually names a total: "~23B activated parameters"
    names none, and inventing one would fabricate a vendor claim.
    """
    if notes is not None and stated_total is not None:
        notes["stated_total"] = stated_total


# "Qwen3-VL-235B-A22B-Instruct", "gemma-4-26B-A4B-it": the vendor states the
# split in the repo id itself. Both halves are required -- a bare "A22" would
# match an accelerator name or a revision tag, and "17B-128E" (Llama 4's expert
# count) must not be read as an activation figure.
_REPO_A_NOTATION = re.compile(r"(\d+(?:\.\d+)?)B[-_]A(\d+(?:\.\d+)?)B\b", re.I)


def active_params_from_repo_name(repo, total_b=None, notes=None):
    """(billions, quoted_source) from the repo id's A-notation, or None.

    A last resort, tried only after the card. Qwen3-VL-235B-A22B states its
    activation figure nowhere in its card, and the naming convention is the
    vendor's own published statement about the model -- not an inference from
    architecture, which this module must never make.

    The name encodes the TOTAL as well, which is the safety check: if that
    disagrees with the row's measured total the name is describing a different
    model (a distill, a mirror, a renamed re-release) and its activation
    figure cannot be carried onto this row. The card is preferred wherever it
    speaks, because the name rounds and the card does not -- Gemma 4 is
    "26B-A4B" in its id and 3.8B in its table.
    """
    if not repo:
        return None
    m = _REPO_A_NOTATION.search(str(repo))
    if not m:
        return None
    stated_total = _to_b(m.group(1), "b")
    if total_b and not _matches_total(stated_total, total_b):
        return None
    _record_stated_total(notes, stated_total)
    return _to_b(m.group(2), "b"), f"repo name: {m.group(0)}"


def fetch_card(repo, get_text=_http_get_text):
    """The repo's README.md, or None on any failure."""
    try:
        return get_text(CARD_URL.format(repo=repo))
    except Exception:
        return None


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


TOKENIZER_URL = "https://huggingface.co/{repo}/resolve/main/tokenizer_config.json"

# Transformers writes this when model_max_length is unset. It is not a context
# length; treating it as one would publish a 1e30-token window.
_SENTINEL_MAX_LENGTH = 1_000_000_000_000_000


def context_from_tokenizer(repo, get_json, notes=None):
    """Context length from tokenizer_config.json, or None.

    Tried only after config.json has already failed — 80 discovered rows had
    no length in either the API expand or config.json, and this is where the
    remainder publish it.

    `notes` works exactly as in hf_meta.fetch_config: an access failure sets
    notes["gated"] = True. A gated repo 403s here for the same reason it does
    on config.json, so this reports the access problem at no extra request.
    """
    try:
        cfg = get_json(TOKENIZER_URL.format(repo=repo))
    except Exception as exc:
        if notes is not None and hf_meta.is_gated_error(exc):
            notes["gated"] = True
        return None
    if not isinstance(cfg, dict):
        return None
    n = cfg.get("model_max_length")
    if isinstance(n, int) and not isinstance(n, bool) and 0 < n < _SENTINEL_MAX_LENGTH:
        return n
    return None
