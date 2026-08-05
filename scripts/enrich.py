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
