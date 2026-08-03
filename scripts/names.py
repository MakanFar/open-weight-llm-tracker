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
