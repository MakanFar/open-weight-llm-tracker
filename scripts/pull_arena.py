#!/usr/bin/env python3
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
import argparse
import os
import re
import sys
from pathlib import Path

import yaml

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Install deps first:  pip install -r requirements.txt")

try:
    from jsonschema import validate as _js_validate, ValidationError
except ImportError:
    _js_validate = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names
slug = names.slug
_VARIANT_SUFFIXES = names.VARIANT_SUFFIXES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "arena_agent_rankings.yaml"
DEFAULT_URL = "https://arena.ai/leaderboard/agent"

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

# Vendor names that appear at the START of an arena display label. Sorted
# longest-first at use so a multi-word vendor beats its first word.
_LEADING_ORG_PHRASES = (
    "thinking machines", "meta", "anthropic", "google", "openai", "nvidia",
    "microsoft", "alibaba", "tencent", "xiaomi", "baidu", "mistral", "cohere",
)

# Repo-slug decorations that carry no identity: parameter counts (550B),
# active-param counts (A55B), and weight precisions. A repo differing from
# the arena name only by these is the same model, so they are stripped before
# comparison — "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" -> "Nemotron-3-Ultra".
_SIZE_TOKEN = re.compile(r"^[aA]?\d+(?:\.\d+)?[bBmM]$")
_PRECISION_TOKENS = {
    "bf16", "fp16", "fp32", "fp8", "int4", "int8", "nvfp4", "mxfp8",
    "w4a16", "w8a8", "4bit", "8bit", "gguf", "awq", "gptq",
}

SCORE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%\s*(?:±|\+/-|\+-|\+\-)?\s*(\d+(?:\.\d+)?)?\s*%?")
INT_RE = re.compile(r"^\s*#?\s*(\d{1,3})\s*$")

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


def derive_org(name):
    """Return (org, matched_keyword) from a model display name.

    Makes no open-weight claim — that is decided by repo resolution.
    """
    low = name.lower()
    for kw, org in KEYWORD_MAP.items():
        if re.search(rf"\b{re.escape(kw)}", low):
            return org, kw
    return None, None


def parse_score(text):
    m = SCORE_RE.search(text)
    if not m:
        return None, None
    val = float(m.group(1))
    ci = float(m.group(2)) if m.group(2) else None
    return val, ci


def looks_like_model(text):
    return derive_org(text)[1] is not None or bool(re.search(r"[A-Za-z]{3,}", text))


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


def without_leading_vendor(name):
    """Drop a leading vendor phrase: 'Thinking Machines Inkling' -> 'Inkling'.

    Only score_match uses this. normalize_model_name keeps the vendor because
    that string is the human-readable label printed in reports; it is the
    matcher that has to tolerate a repo slug carrying just the model name.
    Longest phrase first, so "thinking machines" beats "thinking".
    """
    tokens = name.split(" ")
    for phrase in sorted(_LEADING_ORG_PHRASES, key=len, reverse=True):
        parts = phrase.split(" ")
        if len(tokens) > len(parts) and \
                [t.lower() for t in tokens[:len(parts)]] == parts:
            return " ".join(tokens[len(parts):]).strip()
    return name


# Minimum length ratio (shorter slug / longer slug) for a prefix relation to
# count as "medium". A bare prefix test is far too weak: "Qwen3" is a prefix of
# "Qwen37Max", "grok" of "grok45", "gpt" of "gpt56sol" — so every proprietary
# flagship matched some unrelated community repo whose name happened to start
# with the vendor's brand, and minted open_weight: true for it.
#
# A ratio buys precision by rejecting short-name collisions ("Grok 4.5" vs "grok",
# 4/6 = 0.67, rejected), but trades recall: repos whose names extend the family
# name with a size/date/variant suffix (e.g. "Llama 4 Maverick" vs
# "meta-llama/Llama-4-Maverick-17B-128E-Instruct", 12/36 = 0.33) also score
# "low" and are reported as unresolved rather than mis-resolved. Token-level
# matching (keeping "Llama" and "4" and "Maverick" distinct) would address this
# and is deliberately deferred pending a clearer requirement shape.
_MIN_PREFIX_RATIO = 0.7


def strip_repo_decorations(repo_id):
    """Reduce a repo id to its identifying name.

    Drops the author, a duplicated vendor prefix ("nvidia/NVIDIA-Nemotron-…"),
    then trailing size/precision/variant tokens. Only decorations go: a token
    like "Nano" or "Ultra" distinguishes two real models and is kept, so
    Nemotron-3-Ultra and Nemotron-3-Nano do not collapse together.
    """
    author, _, tail = repo_id.rpartition("/")
    parts = [p for p in re.split(r"[-_.]", tail) if p]

    if author and parts and parts[0].lower() == author.split("/")[-1].lower():
        parts = parts[1:]

    return "-".join(_strip_decorations(parts))


def _strip_decorations(parts):
    """Drop trailing size/precision/variant tokens from a token list."""
    while parts:
        last = parts[-1].lower()
        if last in _PRECISION_TOKENS or last in _VARIANT_SUFFIXES \
                or _SIZE_TOKEN.match(parts[-1]):
            parts.pop()
            continue
        break
    return parts


def score_match(query, repo_id):
    """Rate how well an arena name matches an HF repo id: high/medium/low.

    Decorations are stripped from BOTH sides. Stripping only the repo breaks
    the symmetric case: "Gemma 4 31B" vs "google/gemma-4-31b-it" would compare
    "gemma431b" against "gemma4" and score low.
    """
    r = slug(strip_repo_decorations(repo_id))
    for candidate in (query, without_leading_vendor(query)):
        q = slug("-".join(_strip_decorations(
            [p for p in re.split(r"[-_.\s]", candidate) if p])))
        conf = _rate(q, r)
        if conf != "low":
            return conf
    return "low"


def _rate(q, r):
    if not q or not r:
        return "low"
    if q == r:
        return "high"
    if q.startswith(r) or r.startswith(q):
        # A prefix relation only means something if the two names are of
        # comparable length — see _MIN_PREFIX_RATIO above.
        if min(len(q), len(r)) / max(len(q), len(r)) >= _MIN_PREFIX_RATIO:
            return "medium"
    return "low"


# Canonical org name -> HF author namespace, used to scope the repo search.
#
# An org with NO entry here searches all of Hugging Face unscoped, which is the
# worst case for precision: the top hits for "grok" or "gpt" are community
# re-uploads and look-alikes, not the vendor's own repos. So the mostly-closed
# vendors belong here too, which is why a map of HF namespaces contains names
# you would not expect to find open weights under:
#   - OpenAI and xAI do publish *some* open weights (gpt-oss, grok-1), so they
#     scope to their real namespaces and their proprietary models simply fail
#     to match anything inside them.
#   - Anthropic publishes no weights at all and holds no HF namespace. It maps
#     to None, an explicit "never search" — without it, every Claude row would
#     trigger an unscoped search across all of HF.
# None therefore means "known vendor, no HF namespace"; a missing key means
# "org we do not track yet" and is what feeds the new-orgs report in main().
HF_AUTHOR_HINTS = {
    "Google": "google", "Meta": "meta-llama", "Alibaba": "Qwen",
    "DeepSeek": "deepseek-ai", "Moonshot AI": "moonshotai",
    "Zhipu AI": "zai-org", "MiniMax": "MiniMaxAI", "NVIDIA": "nvidia",
    "Microsoft": "microsoft", "Cohere": "CohereLabs", "Mistral AI": "mistralai",
    "01.AI": "01-ai", "TII": "tiiuae", "IBM": "ibm-granite", "Ai2": "allenai",
    "Baidu": "baidu", "Xiaomi": "XiaomiMiMo",
    "Thinking Machines": "thinkingmachines",
    "OpenAI": "openai", "xAI": "xai-org",
    "Anthropic": None,
}


def resolve_row(row, search_fn):
    """Resolve an arena row to an HF repo. Mutates and returns the row.

    Open-weight status is decided HERE, and only here: a model is open-weight
    if and only if public weights were found. arena's own license label and
    KEYWORD_MAP are never used to make that call.
    """
    query = normalize_model_name(row["model"])
    org = row.get("org")

    searched, repo_ids = True, []
    if org in HF_AUTHOR_HINTS and HF_AUTHOR_HINTS[org] is None:
        # Vendor is known to publish no weights on HF (see HF_AUTHOR_HINTS).
        # Searching anyway would scan all of HF and surface look-alikes.
        pass
    else:
        author = HF_AUTHOR_HINTS.get(org)
        try:
            repo_ids = search_fn(query, author) or []
        except Exception as exc:      # rate limit, network, API change
            print(f"  ! search failed for {query!r}: {exc}")
            searched = False

    if not searched:
        # A failed lookup is NOT evidence of a closed model. Writing
        # open_weight: False here is how a 429 demoted Kimi K3 — which has
        # public, ungated weights — to proprietary in committed data. Leave it
        # unknown and let carry_forward_resolutions() restore what we knew.
        row["resolved_repo"] = None
        row["resolution_confidence"] = None
        row["open_weight"] = None
        row["needs_hf_repo"] = True
        row["search_failed"] = True
        return row

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


def unmapped_orgs(rows):
    """Orgs seen on the leaderboard that this repo has no HF namespace for.

    Computed over EVERY parsed row, not just the ones that resolved. That
    distinction is the whole point: an org with no HF_AUTHOR_HINTS entry
    searches HF unscoped and is therefore *less* likely to resolve, so keying
    this off resolved rows reported the orgs we already handle well and stayed
    silent about exactly the ones a human needs to add.

    A key mapped to None (vendor known to publish no weights — see
    HF_AUTHOR_HINTS) is deliberate coverage, not a gap, so it is not reported.

    We test membership against HF_AUTHOR_HINTS alone and do not import
    ORG_ALLOWLIST from discover.py: discover.py imports huggingface_hub at
    module scope, so importing it here would make `--no-resolve` (the offline
    parse-only path) require the HF client. The two lists are instead kept in
    step by a test that asserts every namespace here is swept there.
    """
    return sorted({r["org"] for r in rows
                   if r.get("org") and r["org"] not in HF_AUTHOR_HINTS})


_RESOLUTION_FIELDS = ("resolved_repo", "resolution_confidence",
                      "open_weight", "needs_hf_repo")


def carry_forward_resolutions(rows, previous):
    """Restore the last known resolution for rows whose search failed.

    Keyed on the normalized model name, not rank — ranks move between runs.
    Only rows flagged search_failed are touched, so a fresh result always
    wins over history; a failed row with no history simply stays unknown.
    """
    by_name = {}
    for old in previous or []:
        if isinstance(old, dict) and old.get("model") and old.get("resolved_repo"):
            by_name.setdefault(slug(normalize_model_name(old["model"])), old)

    for row in rows:
        if not row.get("search_failed"):
            continue
        old = by_name.get(slug(normalize_model_name(row.get("model", ""))))
        if not old:
            continue
        for field in _RESOLUTION_FIELDS:
            row[field] = old.get(field)
        row["carried_forward"] = True
    return rows


def load_previous(path=OUT):
    """Rows from the committed arena file, for carry-forward. [] on any problem."""
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return []
    rows = doc.get("arena_agent") if isinstance(doc, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def resolve_all(rows, search_fn, previous=None):
    for row in rows:
        resolve_row(row, search_fn)
    carry_forward_resolutions(rows, previous)
    return rows


def hf_search_fn(api):
    """Adapt HfApi into the search_fn contract: (query, author) -> [repo_id]."""
    def _search(query, author):
        kwargs = {"search": query, "limit": 10}
        if author:
            kwargs["author"] = author
        return [m.id for m in api.list_models(**kwargs)]
    return _search


def parse_leaderboard(html):
    """Return list of row dicts. Header-driven where possible, heuristic always."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    rows = []

    tr_list = []
    headers = []
    if table:
        head = table.find("thead")
        if head:
            headers = [c.get_text(" ", strip=True)
                       for c in head.find_all(["th", "td"])]
        body = table.find("tbody") or table
        tr_list = body.find_all("tr")
    else:
        # fallback: ARIA grid (role=row / role=cell)
        grid_rows = soup.select('[role="row"]')
        tr_list = grid_rows

    for i, tr in enumerate(tr_list):
        cells = [c.get_text(" ", strip=True)
                 for c in tr.find_all(["td", "th"])
                 if c.get_text(strip=True)]
        if not cells:
            cells = [c.get_text(" ", strip=True)
                     for c in tr.select('[role="cell"], [role="gridcell"]')
                     if c.get_text(strip=True)]
        if not cells:
            continue
        # skip a header row that slipped into tbody
        if headers and cells == headers:
            continue

        # --- heuristic field extraction (resilient to column reordering) ---
        rank = None
        for c in cells:
            m = INT_RE.match(c)
            if m:
                rank = int(m.group(1))
                break
        if rank is None:
            rank = len([r for r in rows]) + 1

        # model = first cell that maps to a known keyword, else the longest
        # mostly-alphabetic cell that isn't a pure number/percentage
        model = None
        for c in cells:
            if derive_org(c)[1] is not None:
                model = c
                break
        if model is None:
            cand = [c for c in cells
                    if not INT_RE.match(c) and "%" not in c and len(c) > 2]
            model = max(cand, key=len) if cand else (cells[1] if len(cells) > 1 else cells[0])

        pct, ci = None, None
        for c in cells:
            if "%" in c:
                pct, ci = parse_score(c)
                if pct is not None:
                    break

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

    # de-dup + stable sort by rank
    seen, deduped = set(), []
    for r in rows:
        key = (r["rank"], r["model"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    deduped.sort(key=lambda r: r["rank"])
    return deduped


def validate_rows(rows):
    if _js_validate is None:
        print("  (jsonschema not installed — skipping row validation)")
        return
    bad = 0
    for r in rows:
        try:
            _js_validate(r, ROW_SCHEMA)
        except ValidationError as e:
            bad += 1
            print(f"  ! row failed schema: {r.get('model')}: {e.message}")
    if not bad:
        print(f"  schema OK — {len(rows)} rows valid")


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
        resolve_all(rows, hf_search_fn(HfApi(token=token)), load_previous())
        failed = [r for r in rows if r.get("search_failed")]
        carried = [r for r in failed if r.get("carried_forward")]
        if failed:
            print(f"  {len(failed)} search(es) failed; {len(carried)} restored "
                  f"from the committed file, {len(failed)-len(carried)} left unknown")

    ow = [r for r in rows if r.get("open_weight")]
    print(f"\nParsed {len(rows)} models; {len(ow)} resolved to a weights repo.")
    for r in ow:
        s = f"{r['net_improvement_pct']}%" if r["net_improvement_pct"] is not None else "?"
        flag = "  [VERIFY]" if r.get("needs_hf_repo") else ""
        print(f"  #{r['rank']:>2}  {r['model'][:45]:<45} -> {r['resolved_repo']} ({s}){flag}")

    new_orgs = unmapped_orgs(rows)
    if new_orgs:
        print(f"\nOrgs seen on the leaderboard with no HF namespace mapping: "
              f"{', '.join(new_orgs)}")
        print("Add each to HF_AUTHOR_HINTS here (so its models resolve against "
              "the right namespace) and its HF org to ORG_ALLOWLIST in "
              "scripts/discover.py (so the sweep covers it).")

    out_rows = ow if args.open_weight_only else rows
    header = ("# AUTO-SCRAPED from arena.ai Agent Arena by scripts/pull_arena.py\n"
              "# open_weight means: a public weights repo was FOUND on Hugging Face.\n"
              "# It is not arena's license label and not an org guess.\n"
              "# needs_hf_repo=true means the match was inexact — verify by hand.\n")
    OUT.write_text(header + yaml.safe_dump(
        {"arena_agent": out_rows, "new_orgs": new_orgs},
        sort_keys=False, allow_unicode=True, width=100))
    print(f"\nWrote {len(out_rows)} rows to {OUT.name}")


if __name__ == "__main__":
    main()
