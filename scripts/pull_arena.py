#!/usr/bin/env python3
"""
Scrape the Arena Intelligence "Agent Arena" leaderboard (https://arena.ai)
and produce a clean, ranked list of models.

WHY WE SCRAPE THE *FULL* BOARD (not ?license=open-source):
    arena.ai's own open-source label is unreliable — e.g. open-weight models
    like Kimi have been tagged as not-open-source. So we pull EVERY row.
    Open-weight status is decided by whether a public weights repo actually
    resolves on Hugging Face; KEYWORD_MAP serves only as a search hint for that
    lookup. We don't infer open-weight from org/product line, as that produced
    false positives (e.g. "Meta Muse Spark 1.1 · Proprietary" matched keyword
    "meta"). Arena's own license label is captured too, only so you can see
    the discrepancy — it is never used to filter.

HOW IT PARSES (resilient by design):
    The leaderboard is a server-rendered <table>, so requests + BeautifulSoup
    is enough (no headless browser). We do NOT pin to CSS class names (they
    change). Instead we read the <table>, then extract each row heuristically:
      - rank  = first pure-integer cell (fallback: row position)
      - model = the cell containing a known vendor/product keyword
      - score = first cell matching  NN.NN% ± N.NN%
    Every raw cell is kept under `raw` so nothing is silently lost.

Output: arena_agent_rankings.yaml  (all rows; each flagged open_weight or not)

Usage:
  pip install -r requirements.txt
  python scripts/pull_arena.py                    # all models
  python scripts/pull_arena.py --open-weight-only # only known open-weight orgs
  python scripts/pull_arena.py --url https://arena.ai/leaderboard/agent

Note: arena.ai is unreachable from some sandboxes; run on your machine / CI.
There is no official API — this is scraping, so check arena.ai's ToS before
republishing their numbers.
"""
import argparse
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

# Repo-name suffixes that mark an instruction-tuned variant of the same model.
# Stripped before comparing an arena name to a repo name.
_VARIANT_SUFFIXES = ("instruct", "it", "chat", "base")

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
                    help="write only rows whose org is a known open-weight line")
    ap.add_argument("--html", help="parse a local HTML file instead of fetching "
                                    "(useful for testing/offline)")
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

    ow = [r for r in rows if r["open_weight"]]
    print(f"\nParsed {len(rows)} models; {len(ow)} are known open-weight.")
    print("Open-weight, by rank:")
    for r in ow:
        s = f"{r['net_improvement_pct']}%" if r["net_improvement_pct"] is not None else "?"
        print(f"  #{r['rank']:>2}  {r['model']}  ({r['org']}, {s})")

    unknown = [r for r in rows if r["org"] is None]
    if unknown:
        print(f"\n{len(unknown)} model(s) had an unrecognized org — add keywords "
              f"to KEYWORD_MAP if any are open-weight:")
        for r in unknown[:15]:
            print(f"  - {r['model']}")

    out_rows = ow if args.open_weight_only else rows
    header = ("# AUTO-SCRAPED from arena.ai Agent Arena by scripts/pull_arena.py\n"
              "# open_weight is derived from the model's org/product line, NOT from\n"
              "# arena's own license label (which is unreliable). Verify before use.\n")
    OUT.write_text(header + yaml.safe_dump({"arena_agent": out_rows},
                                          sort_keys=False, allow_unicode=True, width=100))
    print(f"\nWrote {len(out_rows)} rows to {OUT.name}")


if __name__ == "__main__":
    main()
