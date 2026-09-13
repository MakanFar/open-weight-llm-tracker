---
name: fill-review-gaps
description: Use when working the candidates.yaml review queue to fill the two gap types that are answerable from a document — moe-active-params-unknown and license-not-allowlisted. Runs unattended in discover.yml after a discovery sweep, and by hand against a local queue. Never decides supersede-or-coexist, never judges whether an old release is still worth carrying, never edits models.yaml.
---

# Fill the review gaps that a document can answer

You are filling **facts**, not making **decisions**. Those are different jobs
and this skill is only the first one.

## Scope: exactly two reasons, and nothing else

Work only on rows whose `needs_review` contains:

- **`moe-active-params-unknown`** — the row is MoE but `params_active_b`
  still equals `params_total_b`, the "activation unknown" sentinel.
- **`license-not-allowlisted`** — `license` is a string `validate.LICENSES`
  does not recognise, usually a vendor licence or a bare `other`.

**Every other reason is out of scope. Do not touch those rows' other fields
and do not clear those reasons.** Specifically:

| reason | why it is not yours |
|---|---|
| `duplicates-tracked-row` | Whether two repo ids naming the same weights are really one row is a decision about what the index *means*. No document answers it. |
| `signal-too-stale` | Exists precisely to make a human confirm an old release is still worth carrying. Clearing it automatically deletes the check. |
| `derivative-or-base` | Same shape: a judgement about what belongs in the index. |
| `gated-repo-no-access` | Not a research problem. The HF token's account has not accepted the vendor's terms; someone must click accept. |
| `inexact-repo-match` | A human confirms an identity claim before the tracker publishes it. |
| `schema-invalid: …` | Report it, do not paper over it. It usually means the pipeline mis-derived something. |

## The one rule that matters

**Your knowledge directs the search. It is never the evidence.**

This is the same rule as [verify-commercial-use](../verify-commercial-use/SKILL.md),
and it is here for the same reason. You may believe you know what
DeepSeek-V4.1-Flash activates. That belief is a starting point for where to
look, never a value to write. Every field you set cites a URL you actually
fetched and quotes the text it came from.

**Abstention is a finished, correct outcome.** A row you leave alone stays in
the queue and a human handles it — that costs one review. A row you fill
wrongly is *indistinguishable in the data from a checked one*, renders as
fact in a published table, and nobody ever finds it. `validate.py` will not
catch you: it checks that `params_active_b` is a positive number, not that it
is the right one.

## Hard constraints

1. **Never edit `models.yaml`.** It is append-only and owned by
   `discover.py`. The workflow verifies this with a checksum and fails the
   job if it moved. Your edits go to `candidates.yaml` only.
2. **Never edit `needs_review` yourself.** `scripts/refresh_review.py` runs
   after you and re-derives it from the facts you wrote. If you clear a
   reason by hand you are asserting a verdict rather than supplying a fact.
3. **Never edit code.** Not `validate.py`, not `classify.py`, not the
   allowlist. See the licence section for why that matters.
4. **Never commit or push.** Leave your changes in the working tree; the
   workflow's PR step collects them.
5. **Never promote anything.** You do not move rows into `models.yaml`. A row
   you unblock is promoted by the *next* discovery run, after a human has
   seen your citation in the PR diff. That ordering is the point.

## `moe-active-params-unknown`

**You are the reviewer.** Not a card-reader that gives up when the card is
quiet. The card is one source and usually not the best one: measured over the
real queue, **22 of 25** blocked rows had cards stating no activation figure at
all, while their vendors had published the number in a tech report, a GitHub
README or a release post. Stopping at the card sent all 22 to a human who
would have run the same search.

### 1. Establish the row's own total first

Everything downstream is an attribution problem, so anchor it before looking
anywhere. `params_total_b` is HF's measured safetensors count for **this
repo**. A claim belongs to this row only if the total stated alongside it
lands within about 15% of that figure — the same `_TOTAL_MATCH_TOLERANCE`
`enrich.active_params_from_card()` uses, loose because cards round to a
headline (`744B`) while safetensors counts every tensor (`753.9B`).

**A figure with no total stated beside it cannot be attributed and is not
evidence.** This is the rule that makes searching safe. Without it, widening
the search widens the damage, because the open web is full of confident
sentences about a model's siblings.

### 2. Look in this order, and stop at the first tier that answers

| tier | source | strength |
|---|---|---|
| 1 | The row's own card, figure paired with a matching total | Definitive. `enrich` already tried this and failed, so expect to go further. |
| 2 | `config.json` in the same repo | Definitive for *architecture* — it is the artifact, not a claim about it. See §3 for what that can and cannot settle. |
| 3 | The vendor's tech report, arXiv paper or GitHub README **for this release** | Strong. The same vendor stating the same fact somewhere with more room than a card. |
| 4 | The vendor's release post or announcement naming this exact model | Strong when it names the repo or the size; weak when it says only "our new model". |
| 5 | Third-party write-ups, news, leaderboard blurbs | Corroboration only, never the sole basis — they routinely copy a sibling's number. |

Searches that work: the repo tail plus `activated parameters`; the vendor and
model name plus `total parameters active`; the model name plus `technical
report`. Research **per family, not per row** — one DeepSeek-V3 tech report
answers every V3.x row whose measured total it matches.

### 3. What `config.json` settles, and what it does not

It settles **whether two repos are the same architecture**. When
`num_local_experts`, `num_experts_per_tok`, `hidden_size`,
`num_hidden_layers` and `intermediate_size` all match a model whose
activation is documented, at the same measured total, the activation is
necessarily the same and you may carry the documented figure across — citing
**both** the config comparison and the document the figure came from.
MiniMax-M2.5 and M2.7 are exactly this shape: routing byte-identical to M2 at
the same 228.7B, with M2's own card stating 10B.

It does **not** settle the figure on its own. **Never compute activated
parameters from expert counts and dimensions.** Shared experts, dense prefix
layers and attention parameters all enter the sum, vendors round differently,
and a derived number carries no citation anyone can re-check. If no document
anywhere states a figure, abstain — a computed number is the one thing this
skill must never publish.

### 4. Decide

| finding | action |
|---|---|
| One figure, stated total matches this row | Write it |
| Several figures, exactly one whose stated total matches this row | Write that one |
| Several figures, none whose total matches this row | **Abstain** — the document describes siblings |
| Two figures for this same model (`8B / 16B`, prefill/decode) | **Abstain** — the schema holds one float. Not yours to pick; see *Maintainer decisions* below |
| A figure with no total stated near it | **Abstain** — unattributable |
| Nothing found across tiers 1–5 | **Abstain**, and name where you looked |

### 5. Maintainer decisions are recorded as such

Some rows have no single right answer and no amount of searching produces one.
`deepseek-ai/DeepSeek-V4.1-Flash` publishes `# Activated Params | 8B / 16B`,
prefill against decode, and the schema holds one float. **Picking one is not
research and is not yours to do** — present both figures and what turns on the
choice, and let the maintainer decide.

When they do, say so **in those words** at the front of the field:

```yaml
    params_active_b: 16.0
    params_active_source: "maintainer decision, not a published figure. <URL> states TWO activation figures for this one model - ... - and params_active_b holds one float. The maintainer chose the DECODE figure: <their reasoning>. The vendor has not published a single combined figure."
```

This is the same rule as
[verify-commercial-use](../verify-commercial-use/SKILL.md)'s "maintainer
decision" wording, and it exists for the same reason: without it, the row is
indistinguishable from one where somebody read the figure off a card. A reader
six months later must be able to tell a sourced number from a chosen one, and
`validate.py` will never tell them — it checks only that the value is
positive.

### 6. Write it

`params_active_source` carries the provenance, and its *shape* says where the
number came from.

```yaml
    params_active_b: 10.0
    params_active_source: "230 billion total parameters with 10 billion active parameters"
```

```yaml
    params_active_b: 37.0
    params_active_source: "https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README.md - \"671B total parameters with 37B activated for each token\"; matches this row's measured 684.5B within 2%"
```

**A bare quote means the row's own card.** That is what `discover.enrich_row`
writes, and its meaning must not shift under it. **A value beginning with a
URL means somebody went looking**, and it must carry the URL you actually
fetched, what it said, and why it attaches to *this* row. Never "verified via
web search" — the next person has to re-check it without repeating your
search.

When you carried a figure across identical configs, say both halves:

```yaml
    params_active_source: "https://huggingface.co/MiniMaxAI/MiniMax-M2 - \"230 billion total parameters with 10 billion active parameters\"; this repo's config.json is identical to M2's (256 experts, 8 per token, 62 layers, hidden 3072) at the same 228.7B measured total"
```

## `license-not-allowlisted`

Follow [verify-commercial-use](../verify-commercial-use/SKILL.md) — it is the
established workflow and its evidence rule is the one above. Research the
licence from the card's `license_link`, the vendor's licence page, their
GitHub repo, and web search. Research **per vendor, not per row**: six of
these rows are `tongyi-qianwen`, and that is one determination, not six.

Write `license`, `commercial_use` (`true` / `false` / `conditional`) and a
`license_notes` that cites the URL and quotes the operative clause, replacing
the `AUTO-DISCOVERED` placeholder.

**If the licence you resolve is not in `validate.LICENSES`, stop there.**
Do not add it to the allowlist. Whether the tracker accepts a new licence
string is a policy decision about what the index carries, and CLAUDE.md
requires the allowlist change to land in the same reviewed change as the data.
Leave the row flagged and say plainly in `license_notes` what you resolved it
to, so the human making the allowlist call has your research in hand.

A bare `other` that resolves to a licence already in the allowlist is the easy
case and the one most worth doing: it needs no policy call at all.

## Finish

Report, per row you touched: the repo, the field you set, the URL you fetched,
and the text it said. Then report the rows you deliberately left alone and
why.

**That second list is the useful half.** A run that resolves 9 of 25 and names
the other 16 is a success. A run that resolves 25 is a claim to be suspicious
of — searching hard enough to find *a* number for every row means you have
started attributing siblings' figures, which is the one failure this whole
procedure is shaped to prevent. A run that fills nothing because no document
anywhere states a figure is also a success.

Say explicitly where you looked for anything you abstained on. "No figure
found" is not a finding; "the card, config.json, the GitHub README and the
tech report all state totals only" is.
