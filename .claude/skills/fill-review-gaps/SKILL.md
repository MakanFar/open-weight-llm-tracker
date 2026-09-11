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

Read the model card and find the activation figure **for this row's model**.

The hard part is never the phrasing — it is that one card usually describes a
whole family. `enrich.active_params_from_card()` already abstains whenever two
activation figures are in play, and you must hold the same bar. Cross-check
against the row's own `params_total_b`: a claim whose stated total is within
about 15% of the row's measured total is the one describing this model.

**A model with two legitimate activation figures is not resolvable — abstain.**
The live example: `deepseek-ai/DeepSeek-V4.1-Flash` publishes
`# Activated Params | 8B / 16B` and prose reading "activate only **8B
parameters per token during prefill** and **16B during decode**". There is no
single correct value, `params_active_b` is one float, and picking either one
publishes a number the vendor did not claim. Leave it; add a `notes` line
saying what the card says and that the schema cannot express it.

When you do resolve it, write:

```yaml
params_active_b: 13.0
params_active_source: "DeepSeek-V4-Pro ... 284B parameters (13B activated)"
```

`params_active_source` is a **verbatim quote** from the card, not a summary.
If the card also states a headline total that differs from the measured one by
more than a few percent, note it — do not write `params_total_stated_b`
yourself; `enrich` owns that field and has a tolerance check you would bypass.

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
and the quoted text. Then report the rows you deliberately left alone and why
— that list is the useful half of the output, and a run that fills nothing
because no card stated anything is a **successful run**.
