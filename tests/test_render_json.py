import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import render_json as rj


def _model(**kw):
    base = dict(name="M", developer="Org", release_date=date(2025, 1, 1),
                params_total_b=7, params_active_b=7, architecture="dense",
                context_window=4096, modality="text", license="mit",
                commercial_use=True, hf_repo="org/m")
    base.update(kw)
    return base


EMPTY_AA = {"repos": {}, "identities": {}}
EMPTY_RANKS = {"repos": {}, "names": {}, "identities": {}}


def test_dates_become_iso_strings():
    """json.dumps cannot serialise a datetime.date, so a raw YAML value would
    crash the render rather than degrade."""
    out = rj.build([_model()], EMPTY_AA, EMPTY_RANKS, generated="2026-08-28")
    assert out["models"][0]["release_date"] == "2025-01-01"
    json.dumps(out)  # must not raise


def test_arena_rank_and_aa_index_are_numbers_or_null():
    """The table prints an em dash for a missing value; JSON must not -- a
    consumer filtering on aa_index would have to special-case a glyph."""
    aa = {"repos": {"org/m": {"index": 42, "variant": None}}, "identities": {}}
    ranks = {"repos": {"org/m": 10}, "names": {}, "identities": {}}
    out = rj.build([_model()], aa, ranks, generated="2026-08-28")
    assert out["models"][0]["arena_rank"] == 10
    assert out["models"][0]["aa_index"] == 42

    out = rj.build([_model()], EMPTY_AA, EMPTY_RANKS, generated="2026-08-28")
    assert out["models"][0]["arena_rank"] is None
    assert out["models"][0]["aa_index"] is None


def test_rows_are_ordered_newest_first_like_the_table():
    """The JSON and the README table are the same index; a consumer reading
    both should not have to reconcile two orderings."""
    rows = [_model(name="old", release_date=date(2024, 1, 1)),
            _model(name="new", release_date=date(2026, 1, 1))]
    out = rj.build(rows, EMPTY_AA, EMPTY_RANKS, generated="2026-08-28")
    assert [m["name"] for m in out["models"]] == ["new", "old"]


def test_envelope_carries_count_licence_and_stamp():
    out = rj.build([_model()], EMPTY_AA, EMPTY_RANKS, generated="2026-08-28")
    assert out["count"] == 1
    assert out["generated"] == "2026-08-28"
    assert out["license"] == "CC-BY-4.0"
    assert "github.com" in out["source"]


def test_unset_optional_fields_are_omitted_not_nulled():
    """A row that never had license_notes should not sprout a null one."""
    out = rj.build([_model()], EMPTY_AA, EMPTY_RANKS, generated="2026-08-28")
    assert "license_notes" not in out["models"][0]


def test_discovery_only_fields_never_reach_the_json():
    """models.yaml is the source, but a stray staging field on a hand-edited
    row must not be republished as though it were part of the index."""
    row = _model(needs_review=["x"], discovered_via="org-sweep", downloads=5)
    out = rj.build([row], EMPTY_AA, EMPTY_RANKS, generated="2026-08-28")
    assert "needs_review" not in out["models"][0]
    assert "discovered_via" not in out["models"][0]
    assert "downloads" not in out["models"][0]


def test_commercial_use_verified_is_always_present():
    """This flag is what separates a licence someone read from one inferred
    from a tag. Omitting it when unset makes a consumer default it -- and the
    safe default is the opposite of the one absence usually implies, so a
    trailing `?` in the table would silently become a settled claim in JSON."""
    out = rj.build([_model()], EMPTY_AA, EMPTY_RANKS, generated="2026-08-28")
    assert out["models"][0]["commercial_use_verified"] is False
