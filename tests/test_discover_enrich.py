import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover
from test_enrich import FakeInfo


def _row(**kw):
    base = dict(name="M", hf_repo="org/m", architecture="moe",
                params_total_b=753.3, params_active_b=753.3, license="other")
    base.update(kw)
    return base


CARD = "It has ~428B parameters and ~23B activated parameters."


def test_enrich_fills_active_params_from_the_card():
    row = _row()
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: CARD, get_json=lambda u: {})
    assert row["params_active_b"] == 23.0
    assert "23B activated" in row["params_active_source"]


def test_enrich_leaves_active_params_alone_when_the_card_is_silent():
    row = _row()
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: "no figure", get_json=lambda u: {})
    assert row["params_active_b"] == 753.3
    assert "params_active_source" not in row


def test_enrich_does_not_touch_a_dense_row():
    """Dense rows must keep active == total; validate.py enforces it."""
    row = _row(architecture="dense", params_total_b=70.0, params_active_b=70.0)
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: CARD, get_json=lambda u: {})
    assert row["params_active_b"] == 70.0


def test_enrich_recovers_the_licence():
    row = _row()
    discover.enrich_row(row, FakeInfo({"license": "other", "license_name": "kimi-k3"}),
                        get_text=lambda u: "", get_json=lambda u: {})
    assert row["license"] == "kimi-k3"


def test_enrich_keeps_the_original_licence_when_nothing_better_is_found():
    row = _row(license="other")
    discover.enrich_row(row, FakeInfo({"license": "other"}),
                        get_text=lambda u: "", get_json=lambda u: {})
    assert row["license"] == "other"


def test_enrich_survives_a_card_fetch_failure():
    def boom(url):
        raise RuntimeError("429")
    row = _row()
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=boom, get_json=lambda u: {})
    assert row["params_active_b"] == 753.3


def test_enrich_recovers_a_missing_context_window():
    row = _row(context_window=0)
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: "", get_json=lambda u: {"model_max_length": 262144})
    assert row["context_window"] == 262144
