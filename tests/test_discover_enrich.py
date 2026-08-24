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


# --- gated repos ----------------------------------------------------------

def _gated(url):
    import urllib.error
    raise urllib.error.HTTPError(url, 403, "no", {}, None)


def test_enrich_row_flags_a_gated_repo():
    row = _row(hf_repo="google/gemma-3-12b-it", context_window=0)
    discover.enrich_row(row, FakeInfo({"license": "gemma"}),
                        get_text=lambda u: "", get_json=_gated)
    assert row["gated_no_access"] is True


def test_enrich_row_clears_the_flag_once_access_is_granted():
    """The flag MUST be re-derived every run, not carried forward.

    candidates.yaml rows persist across runs, so a stale True would outlive
    the access grant that fixed it and the row would claim to be locked
    forever -- the same freezing bug arena_rank had before it was re-stamped
    on every run.
    """
    row = _row(hf_repo="google/gemma-3-12b-it", context_window=0,
               gated_no_access=True)
    discover.enrich_row(row, FakeInfo({"license": "gemma"}),
                        get_text=lambda u: "",
                        get_json=lambda u: {"model_max_length": 131072})
    assert row["context_window"] == 131072
    assert "gated_no_access" not in row


def test_enrich_row_clears_the_flag_when_the_repo_merely_has_no_length():
    """An ordinary empty tokenizer_config is not an access problem."""
    row = _row(hf_repo="org/m", context_window=0, gated_no_access=True)
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: "", get_json=lambda u: {})
    assert "gated_no_access" not in row


def test_enrich_row_leaves_a_complete_row_unflagged():
    row = _row(hf_repo="google/gemma-3-12b-it", context_window=131072)
    discover.enrich_row(row, FakeInfo({"license": "gemma"}),
                        get_text=lambda u: "", get_json=_gated)
    assert "gated_no_access" not in row


CONFIG_URL = "https://huggingface.co/{}/resolve/main/config.json"
TOKENIZER_URL = "https://huggingface.co/{}/resolve/main/tokenizer_config.json"


def test_carried_row_retries_config_json_for_the_context_window():
    """thinkingmachines/Inkling states its window only in config.json
    (text_config.model_max_length) and its tokenizer_config.json carries the
    1e30 sentinel, which context_from_tokenizer correctly refuses.

    A row already in candidates.yaml skips resolve_facts entirely -- it is
    "known", so no sweep rebuilds it -- and enrich_row only ever tried the
    tokenizer. So the one file with the answer was never read again and the
    row stayed at 0 forever.
    """
    seen = []
    def get_json(url):
        seen.append(url)
        if url == CONFIG_URL.format("thinkingmachines/Inkling"):
            return {"text_config": {"model_max_length": 1048576}}
        return {"model_max_length": 10**30}

    row = _row(hf_repo="thinkingmachines/Inkling", context_window=0,
               architecture="dense", params_active_b=753.3)
    discover.enrich_row(row, None, get_text=lambda u: "", get_json=get_json)
    assert row["context_window"] == 1048576


def test_carried_row_falls_back_to_the_tokenizer_when_config_has_nothing():
    def get_json(url):
        if "config.json" in url and "tokenizer" not in url:
            return {"foo": 1}
        return {"model_max_length": 131072}

    row = _row(hf_repo="org/m", context_window=0, architecture="dense",
               params_active_b=753.3)
    discover.enrich_row(row, None, get_text=lambda u: "", get_json=get_json)
    assert row["context_window"] == 131072


def test_carried_row_flags_a_gated_config():
    row = _row(hf_repo="google/gemma-3-12b-it", context_window=0,
               architecture="dense", params_active_b=753.3)
    discover.enrich_row(row, None, get_text=lambda u: "", get_json=_gated)
    assert row["gated_no_access"] is True


def test_fresh_row_does_not_refetch_config_json():
    """A row built this run already went through resolve_facts, which fetched
    config.json. Re-reading it here would double the requests on every
    incomplete row in an org sweep for no new information."""
    seen = []
    def get_json(url):
        seen.append(url)
        return {}

    row = _row(hf_repo="org/m", context_window=0, architecture="dense",
               params_active_b=753.3)
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: "", get_json=get_json)
    assert not any(u.endswith("/config.json") for u in seen), seen
