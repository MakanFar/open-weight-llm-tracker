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


# --- carried rows must not freeze a wrong modality -------------------------

class _FactsInfo:
    """Only the attributes modality_of and params_b_of read."""

    def __init__(self, pipeline_tag=None, tags=None, safetensors=None):
        self.pipeline_tag = pipeline_tag
        self.tags = tags or []
        self.safetensors = safetensors


class _FactsApi:
    """model_info stub: maps repo -> pipeline_tag (and optionally a
    safetensors total in billions), or raises."""

    def __init__(self, tags, totals=None, errors=()):
        self.tags = tags
        self.totals = totals or {}
        self.errors = set(errors)
        self.calls = []

    def model_info(self, repo, **kw):
        self.calls.append(repo)
        if repo in self.errors:
            raise RuntimeError("HF 429")
        total = self.totals.get(repo)
        return _FactsInfo(pipeline_tag=self.tags.get(repo),
                          safetensors=None if total is None
                          else {"total": int(total * 1e9)})


def test_refresh_modality_corrects_a_stale_value():
    """thinkingmachines/Inkling sits in the queue as modality: text and is
    multimodal. It was staged before the pipeline requested pipeline_tag, and
    nothing re-derives modality on a carried row -- so the wrong value would
    ride into models.yaml on promotion. Same freezing bug arena_rank and
    gated_no_access each had.
    """
    rows = [{"hf_repo": "thinkingmachines/Inkling", "modality": "text"}]
    api = _FactsApi({"thinkingmachines/Inkling": "image-text-to-text"})
    discover.refresh_hf_facts(api, rows)
    assert rows[0]["modality"] == "multimodal"


def test_refresh_modality_leaves_a_row_alone_when_hf_is_silent():
    """modality_of returns None when HF publishes no usable signal. That is
    'we do not know', not 'text' -- overwriting a real value with a guess is
    exactly what enrich.py's contract forbids."""
    rows = [{"hf_repo": "org/m", "modality": "vision-language"}]
    discover.refresh_hf_facts(_FactsApi({"org/m": None}), rows)
    assert rows[0]["modality"] == "vision-language"


def test_refresh_modality_survives_a_fetch_failure():
    rows = [{"hf_repo": "org/m", "modality": "text"}]
    discover.refresh_hf_facts(_FactsApi({}, errors=["org/m"]), rows)
    assert rows[0]["modality"] == "text"


def test_refresh_modality_is_quiet_when_nothing_changes(capsys):
    rows = [{"hf_repo": "org/m", "modality": "text"}]
    discover.refresh_hf_facts(_FactsApi({"org/m": "text-generation"}), rows)
    assert rows[0]["modality"] == "text"
    assert capsys.readouterr().out == ""


def test_enrich_disambiguates_a_family_card_with_the_row_total():
    """The card documents Pro and Flash; the row's own total says which."""
    card = ("**DeepSeek-V4-Pro** with 1.6T parameters (49B activated) and "
            "**DeepSeek-V4-Flash** with 284B parameters (13B activated).")
    row = _row(hf_repo="deepseek-ai/DeepSeek-V4-Flash",
               params_total_b=290.9, params_active_b=290.9)
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: card, get_json=lambda u: {})
    assert row["params_active_b"] == 13.0


def test_enrich_falls_back_to_the_repo_name_when_the_card_is_silent():
    """Qwen3-VL-235B-A22B states its activation figure only in its repo id."""
    row = _row(hf_repo="Qwen/Qwen3-VL-235B-A22B-Instruct",
               params_total_b=235.7, params_active_b=235.7)
    discover.enrich_row(row, FakeInfo({"license": "apache-2.0"}),
                        get_text=lambda u: "no figure here",
                        get_json=lambda u: {})
    assert row["params_active_b"] == 22.0
    assert row["params_active_source"] == "repo name: 235B-A22B"


def test_enrich_prefers_the_card_over_the_repo_name():
    """Gemma 4 rounds to A4B in its id and states 3.8B in its table. The
    card is the precise figure, so the name must not win."""
    row = _row(hf_repo="google/gemma-4-26B-A4B-it",
               params_total_b=26.5, params_active_b=26.5)
    discover.enrich_row(row, FakeInfo({"license": "apache-2.0"}),
                        get_text=lambda u: "| **Active Parameters** | 3.8B |",
                        get_json=lambda u: {})
    assert row["params_active_b"] == 3.8


# --- carried rows must not freeze a wrong parameter count ------------------

def test_refresh_corrects_a_stale_params_total():
    """deepseek-ai/DeepSeek-V4-Flash sat in the queue at 158.1B against a real
    290.9B: the row froze a partial count taken while the repo was still
    uploading its shards. Unlike a frozen modality, validate.py cannot even
    flag this -- it only checks the number is positive -- so a wrong total
    promotes and renders as fact."""
    rows = [{"hf_repo": "deepseek-ai/DeepSeek-V4-Flash", "architecture": "moe",
             "params_total_b": 158.1, "params_active_b": 158.1}]
    api = _FactsApi({}, totals={"deepseek-ai/DeepSeek-V4-Flash": 290.9})
    discover.refresh_hf_facts(api, rows)
    assert rows[0]["params_total_b"] == 290.9


def test_refresh_keeps_a_dense_row_self_consistent():
    """validate.py requires active == total on a dense row, so correcting one
    without the other would append a row that fails its own schema gate."""
    rows = [{"hf_repo": "org/m", "architecture": "dense",
             "params_total_b": 7.0, "params_active_b": 7.0}]
    discover.refresh_hf_facts(_FactsApi({}, totals={"org/m": 8.0}), rows)
    assert rows[0]["params_total_b"] == 8.0
    assert rows[0]["params_active_b"] == 8.0


def test_refresh_preserves_a_known_moe_activation():
    """An activation figure read from a card is about routing, not file size.
    A corrected total says nothing about it and must not overwrite it."""
    rows = [{"hf_repo": "org/m", "architecture": "moe",
             "params_total_b": 158.1, "params_active_b": 13.0,
             "params_active_source": "284B parameters (13B activated"}]
    discover.refresh_hf_facts(_FactsApi({}, totals={"org/m": 290.9}), rows)
    assert rows[0]["params_total_b"] == 290.9
    assert rows[0]["params_active_b"] == 13.0


def test_refresh_carries_the_unknown_activation_sentinel():
    """An MoE row with active == total means 'activation unknown'. Correcting
    only the total would break the sentinel and the row would read as a model
    that routes every one of its parameters."""
    rows = [{"hf_repo": "org/m", "architecture": "moe",
             "params_total_b": 158.1, "params_active_b": 158.1}]
    discover.refresh_hf_facts(_FactsApi({}, totals={"org/m": 290.9}), rows)
    assert rows[0]["params_active_b"] == rows[0]["params_total_b"] == 290.9


def test_refresh_leaves_params_alone_when_hf_publishes_none():
    """No safetensors metadata is 'we do not know', not zero -- the same
    contract the modality half keeps."""
    rows = [{"hf_repo": "org/m", "architecture": "moe",
             "params_total_b": 158.1, "params_active_b": 13.0}]
    discover.refresh_hf_facts(_FactsApi({}), rows)
    assert rows[0]["params_total_b"] == 158.1


def test_refresh_is_quiet_when_the_params_are_already_right(capsys):
    rows = [{"hf_repo": "org/m", "architecture": "moe",
             "params_total_b": 290.9, "params_active_b": 13.0}]
    discover.refresh_hf_facts(_FactsApi({}, totals={"org/m": 290.9}), rows)
    assert capsys.readouterr().out == ""


def test_refresh_reads_one_model_info_per_row():
    """Both facts come off the same response. Fetching twice would double the
    request count on every carried row of every weekly run for nothing."""
    rows = [{"hf_repo": "org/m", "modality": "text", "architecture": "dense",
             "params_total_b": 7.0, "params_active_b": 7.0}]
    api = _FactsApi({"org/m": "image-text-to-text"}, totals={"org/m": 8.0})
    discover.refresh_hf_facts(api, rows)
    assert api.calls == ["org/m"]
    assert rows[0]["modality"] == "multimodal"
    assert rows[0]["params_total_b"] == 8.0


def test_enrich_records_the_total_the_card_stated():
    """params_total_b is the measured tensor count; the card's headline figure
    is a different quantity. Both belong in the row, labelled -- carrying only
    the quote made the record argue with itself."""
    card = "**Flash** with 284B parameters (13B activated)."
    row = _row(hf_repo="deepseek-ai/DeepSeek-V4-Flash",
               params_total_b=290.9, params_active_b=290.9)
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: card, get_json=lambda u: {})
    assert row["params_total_b"] == 290.9
    assert row["params_total_stated_b"] == 284.0


def test_enrich_records_no_stated_total_when_the_card_names_none():
    row = _row()
    discover.enrich_row(row, FakeInfo({"license": "mit"}),
                        get_text=lambda u: CARD, get_json=lambda u: {})
    assert row["params_active_b"] == 23.0
    assert "params_total_stated_b" not in row


def test_enrich_records_the_repo_names_total_on_the_fallback_path():
    row = _row(hf_repo="Qwen/Qwen3-VL-235B-A22B-Instruct",
               params_total_b=235.7, params_active_b=235.7)
    discover.enrich_row(row, FakeInfo({"license": "apache-2.0"}),
                        get_text=lambda u: "nothing here", get_json=lambda u: {})
    assert row["params_total_stated_b"] == 235.0
