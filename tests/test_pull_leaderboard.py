import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pull_leaderboard as pl


def test_extract_mmlu_reads_various_key_casings():
    assert pl.extract_mmlu({"MMLU": 78.6}) == 78.6
    assert pl.extract_mmlu({"mmlu": 78.6}) == 78.6
    assert pl.extract_mmlu({"MMLU-PRO": 50.0, "MMLU": 78.6}) == 78.6  # prefers plain MMLU
    assert pl.extract_mmlu({"other": 1}) is None


def test_build_scores_maps_repos_case_insensitively():
    rows = [
        {"fullname": "meta-llama/Llama-3.1-405B-Instruct", "MMLU": 88.6},
        {"fullname": "some/UnlistedModel", "MMLU": 10.0},
    ]
    repos = ["meta-llama/Llama-3.1-405B-Instruct", "google/gemma-2-27b-it"]
    scores = pl.build_scores(repos, rows)
    assert scores == {
        "meta-llama/Llama-3.1-405B-Instruct": {"mmlu": 88.6, "source": pl.LEADERBOARD_URL}
    }


def test_build_scores_skips_rows_without_mmlu():
    rows = [{"fullname": "org/m", "MMLU-PRO": 40.0}]
    assert pl.build_scores(["org/m"], rows) == {}
