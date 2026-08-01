from __future__ import annotations

import json
import math


def test_runs(examples):
    assert examples["proteinmpnn"] == ""


def test_scoring_example(examples, examples_directory):
    assert examples["proteinmpnn"] == ""
    assert examples["proteinmpnn_score"] == ""
    scores = json.loads((examples_directory / "proteinmpnn-scores.json").read_text())
    assert set(scores) == {"designed_sequence_nll", "global_sequence_nll"}
    expected_scope = [{"instance": "A", "component": index} for index in range(3)]
    for score in scores.values():
        assert math.isfinite(score["value"]) and score["value"] >= 0
        assert score["unit"] == "nat/residue"
        assert score["preferred"] == "lower"
        assert score["scope"] == expected_scope
