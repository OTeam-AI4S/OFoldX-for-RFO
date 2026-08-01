"""Behavior of ofoldx.models.membranempnn.pipeline."""

from __future__ import annotations

import pytest

from ofoldx.data.molecular import ComponentRef
from ofoldx.models.conditions import AllowedResidueIdentities
from ofoldx.models.design import PositionSelection
from ofoldx.models.membranempnn import MembraneMpnnScorePipeline


def test_replacement(designer, peptide, library):
    condition = AllowedResidueIdentities((ComponentRef("A", 0),), frozenset({"LYS"}))
    result = designer.design(
        peptide("ALA", "GLY"), membrane_labels=2, conditions=(condition,), seed=19, library=library
    )
    target = result.candidates[0].target
    assert target.system.text("component", target.system.component_symbol)[0] == "LYS"
    assert 0 in result.records[0].replaced().tolist()


@pytest.mark.parametrize("use_sequence", [True, False])
def test_score_scopes(designer, peptide, use_sequence):
    scorer = MembraneMpnnScorePipeline(model=designer.model, processor=designer.processor)
    result = scorer.score(
        peptide("ALA", "GLY"),
        membrane_labels={ComponentRef("A", 0): 2},
        design=PositionSelection(components=(ComponentRef("A", 1),)),
        use_sequence=use_sequence,
        num_decoding_orders=2,
        seed=29,
    )
    assert result.scores["designed_sequence_nll"].scope == frozenset({ComponentRef("A", 1)})
    assert len(result.scores["global_sequence_nll"].evaluations) == 2
