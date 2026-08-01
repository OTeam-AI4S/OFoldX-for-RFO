"""Behavior of ofoldx.models.membranempnn.processor."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.molecular import ComponentRef, MolecularSystem, SystemArrays
from ofoldx.data.structure import Structure
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.conditions import TiedResidueIdentities
from ofoldx.models.membranempnn import MembraneMpnnProcessor


class TestLabels:
    @pytest.mark.parametrize(
        "labels,expected",
        [(2, [2, 2, 2]), ({ComponentRef("A", 1): 1}, [0, 1, 0])],
        ids=["scalar", "per-residue"],
    )
    def test_row_labels(self, peptide, labels, expected):
        prepared = MembraneMpnnProcessor().prepare_input(peptide("ALA", "GLY", "LYS"), membrane_labels=labels)
        assert prepared.features.membrane_labels.tolist() == [expected]

    def test_unknown_residue_refused(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_positions"):
            MembraneMpnnProcessor().prepare_input(peptide("ALA"), membrane_labels={ComponentRef("B", 0): 1})


def test_tied_weight_default(peptide):
    pair = (ComponentRef("A", 0), ComponentRef("A", 1))
    prepared = MembraneMpnnProcessor().prepare_input(
        peptide("ALA", "GLY"), conditions=(TiedResidueIdentities((pair,)),), tied_weights={pair[0]: 0.5}
    )
    assert prepared.features.tied_weights.tolist() == [[0.5, 1.0]]


class TestSequenceTarget:
    def test_prepare_pair_and_separate_target_agree(self, peptide):
        source = peptide("ALA", "GLY", "LYS")
        truth = peptide("GLY", "ALA", "SER")
        processor = MembraneMpnnProcessor()
        baseline = processor.prepare_input(source)
        prepared, target = processor.prepare(source, target=truth)
        separate = processor.prepare_target(truth, prepared=prepared)
        torch.testing.assert_close(prepared.features.to_dict(), baseline.features.to_dict())
        assert prepared.context.structure is source
        assert target.tokens.tolist() == [[TOKEN_INDEX[token] for token in "GAS"]]
        assert target.mask.all()
        torch.testing.assert_close(target.tokens, separate.tokens)
        torch.testing.assert_close(target.mask, separate.mask)
        unlabelled, absent = processor.prepare(source)
        assert absent is None
        torch.testing.assert_close(unlabelled.features.to_dict(), baseline.features.to_dict())

    def test_partial_truth_is_not_filled_from_source(self, peptide):
        source = peptide("ALA", "GLY", "LYS")
        truth = peptide("GLY", "ALA")
        processor = MembraneMpnnProcessor()
        prepared = processor.prepare_input(source)
        target = processor.prepare_target(truth, prepared=prepared)
        assert target.mask.tolist() == [[True, True, False]]
        assert target.tokens[0, 2] == TOKEN_INDEX["<unk>"]
        assert prepared.features.sequence_tokens.tolist() == [[TOKEN_INDEX[token] for token in "AGK"]]

    def test_known_untrained_classes_remain_known(self, peptide, polymers):
        source = peptide("ALA", "GLY")
        arrays = polymers(("A", "P", ("ALA", "GLY"))).build()
        # Sequence truth can be known even when only partial chemical observations exist.
        symbols = {**arrays.symbols, "component": ("SEC", "PYL")}
        truth = Structure.unobserved(MolecularSystem.from_arrays(SystemArrays(symbols, arrays.columns)))
        processor = MembraneMpnnProcessor()
        prepared = processor.prepare_input(source)
        target = processor.prepare_target(truth, prepared=prepared)
        assert target.tokens.tolist() == [[TOKEN_INDEX["U"], TOKEN_INDEX["O"]]]
        assert target.mask.all()
