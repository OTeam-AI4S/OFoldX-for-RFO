"""Behavior of ofoldx.models.oinvfold.processor."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch

from ofoldx.data.crop import continuous_crop
from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import Structure
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError, UnsupportedError
from ofoldx.models.design import PositionSelection
from ofoldx.models.oinvfold import OInvFoldLigandProcessor, OInvFoldPolymerProcessor, OInvFoldPolymerProcessorConfig


def nucleic_processor(modality: str) -> OInvFoldPolymerProcessor:
    return OInvFoldPolymerProcessor(OInvFoldPolymerProcessorConfig(modality=modality))


class TestPolymerPrepare:
    def test_protein_rows(self, peptide):
        prepared = OInvFoldPolymerProcessor().prepare_input(peptide("ALA", "MSE", "GLY", ligands=("ACT",)))
        assert prepared.features.sequence_tokens.tolist() == [TOKEN_INDEX[token] for token in "AMG"]
        assert prepared.features.design_mask.all()
        assert prepared.context.scope.rule == "none"
        assert prepared.context.record.records.counts()["encoding_loss"] == 1

    def test_free_residue_not_row(self, peptide):
        prepared = OInvFoldPolymerProcessor().prepare_input(peptide("ALA", "GLY", ligands=("GLY",)))
        assert prepared.features.component_index.tolist() == [0, 1]

    def test_design_selection(self, peptide):
        selection = PositionSelection(components=(ComponentRef("A", 1),))
        prepared = OInvFoldPolymerProcessor().prepare_input(peptide("ALA", "GLY", "ALA"), design=selection)
        assert prepared.features.design_mask.tolist() == [False, True, False]

    def test_several_instances_refused(self, chains):
        with pytest.raises(InvalidRequestError, match="request_invalid_positions"):
            OInvFoldPolymerProcessor().prepare_input(chains("polypeptide(L)", "ALA", "GLY", instances=("A", "B")))

    def test_crop_keeps_designed(self, peptide):
        selection = PositionSelection(components=(ComponentRef("A", 3),))
        prepared = OInvFoldPolymerProcessor().prepare_input(
            peptide("ALA", "GLY", "ALA", "GLY"),
            design=selection,
            cropper=partial(continuous_crop, max_tokens=2),
            generator=np.random.default_rng(0),
        )
        assert len(prepared.features.design_mask) == 2
        assert prepared.features.component_index[prepared.features.design_mask].tolist() == [3]

    def test_glycosidic_nitrogen(self, chains):
        prepared = nucleic_processor("dna").prepare_input(chains("polydeoxyribonucleotide", "DA", "DA"))
        system = prepared.context.structure.system
        names = system.text("atom", system.atom_symbol[prepared.context.slots.atom[7:]]).tolist()
        assert names == ["O5'", "C5'", "C4'", "P", "C3'", "O3'", "N9"]
        assert prepared.features.sequence_tokens.tolist() == [TOKEN_INDEX["DNA_A"]] * 2

    @pytest.mark.parametrize(
        ("modality", "polymer_type", "components", "tokens"),
        [
            ("dna", "polyribonucleotide", ("A", "U"), ("DNA_A", "DNA_T")),
            ("rna", "polydeoxyribonucleotide", ("DA", "DT"), ("RNA_A", "RNA_U")),
        ],
    )
    def test_modality_columns(self, chains, modality, polymer_type, components, tokens):
        prepared = nucleic_processor(modality).prepare_input(chains(polymer_type, *components))
        assert prepared.features.sequence_tokens.tolist() == [TOKEN_INDEX[token] for token in tokens]

    def test_one_row_refused(self, peptide):
        with pytest.raises(UnsupportedError, match="model_capacity"):
            OInvFoldPolymerProcessor().prepare_input(peptide("ALA"))


class TestLigandPrepare:
    def test_instance_atoms(self, peptide):
        prepared = OInvFoldLigandProcessor().prepare_input(peptide("ALA", "GLY", ligands=("ACT",)))
        system = prepared.context.structure.system
        ligand = system.component_atom_ptr[2:4]
        assert len(prepared.features.slot_ids) == ligand[1] - ligand[0]
        assert all(slot.startswith("B:0:") for slot in prepared.features.slot_ids)

    def test_ambiguous_instance_refused(self, peptide):
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            OInvFoldLigandProcessor().prepare_input(peptide("ALA", "GLY", ligands=("ACT", "ACT")))

    def test_numeric_range(self):
        points = torch.tensor([[1e100, 0, 0], [0, 1e100, 0], [0, 0, 1e100]], dtype=torch.float64)
        with pytest.raises(UnsupportedError, match="model_numeric"):
            OInvFoldLigandProcessor()(points, slot_ids=("a", "b", "c"))


class TestSequenceTarget:
    @pytest.mark.parametrize("modality", ["dna", "rna"])
    def test_physical_nucleic_truth_uses_declared_columns(self, chains, polymers, modality):
        source = chains("polyribonucleotide", "A", "A")
        truth = Structure.unobserved(polymers(("A", "R", ("DG", "DA"))))
        processor = nucleic_processor(modality)
        baseline = processor.prepare_input(source)
        prepared, target = processor.prepare(source, target=truth)
        prefix = "DNA_" if modality == "dna" else "RNA_"
        assert target.tokens.tolist() == [TOKEN_INDEX[prefix + token] for token in "GA"]
        assert target.mask.all()
        torch.testing.assert_close(prepared.features.to_dict(), baseline.features.to_dict())

    def test_partial_protein_truth_stays_unavailable(self, peptide):
        source, truth = peptide("ALA", "GLY", "LYS"), peptide("GLY", "ALA")
        processor = OInvFoldPolymerProcessor()
        prepared, target = processor.prepare(source, target=truth)
        assert target.mask.tolist() == [True, True, False]
        assert target.tokens[-1] == TOKEN_INDEX["<unk>"]
        assert prepared.features.sequence_tokens.tolist() == [TOKEN_INDEX[token] for token in "AGK"]

    def test_ligand_supervision_is_explicitly_unsupported(self, peptide):
        structure = peptide("ALA", "GLY", ligands=("ACT",))
        processor = OInvFoldLigandProcessor()
        with pytest.raises(UnsupportedError, match="model_capacity"):
            processor.prepare(structure, target=structure)
