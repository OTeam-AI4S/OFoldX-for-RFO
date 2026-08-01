"""Behavior of ofoldx.models.mpnn_packer.processor."""

from __future__ import annotations

import pytest

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import UnsupportedError
from ofoldx.models.design import PositionSelection
from ofoldx.models.mpnn_packer import MpnnPackerProcessor, MpnnPackerProcessorConfig


@pytest.fixture
def processor():
    return MpnnPackerProcessor(MpnnPackerProcessorConfig(num_context_atoms=1))


class TestPrepare:
    def test_default_packs_tabled_residues(self, processor, peptide):
        prepared = processor.prepare_input(peptide("LYS", "MSE", "ALA"))
        assert prepared.features.pack_mask.tolist() == [[True, False, True]]
        assert prepared.features.sequence_tokens.tolist() == [[TOKEN_INDEX["K"], TOKEN_INDEX["X"], TOKEN_INDEX["A"]]]
        assert prepared.context.record.records.counts().get("encoding_loss") == 1

    def test_side_chain_scope(self, processor, peptide):
        structure = peptide("LYS")
        prepared = processor.prepare_input(structure)
        names = set(structure.system.text("atom", structure.system.atom_symbol[prepared.context.scope.atoms]).tolist())
        assert {"CB", "NZ"} <= names
        assert not names & {"N", "CA", "C", "O"}

    def test_packed_mse_refused(self, processor, peptide):
        selection = PositionSelection(components=(ComponentRef("A", 1),))
        with pytest.raises(UnsupportedError, match="model_capacity"):
            processor.prepare_input(peptide("LYS", "MSE", "ALA"), positions=selection)

    def test_ligand_context_consumed(self, processor, peptide):
        structure = peptide("LYS", ligands=("ACT",))
        prepared = processor.prepare_input(structure)
        ligand = range(structure.system.component_atom_ptr[1], structure.system.component_atom_ptr[2])
        assert any(atom in ligand for atom in prepared.context.slots.atom.tolist())
