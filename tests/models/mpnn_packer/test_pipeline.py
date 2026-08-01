"""Behavior of ofoldx.models.mpnn_packer.pipeline."""

from __future__ import annotations

import numpy as np

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.models.design import PositionSelection


class TestPack:
    def test_side_chains_generated(self, small_packer, peptide):
        structure = peptide("LYS", "ALA")
        selection = PositionSelection(components=(ComponentRef("A", 0),))
        result = small_packer.pack(structure, positions=selection, num_denoising_steps=1, num_torsion_samples=1, seed=3)
        packed = result.predictions[0].structure
        names = structure.system.text("atom", structure.system.atom_symbol)
        lysine = np.arange(structure.system.component_atom_ptr[0], structure.system.component_atom_ptr[1])
        side_chain = lysine[np.isin(names[lysine], ["CB", "CG", "CD", "CE", "NZ"])]
        assert (packed.coordinate_evidence[side_chain] == CoordinateEvidence.GENERATED).all()
        backbone = lysine[np.isin(names[lysine], ["N", "CA", "C", "O"])]
        np.testing.assert_array_equal(packed.coordinates[backbone], structure.coordinates[backbone])
        assert "chi1_log_density" in result.predictions[0].metrics

    def test_nothing_packed(self, small_packer, peptide):
        structure = peptide("ALA")
        result = small_packer.pack(structure, positions=PositionSelection(indices=np.zeros(0, dtype=np.int64)))
        assert result.predictions[0].structure is structure
