"""Behavior of ofoldx.models.tensors."""

from __future__ import annotations

import torch

from ofoldx.models.tensors import to_tensors


class TestToTensors:
    def test_columns_of_both_records(self, peptide):
        structure = peptide("GLY")
        tensors = to_tensors(structure, ("coordinates", "atomic_number"), device=torch.device("cpu"))
        assert tensors["coordinates"].shape == (len(structure.coordinates), 3)
        assert tensors["atomic_number"].tolist() == structure.system.atomic_number.tolist()
