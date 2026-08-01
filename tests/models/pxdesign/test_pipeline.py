"""Behavior of ofoldx.models.pxdesign.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.structure import CoordinateEvidence
from ofoldx.errors import InvalidRequestError
from ofoldx.models.pxdesign import PxDesignConfig, PxDesignModel, PxDesignProcessor, PxDesignSamplingOutput
from ofoldx.models.pxdesign.pipeline import PxDesignBackbonePipeline, generated_components
from ofoldx.models.pxdesign.processor import from_binder


@pytest.fixture
def pipeline():
    with torch.device("meta"):
        model = PxDesignModel(PxDesignConfig())
    return PxDesignBackbonePipeline(model=model, processor=PxDesignProcessor(dtype=torch.float64))


@pytest.fixture
def prepared(pipeline, peptide, design_library):
    placement = from_binder(2, [("A", [(0, 2)])])
    return pipeline.prepare(peptide("ALA", "GLY", "LYS"), placement=placement, library=design_library)


def samples(prepared, count=2):
    atoms = len(prepared.context.slots.atom)
    return PxDesignSamplingOutput(
        coordinates=torch.arange(count * atoms * 3, dtype=torch.float64).reshape(1, count, atoms, 3),
        batch_size=[1],
    )


class TestPostprocess:
    def test_generated_components(self, prepared):
        assert generated_components(prepared).tolist() == [3, 4]

    def test_decoded_candidates(self, pipeline, prepared):
        output = samples(prepared)
        result = pipeline.postprocess(prepared, output)
        assert len(result.candidates) == 2
        structure = result.candidates[1]
        rows = np.flatnonzero(
            prepared.context.scope.atoms[np.maximum(prepared.context.slots.atom, 0)]
            & (prepared.context.slots.atom >= 0)
        )
        atoms = prepared.context.slots.atom[rows]
        np.testing.assert_array_equal(structure.coordinates[atoms], output.coordinates[0, 1].numpy()[rows])
        assert (structure.coordinate_evidence[atoms] == CoordinateEvidence.GENERATED).all()

    def test_target_coordinates_kept(self, pipeline, prepared):
        result = pipeline.postprocess(prepared, samples(prepared))
        structure = result.candidates[0]
        outside = np.flatnonzero(~prepared.context.scope.atoms)
        np.testing.assert_array_equal(structure.coordinates[outside], prepared.context.structure.coordinates[outside])
        assert (structure.coordinate_evidence[outside] == prepared.context.structure.coordinate_evidence[outside]).all()

    def test_unwritten_side_chain(self, pipeline, prepared):
        result = pipeline.postprocess(prepared, samples(prepared))
        system = prepared.context.structure.system
        names = system.text("atom", system.atom_symbol[result.unwritten[0]]).tolist()
        assert names and not {"N", "CA", "C", "O"} & set(names)


class TestPrepare:
    def test_target_required(self, pipeline, design_library):
        placement = from_binder(1, [("A", [(0, 1)])])
        with pytest.raises(InvalidRequestError, match="request_invalid_positions"):
            pipeline.prepare(placement=placement, library=design_library)
