"""Behavior of ofoldx.models.boltzgen.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import ComponentRef
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.errors import InvalidRequestError
from ofoldx.models.boltzgen.model import BoltzGenInverseFoldingModel, BoltzGenModel
from ofoldx.models.boltzgen.pipeline import BoltzGenBackbonePipeline, BoltzGenInverseFoldingPipeline
from ofoldx.models.boltzgen.processor import BoltzGenInverseFoldingProcessor, BoltzGenProcessor
from ofoldx.models.conditions import GeneratedSegment
from ofoldx.models.design import PositionSelection

PLACEMENT = {"binder": {"binder": GeneratedSegment("protein", 2, 2)}}


class TestBackboneDesign:
    @pytest.fixture
    def pipeline(self, design_config):
        torch.manual_seed(37)
        model = BoltzGenModel(design_config).double().eval()
        return BoltzGenBackbonePipeline(model=model, processor=BoltzGenProcessor(dtype=torch.float64))

    @pytest.fixture
    def candidate(self, pipeline, peptide, design_library):
        result = pipeline.design(
            peptide("ALA", "GLY"),
            placement=PLACEMENT,
            library=design_library,
            num_candidates=1,
            num_sampling_steps=1,
            num_recycles=1,
            seed=3,
        )
        return result.candidates[0]

    def test_designed_positions(self, candidate):
        assert candidate.designed_positions == (ComponentRef("chain_0", 0), ComponentRef("chain_0", 1))
        assert candidate.connected_designed_positions == candidate.designed_positions

    def test_target_coordinates_kept(self, pipeline, peptide, design_library, candidate):
        planned = pipeline.prepare(peptide("ALA", "GLY"), placement=PLACEMENT, library=design_library)
        target = np.arange(planned.structure.system.component_atom_ptr[2])
        np.testing.assert_array_equal(candidate.structure.coordinates[target], planned.structure.coordinates[target])
        assert (candidate.structure.coordinate_evidence[target] == CoordinateEvidence.OBSERVED).all()

    def test_invalid_candidate_count(self, pipeline, peptide, design_library):
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            pipeline.design(peptide("ALA"), placement=PLACEMENT, library=design_library, num_candidates=0)

    def test_generated_coordinates_written(self, candidate):
        system = candidate.structure.system
        generated = np.arange(system.component_atom_ptr[2], len(system.atom_symbol))
        written = candidate.structure.coordinate_mask[generated]
        assert written.any() and not written.all()  # the modelled backbone, not the unaddressed side chain
        evidence = candidate.structure.coordinate_evidence[generated][written]
        assert (evidence == CoordinateEvidence.GENERATED).all()


class TestInverseFolding:
    @pytest.fixture
    def pipeline(self, inverse_folding_config):
        torch.manual_seed(11)
        model = BoltzGenInverseFoldingModel(inverse_folding_config).double().eval()
        processor = BoltzGenInverseFoldingProcessor(dtype=torch.float64)
        return BoltzGenInverseFoldingPipeline(model=model, processor=processor)

    def test_sampled_sequence(self, pipeline, peptide, library):
        result = pipeline.design(
            peptide("ALA", "GLY", "LYS"),
            global_omit_tokens=tuple("CDEFGHIKLMNPQRSTVWY"),
            temperature=None,
            library=library,
            seed=5,
        )
        candidate = result.candidates[0]
        system = candidate.structure.system
        assert system.text("component", system.component_symbol).tolist() == ["ALA", "ALA", "ALA"]
        assert candidate.designed_positions == tuple(ComponentRef("A", index) for index in range(3))

    def test_selected_positions(self, pipeline, peptide, library):
        result = pipeline.design(
            peptide("ALA", "GLY", "LYS"),
            design=PositionSelection(components=(ComponentRef("A", 1),)),
            global_omit_tokens=tuple("CDEFGHIKLMNPQRSTVWY"),
            temperature=None,
            library=library,
            seed=5,
        )
        system = result.candidates[0].structure.system
        assert system.text("component", system.component_symbol).tolist() == ["ALA", "ALA", "LYS"]
