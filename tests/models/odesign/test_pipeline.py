"""Behavior of ofoldx.models.odesign.pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data.molecular import AtomSite, ComponentRef, MonomerType
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.errors import InvalidRequestError, UnsupportedError
from ofoldx.models.design import PositionSelection
from ofoldx.models.odesign import (
    ODesignBackboneComponent,
    ODesignConfig,
    ODesignDnaPipeline,
    ODesignLigandCoordinateComponent,
    ODesignLigandProcessor,
    ODesignModel,
    ODesignProcessor,
    ODesignProcessorConfig,
    designed_chains,
)
from ofoldx.models.odesign.model import ODesignSamplingOutput
from ofoldx.models.odesign.pipeline import has_ligand_atom_context
from ofoldx.models.oinvfold import (
    OInvFoldPolymerConfig,
    OInvFoldPolymerModel,
    OInvFoldPolymerPipeline,
    OInvFoldPolymerProcessor,
    OInvFoldPolymerProcessorConfig,
)

FIRST = PositionSelection(components=(ComponentRef("A", 0),))


def small_model() -> ODesignModel:
    with torch.device("meta"):
        return ODesignModel(
            ODesignConfig(
                trunk={"num_pairformer_blocks": 1, "num_msa_stages": 1},
                denoiser={"num_token_encoder_blocks": 1, "num_atom_encoder_blocks": 1, "num_atom_decoder_blocks": 1},
            )
        )


def sampling_output(prepared, samples: np.ndarray) -> ODesignSamplingOutput:
    tokens = len(prepared.features.token_mask[0])
    return ODesignSamplingOutput(
        coordinates=torch.from_numpy(samples)[None, None],
        distogram_logits=torch.zeros(1, 1, tokens, tokens, 4),
        token_bond_logits=torch.zeros(1, 1, tokens, tokens, 2),
        batch_size=[1],
    )


class TestDesignedChains:
    def test_grouped_by_instance(self, peptide):
        structure = peptide("ALA", "GLY", ligands=("ACT",))
        (symbol, positions), *rest = designed_chains(structure, PositionSelection(), MonomerType.PROTEIN)
        assert (symbol, positions.tolist()) == ("A", [0, 1])
        assert not rest

    def test_other_monomer_refused(self, peptide):
        with pytest.raises(UnsupportedError, match="model_capacity"):
            designed_chains(peptide("ALA", "GLY"), PositionSelection(), MonomerType.RNA)


class TestLigandContext:
    @pytest.mark.parametrize("ligands,expected", [((), False), (("ACT",), True)])
    def test_observed_heavy_atoms(self, peptide, ligands, expected):
        assert has_ligand_atom_context(peptide("ALA", "GLY", ligands=ligands)) is expected


class TestBackboneDecode:
    def test_scope_only(self, peptide):
        structure = peptide("ALA", "GLY")
        fixed = AtomSite("A", 1, "CA")
        processor = ODesignProcessor(ODesignProcessorConfig(artifact="protein_rigid"))
        prepared = processor.prepare_input(structure, design=FIRST, fixed_atoms=(fixed,))
        component = ODesignBackboneComponent(model=small_model(), processor=processor)
        samples = np.zeros((1, len(prepared.context.slots.atom), 3))
        (decoded,) = component.postprocess(prepared, sampling_output(prepared, samples))
        system = structure.system
        designed = np.arange(int(system.component_atom_ptr[0]), int(system.component_atom_ptr[1]))
        context = structure.system.atom_index([fixed])[0]
        placed = decoded.coordinate_mask[designed]
        assert (decoded.coordinate_evidence[designed][placed] == CoordinateEvidence.GENERATED).all()
        assert not placed.all()  # the generated residue's unaddressed atoms are withdrawn
        np.testing.assert_array_equal(decoded.coordinates[context], structure.coordinates[context])


class TestLigandCoordinates:
    def test_generated_slots_decoded(self, peptide):
        processor = ODesignLigandProcessor()
        prepared = processor.prepare_input(peptide("ALA", "GLY"), generated_atoms=3)
        component = ODesignLigandCoordinateComponent(model=small_model(), processor=processor)
        samples = np.arange(2 * len(prepared.context.slots.atom) * 3, dtype=np.float64).reshape(2, -1, 3)
        decoded, logits = component.postprocess(prepared, sampling_output(prepared, samples))
        assert len(decoded) == 2
        assert decoded[0].slot_ids == ("0", "1", "2")
        np.testing.assert_array_equal(decoded[1].coordinates, samples[1, -3:])
        assert logits.shape == (3, 3, 2)
        assert decoded[0].context is prepared.context.structure


class TestRouteGuards:
    def test_modality_must_match(self):
        with torch.device("meta"):
            inverse_folder = OInvFoldPolymerPipeline(
                model=OInvFoldPolymerModel(OInvFoldPolymerConfig(num_encoder_blocks=1)),
                processor=OInvFoldPolymerProcessor(OInvFoldPolymerProcessorConfig(modality="protein")),
            )
        backbone = ODesignBackboneComponent(
            model=small_model(), processor=ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
        )
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            ODesignDnaPipeline(backbone=backbone, inverse_folder=inverse_folder)
