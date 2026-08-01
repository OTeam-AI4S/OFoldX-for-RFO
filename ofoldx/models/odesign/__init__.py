r"""
ODesign biomolecular structure-design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    ODesignConfig,
    ODesignConstraintEmbeddingsConfig,
    ODesignDenoiserConfig,
    ODesignLigandProcessorConfig,
    ODesignPairwiseConfig,
    ODesignProcessorConfig,
    ODesignTrunkConfig,
)
from .model import (
    ODesignInput,
    ODesignLigandInput,
    ODesignModel,
    ODesignOutput,
    ODesignRawOutput,
    ODesignSamplingOutput,
    ODesignStructureTarget,
)
from .objective import (
    ODesignLoss,
    covalent_bond_distance_loss,
    distogram_loss,
    odesign_loss,
    smooth_lddt_loss,
    token_bond_loss,
    weighted_denoising_loss,
)
from .pipeline import (
    ODesignBackboneComponent,
    ODesignDnaPipeline,
    ODesignLigandCandidate,
    ODesignLigandCoordinateComponent,
    ODesignLigandCoordinateSample,
    ODesignLigandPipeline,
    ODesignLigandPipelineInput,
    ODesignLigandResult,
    ODesignOInvFoldProteinPipeline,
    ODesignOInvFoldRnaPipeline,
    ODesignPolymerCandidate,
    ODesignPolymerPipelineInput,
    ODesignPolymerResult,
    ODesignProteinPipeline,
    ODesignRnaPipeline,
    designed_chains,
)
from .processor import (
    ODESIGN_GATING,
    ODESIGN_LAYOUT,
    ODESIGN_SYMMETRY,
    ODesignLigandProcessor,
    ODesignLigandTarget,
    ODesignProcessor,
    designable_components,
)

MODELS.register(
    "odesign",
    model=ODesignModel,
    processor=ODesignProcessor,
)
MODELS.register(
    "odesign_ligand",
    model=ODesignModel,
    processor=ODesignLigandProcessor,
)
PIPELINES.register(
    "odesign_protein",
    pipeline=ODesignProteinPipeline,
)
PIPELINES.register(
    "odesign_dna",
    pipeline=ODesignDnaPipeline,
)
PIPELINES.register(
    "odesign_rna",
    pipeline=ODesignRnaPipeline,
)
PIPELINES.register(
    "odesign_ligand",
    pipeline=ODesignLigandPipeline,
)
PIPELINES.register(
    "odesign_oinvfold_protein",
    pipeline=ODesignOInvFoldProteinPipeline,
)
PIPELINES.register(
    "odesign_oinvfold_rna",
    pipeline=ODesignOInvFoldRnaPipeline,
)
