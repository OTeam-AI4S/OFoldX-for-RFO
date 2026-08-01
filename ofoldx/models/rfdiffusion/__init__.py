r"""
RFdiffusion protein-backbone design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    RfDiffusionConfig,
    RfDiffusionDenoiserConfig,
    RfDiffusionDiffuserConfig,
    RfDiffusionPreprocessConfig,
    RfDiffusionSe3Config,
)
from .model import (
    RFDIFFUSION_ATOM14_NAMES,
    RFDIFFUSION_MASK_TOKEN_ID,
    RFDIFFUSION_RESIDUE_TYPE_INDEX,
    RFDIFFUSION_RESIDUE_TYPES,
    RFDIFFUSION_UNKNOWN_TOKEN_ID,
    RfDiffusionInput,
    RfDiffusionModel,
    RfDiffusionOutput,
    RfDiffusionPotentialInput,
    RfDiffusionSamplingOutput,
    RfDiffusionTrajectory,
    rfdiffusion_residue_token_id,
)
from .pipeline import (
    RfDiffusionDesignCandidate,
    RfDiffusionDesignResult,
    RfDiffusionPipeline,
    RfDiffusionPipelineInput,
)
from .processor import (
    RfDiffusionCoordination,
    RfDiffusionFlatBottomRadialPotential,
    RfDiffusionPlacements,
    RfDiffusionPointGroupSymmetry,
    RfDiffusionProcessor,
    RfDiffusionSecondaryStructure,
    RfDiffusionSubstrateInteraction,
    from_contig_string,
    to_contig_string,
)

MODELS.register(
    "rfdiffusion",
    model=RfDiffusionModel,
    processor=RfDiffusionProcessor,
)

PIPELINES.register(
    "rfdiffusion",
    pipeline=RfDiffusionPipeline,
    model_architecture=RfDiffusionPipeline.model_architecture,
)
