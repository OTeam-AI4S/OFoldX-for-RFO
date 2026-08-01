r"""
AlphaFold 3 structure-prediction implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    AlphaFold3ConfidenceConfig,
    AlphaFold3Config,
    AlphaFold3DistogramConfig,
    AlphaFold3ProcessorConfig,
    AlphaFold3TemplateConfig,
    AlphaFold3TrunkConfig,
)
from .model import (
    AlphaFold3Input,
    AlphaFold3Model,
    AlphaFold3Output,
    AlphaFold3SamplingOutput,
    AlphaFold3StructureTarget,
    AlphaFold3TemplateFeatures,
    alphafold3_confidence_summary,
    alphafold3_pae_bin_centers,
    alphafold3_predicted_tm_score,
    alphafold3_ranking_score,
    alphafold3_tm_adjusted_pae,
)
from .pipeline import AlphaFold3Pipeline, alphafold3_fraction_disordered, alphafold3_has_clash
from .processor import AlphaFold3Processor

MODELS.register(
    "alphafold3",
    model=AlphaFold3Model,
    processor=AlphaFold3Processor,
)

PIPELINES.register(
    "alphafold3",
    pipeline=AlphaFold3Pipeline,
    model_architecture=AlphaFold3Pipeline.model_architecture,
)
