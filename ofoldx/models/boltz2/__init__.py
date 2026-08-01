r"""
Boltz-2 folding and affinity implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    Boltz2AffinityConfig,
    Boltz2AffinityHeadConfig,
    Boltz2ConfidenceConfig,
    Boltz2Config,
    Boltz2DenoiserConfig,
    Boltz2InputEmbeddingsConfig,
    Boltz2ProcessorConfig,
    Boltz2SamplerConfig,
    Boltz2TemplateConfig,
    Boltz2TrunkConfig,
)
from .model import (
    Boltz2AffinityModel,
    Boltz2AffinityOutput,
    Boltz2AffinityScoreOutput,
    Boltz2ConfidenceOutput,
    Boltz2Contact,
    Boltz2Input,
    Boltz2Model,
    Boltz2Output,
    Boltz2Pocket,
    Boltz2Representations,
    Boltz2SamplingOutput,
    Boltz2StructureTarget,
    Boltz2Template,
)
from .pipeline import (
    Boltz2AffinityPipeline,
    Boltz2AffinityPipelineInput,
    Boltz2AffinityPipelineOutput,
    Boltz2AffinityResult,
    Boltz2AffinityScorePipeline,
    Boltz2Pipeline,
)
from .processor import Boltz2AffinityProcessor, Boltz2Processor, standardize_boltz2_affinity_smiles

MODELS.register(
    "boltz2",
    model=Boltz2Model,
    processor=Boltz2Processor,
)
MODELS.register(
    "boltz2_affinity",
    model=Boltz2AffinityModel,
    processor=Boltz2AffinityProcessor,
)
PIPELINES.register(
    "boltz2",
    pipeline=Boltz2Pipeline,
    model_architecture=Boltz2Pipeline.model_architecture,
)
PIPELINES.register(
    "boltz2_affinity",
    pipeline=Boltz2AffinityPipeline,
)
PIPELINES.register(
    "boltz2_affinity_score",
    pipeline=Boltz2AffinityScorePipeline,
    model_architecture=Boltz2AffinityScorePipeline.model_architecture,
)
