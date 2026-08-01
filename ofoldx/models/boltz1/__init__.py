r"""
Boltz-1 structure-prediction implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    Boltz1ConfidenceConfig,
    Boltz1Config,
    Boltz1DenoiserConfig,
    Boltz1InputEmbeddingsConfig,
    Boltz1ProcessorConfig,
    Boltz1SamplerConfig,
    Boltz1TrunkConfig,
)
from .model import (
    Boltz1ConfidenceOutput,
    Boltz1Input,
    Boltz1Model,
    Boltz1Output,
    Boltz1Pocket,
    Boltz1Representations,
    Boltz1SamplingOutput,
    Boltz1SamplingWithConfidenceOutput,
    Boltz1StructureTarget,
)
from .pipeline import Boltz1Pipeline
from .processor import Boltz1Processor

MODELS.register(
    "boltz1",
    model=Boltz1Model,
    processor=Boltz1Processor,
)
PIPELINES.register(
    "boltz1",
    pipeline=Boltz1Pipeline,
    model_architecture=Boltz1Pipeline.model_architecture,
)
