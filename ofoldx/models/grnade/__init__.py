r"""
gRNAde RNA inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import GrnadeConfig, GrnadeV032ProcessorConfig
from .model import GRNADE_VOCABULARY, GrnadeInput, GrnadeModel, GrnadeOutput, GrnadeSamplingOutput
from .objective import GrnadeLoss
from .pipeline import GrnadeDesignResult, GrnadePipeline, GrnadeV032Pipeline
from .processor import GrnadeProcessor, GrnadeV032Processor

MODELS.register(
    "grnade",
    model=GrnadeModel,
    processor=GrnadeProcessor,
)
PIPELINES.register(
    "grnade",
    pipeline=GrnadePipeline,
    model_architecture=GrnadePipeline.model_architecture,
)
MODELS.register(
    "grnade_v0_3_2",
    model=GrnadeModel,
    processor=GrnadeV032Processor,
)
PIPELINES.register(
    "grnade_v0_3_2",
    pipeline=GrnadeV032Pipeline,
    model_architecture=GrnadeV032Pipeline.model_architecture,
)
