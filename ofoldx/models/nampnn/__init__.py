r"""
NA-MPNN mixed-polymer inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import NaMpnnConfig, NaMpnnProcessorConfig
from .model import (
    NaMpnnFeatureEncoder,
    NaMpnnInput,
    NaMpnnModel,
    NaMpnnOutput,
    NaMpnnSamplingOutput,
)
from .objective import NaMpnnLoss, smoothed_loss
from .pipeline import NaMpnnDesignResult, NaMpnnPipeline, NaMpnnScorePipeline
from .processor import NaMpnnProcessor

MODELS.register(
    "nampnn",
    model=NaMpnnModel,
    processor=NaMpnnProcessor,
)
PIPELINES.register(
    "nampnn",
    pipeline=NaMpnnPipeline,
    model_architecture=NaMpnnPipeline.model_architecture,
)

PIPELINES.register(
    "nampnn_score",
    pipeline=NaMpnnScorePipeline,
    model_architecture=NaMpnnScorePipeline.model_architecture,
)
