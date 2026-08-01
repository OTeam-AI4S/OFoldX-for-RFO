r"""
LigandMPNN protein-sequence design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import LigandMpnnConfig, LigandMpnnProcessorConfig
from .model import (
    LigandMpnnFeatureEncoder,
    LigandMpnnInput,
    LigandMpnnModel,
    LigandMpnnOutput,
    LigandMpnnSamplingOutput,
)
from .pipeline import (
    LigandMpnnDesignAndPackPipeline,
    LigandMpnnDesignAndPackPipelineInput,
    LigandMpnnDesignResult,
    LigandMpnnPipeline,
    LigandMpnnScorePipeline,
)
from .processor import LigandMpnnProcessor

MODELS.register(
    "ligandmpnn",
    model=LigandMpnnModel,
    processor=LigandMpnnProcessor,
)
PIPELINES.register(
    "ligandmpnn",
    pipeline=LigandMpnnPipeline,
    model_architecture=LigandMpnnPipeline.model_architecture,
)
PIPELINES.register(
    "ligandmpnn_design_and_pack",
    pipeline=LigandMpnnDesignAndPackPipeline,
)

PIPELINES.register(
    "ligandmpnn_score",
    pipeline=LigandMpnnScorePipeline,
    model_architecture=LigandMpnnScorePipeline.model_architecture,
)
