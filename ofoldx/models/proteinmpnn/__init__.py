r"""
ProteinMPNN inverse-folding implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import (
    ProteinMpnnCaFeatureEncoder,
    ProteinMpnnDesignOutput,
    ProteinMpnnInput,
    ProteinMpnnModel,
    ProteinMpnnOutput,
    ProteinMpnnSamplingOutput,
)
from .pipeline import (
    ProteinMpnnDesignAndPackPipeline,
    ProteinMpnnDesignAndPackPipelineInput,
    ProteinMpnnDesignResult,
    ProteinMpnnPipeline,
    ProteinMpnnScorePipeline,
)
from .processor import ProteinMpnnProcessor

MODELS.register(
    "proteinmpnn",
    model=ProteinMpnnModel,
    processor=ProteinMpnnProcessor,
)
PIPELINES.register(
    "proteinmpnn",
    pipeline=ProteinMpnnPipeline,
    model_architecture=ProteinMpnnPipeline.model_architecture,
)
PIPELINES.register(
    "proteinmpnn_design_and_pack",
    pipeline=ProteinMpnnDesignAndPackPipeline,
)

PIPELINES.register(
    "proteinmpnn_score",
    pipeline=ProteinMpnnScorePipeline,
    model_architecture=ProteinMpnnScorePipeline.model_architecture,
)
