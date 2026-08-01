r"""
MPNN side-chain packing implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import (
    MpnnPackerFeatureEncoder,
    MpnnPackerInput,
    MpnnPackerModel,
    MpnnPackerOutput,
    MpnnPackerSamplingOutput,
)
from .pipeline import MpnnPackerPipeline
from .processor import MpnnPackerProcessor

MODELS.register(
    "mpnn_packer",
    model=MpnnPackerModel,
    processor=MpnnPackerProcessor,
)
PIPELINES.register(
    "mpnn_packer",
    pipeline=MpnnPackerPipeline,
    model_architecture=MpnnPackerPipeline.model_architecture,
)
