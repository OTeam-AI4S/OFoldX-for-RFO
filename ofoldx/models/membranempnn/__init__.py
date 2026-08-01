r"""Membrane-label-conditioned MPNN sequence design."""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..proteinmpnn.model import ProteinMpnnInput, ProteinMpnnOutput, ProteinMpnnSamplingOutput
from ..registry import MODELS
from .config import MembraneMpnnConfig
from .model import MembraneMpnnModel
from .pipeline import (
    MembraneMpnnDesignAndPackPipeline,
    MembraneMpnnDesignResult,
    MembraneMpnnPipeline,
    MembraneMpnnScorePipeline,
)
from .processor import MembraneMpnnProcessor

MODELS.register("membranempnn", model=MembraneMpnnModel, processor=MembraneMpnnProcessor)
PIPELINES.register(
    "membranempnn",
    pipeline=MembraneMpnnPipeline,
    model_architecture=MembraneMpnnPipeline.model_architecture,
)
PIPELINES.register(
    "membranempnn_design_and_pack",
    pipeline=MembraneMpnnDesignAndPackPipeline,
)

PIPELINES.register(
    "membranempnn_score",
    pipeline=MembraneMpnnScorePipeline,
    model_architecture=MembraneMpnnScorePipeline.model_architecture,
)
