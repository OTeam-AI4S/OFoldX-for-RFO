r"""
PxDesign protein-structure design implementation.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    PxDesignConditionEmbeddingsConfig,
    PxDesignConfig,
    PxDesignDenoiserConfig,
    PxDesignProcessorConfig,
    PxDesignSamplerConfig,
)
from .model import (
    PxDesignInput,
    PxDesignModel,
    PxDesignSamplingOutput,
)
from .pipeline import (
    PxDesignBackbonePipeline,
    PxDesignBackboneResult,
    PxDesignCandidate,
    PxDesignFilterResult,
    PxDesignPipeline,
    PxDesignPipelineInput,
    PxDesignPipelineOutput,
    PxDesignResult,
    generated_components,
)
from .processor import PxDesignProcessor, TargetChain, from_binder, to_binder

MODELS.register(
    "pxdesign",
    model=PxDesignModel,
    processor=PxDesignProcessor,
)
PIPELINES.register(
    "pxdesign_backbone",
    pipeline=PxDesignBackbonePipeline,
    model_architecture=PxDesignBackbonePipeline.model_architecture,
)
PIPELINES.register(
    "pxdesign",
    pipeline=PxDesignPipeline,
)
