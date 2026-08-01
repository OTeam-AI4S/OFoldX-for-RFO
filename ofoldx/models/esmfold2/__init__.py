r"""
ESMFold2 folding and experimental design implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    EsmFold2AtomEncoderConfig,
    EsmFold2ConfidenceHeadConfig,
    EsmFold2Config,
    EsmFold2DenoiserConfig,
    EsmFold2DiffusionStructureHeadConfig,
    EsmFold2EsmCConfig,
    EsmFold2ExperimentalConfidenceHeadConfig,
    EsmFold2ExperimentalConfig,
    EsmFold2ExperimentalProcessorConfig,
    EsmFold2FoldingStackConfig,
    EsmFold2InputEmbeddingsConfig,
    EsmFold2LanguageModelEncoderConfig,
    EsmFold2MsaEncoderConfig,
    EsmFold2ParcaeConfig,
    EsmFold2ProcessorConfig,
)
from .model import (
    EsmFold2ExperimentalConfidenceOutput,
    EsmFold2ExperimentalModel,
    EsmFold2ExperimentalSamplingOutput,
    EsmFold2ExperimentalSamplingWithConfidenceOutput,
    EsmFold2Input,
    EsmFold2Model,
    EsmFold2Output,
    EsmFold2SamplingWithConfidenceOutput,
)
from .pipeline import (
    EsmFold2DesignCandidate,
    EsmFold2DesignPipeline,
    EsmFold2DesignPipelineInput,
    EsmFold2DesignPipelineOutput,
    EsmFold2DesignResult,
    EsmFold2DesignStep,
    EsmFold2ExperimentalPipeline,
    EsmFold2HeroCriticEvaluation,
    EsmFold2Pipeline,
    EsmFold2ScalingCriticEvaluation,
    cdr_distogram_iptm_proxy,
)
from .processor import (
    EsmFold2ExperimentalProcessor,
    EsmFold2Processor,
    antibody_cdr_indices,
    antibody_cdr_prompt,
    is_antibody_sequence,
)

MODELS.register(
    "esmfold2",
    model=EsmFold2Model,
    processor=EsmFold2Processor,
)

MODELS.register(
    "esmfold2_experimental",
    model=EsmFold2ExperimentalModel,
    processor=EsmFold2ExperimentalProcessor,
)

PIPELINES.register(
    "esmfold2",
    pipeline=EsmFold2Pipeline,
    model_architecture=EsmFold2Pipeline.model_architecture,
)

PIPELINES.register(
    "esmfold2_experimental",
    pipeline=EsmFold2ExperimentalPipeline,
    model_architecture=EsmFold2ExperimentalPipeline.model_architecture,
)

PIPELINES.register(
    "esmfold2_design",
    pipeline=EsmFold2DesignPipeline,
)
