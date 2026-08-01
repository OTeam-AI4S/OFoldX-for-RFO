r"""
AlphaFold 2 monomer and multimer implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    AlphaFold2Config,
    AlphaFold2EvoformerBlockConfig,
    AlphaFold2EvoformerConfig,
    AlphaFold2HeadsConfig,
    AlphaFold2InitialGuessConfig,
    AlphaFold2InputEmbeddingsConfig,
    AlphaFold2MsaConfig,
    AlphaFold2MultimerConfig,
    AlphaFold2MultimerHeadsConfig,
    AlphaFold2MultimerInputEmbeddingsConfig,
    AlphaFold2MultimerProcessorConfig,
    AlphaFold2MultimerStructureModuleConfig,
    AlphaFold2MultimerTemplateConfig,
    AlphaFold2ProcessorConfig,
    AlphaFold2RecyclingConfig,
    AlphaFold2StructureModuleConfig,
    AlphaFold2TemplateConfig,
)
from .model import (
    AlphaFold2InitialGuessInput,
    AlphaFold2InitialGuessModel,
    AlphaFold2Input,
    AlphaFold2Model,
    AlphaFold2MultimerInput,
    AlphaFold2MultimerModel,
    AlphaFold2Output,
    AlphaFold2StructureModuleOutput,
    alphafold2_predicted_interface_tm_score,
    alphafold2_predicted_tm_score,
)
from .objective import AlphaFold2MaskedMsaLoss
from .pipeline import (
    AlphaFold2Casp14Pipeline,
    AlphaFold2Casp14Prediction,
    AlphaFold2Casp14Result,
    AlphaFold2InitialGuessPipeline,
    AlphaFold2MultimerPipeline,
    AlphaFold2Pipeline,
)
from .processor import AlphaFold2InitialGuessProcessor, AlphaFold2MultimerProcessor, AlphaFold2Processor
from .relax import (
    AlphaFold2AmberRelaxation,
    AlphaFold2RelaxationResult,
    alphafold2_structural_violations,
)

MODELS.register(
    "alphafold2",
    model=AlphaFold2Model,
    processor=AlphaFold2Processor,
)
MODELS.register(
    "alphafold2_initial_guess",
    model=AlphaFold2InitialGuessModel,
    processor=AlphaFold2InitialGuessProcessor,
)
MODELS.register(
    "alphafold2_multimer",
    model=AlphaFold2MultimerModel,
    processor=AlphaFold2MultimerProcessor,
)
PIPELINES.register(
    "alphafold2",
    pipeline=AlphaFold2Pipeline,
    model_architecture=AlphaFold2Pipeline.model_architecture,
)
PIPELINES.register(
    "alphafold2_casp14",
    pipeline=AlphaFold2Casp14Pipeline,
)
PIPELINES.register(
    "alphafold2_initial_guess",
    pipeline=AlphaFold2InitialGuessPipeline,
    model_architecture=AlphaFold2InitialGuessPipeline.model_architecture,
)
PIPELINES.register(
    "alphafold2_multimer",
    pipeline=AlphaFold2MultimerPipeline,
    model_architecture=AlphaFold2MultimerPipeline.model_architecture,
)
