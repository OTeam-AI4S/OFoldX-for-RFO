r"""
OInvFold polymer and ligand inverse-folding implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    OInvFoldLigandConfig,
    OInvFoldPolymerConfig,
    OInvFoldPolymerProcessorConfig,
)
from .model import (
    OINVFOLD_LIGAND_ELEMENTS,
    OInvFoldGraph,
    OInvFoldLigandInput,
    OInvFoldLigandModel,
    OInvFoldOutput,
    OInvFoldPolymerInput,
    OInvFoldPolymerModel,
)
from .objective import OInvFoldLoss
from .pipeline import (
    OInvFoldLigandCandidate,
    OInvFoldLigandPipeline,
    OInvFoldLigandResult,
    OInvFoldPolymerPipeline,
    OInvFoldPolymerResult,
)
from .processor import (
    OInvFoldLigandProcessor,
    OInvFoldPolymerProcessor,
)

MODELS.register(
    "oinvfold",
    model=OInvFoldPolymerModel,
    processor=OInvFoldPolymerProcessor,
)
MODELS.register(
    "oinvfold_ligand",
    model=OInvFoldLigandModel,
    processor=OInvFoldLigandProcessor,
)
PIPELINES.register(
    "oinvfold",
    pipeline=OInvFoldPolymerPipeline,
    model_architecture=OInvFoldPolymerPipeline.model_architecture,
)
PIPELINES.register(
    "oinvfold_ligand",
    pipeline=OInvFoldLigandPipeline,
    model_architecture=OInvFoldLigandPipeline.model_architecture,
)
