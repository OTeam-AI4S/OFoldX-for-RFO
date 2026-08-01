r"""
Task pipeline utilities.
"""

from __future__ import annotations

from .base import Pipeline
from .composite import ComponentSpec, CompositePipeline
from .generator import resolve_generator
from .paired import PairedComponent, PairedPipeline
from .registry import (
    PIPELINES,
    PipelineRegistration,
    PipelineTask,
)
