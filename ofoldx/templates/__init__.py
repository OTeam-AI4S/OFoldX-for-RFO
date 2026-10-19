r"""
Structural templates: polymer-chain coordinates projected from a read structure, target alignments with their
resolution and placement frames, and the native template artifact.
"""

from .alignment import (
    ResolvedTemplateAlignment,
    Template,
    TemplateAlignment,
    resolve_template,
    template_alignment_frames,
)
from .coordinates import ChainCoordinates, PolymerCoordinates, chain_coordinates, template_atoms
from .files import load_templates, save_templates

__all__ = [
    "ChainCoordinates",
    "PolymerCoordinates",
    "ResolvedTemplateAlignment",
    "Template",
    "TemplateAlignment",
    "chain_coordinates",
    "load_templates",
    "resolve_template",
    "save_templates",
    "template_alignment_frames",
    "template_atoms",
]
