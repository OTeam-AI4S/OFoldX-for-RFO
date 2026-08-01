r"""
Shared mathematical modules.
"""

from __future__ import annotations

from .attention import GatedMultiheadAttentionProjection, MultiheadAttention, MultiheadAttentionProjection
from .dropout import StructuredDropout
from .feed_forward import FeedForward, SwiGluFeedForward
from .geometry import (
    atom37_to_torsion_angles,
    dihedral_angle,
    frames_to_atom14_positions,
    rotation_from_two_vectors,
    torsion_angles_to_frames,
    uniform_random_rotation,
    virtual_beta_carbon,
    weighted_rigid_align,
)
from .transition import ConditionedTransition, FeedForwardTransition, Transition
