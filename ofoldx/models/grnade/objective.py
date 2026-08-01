r"""
gRNAde training objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..targets import SequenceTarget
from .model import GRNADE_CLASSES, GrnadeOutput


class GrnadeLoss(nn.Module):
    r"""
    Cross entropy over the designable residues of a predicted class, on the four predicted classes (upstream's axis,
    so label smoothing spreads over them only).

    Args:
        label_smoothing: Uniform label-smoothing mass distributed over the predicted classes.
    """

    def __init__(self, *, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred: GrnadeOutput, target: SequenceTarget, *, mask: Tensor | None = None) -> Tensor:
        r"""Score aligned sequence targets within an optional task domain.

        Args:
            pred: Predicted canonical-class logits.
            target: Aligned classes and their knownness mask.
            mask: Optional scoring domain, intersected with target knownness.

        Returns:
            Mean sequence loss over the selected rows.
        """
        classes = torch.tensor(GRNADE_CLASSES, device=target.tokens.device)
        matches = target.tokens.unsqueeze(-1) == classes
        known_design = target.mask & matches.any(dim=-1)
        if mask is not None:
            known_design = known_design & mask.bool()
        return F.cross_entropy(
            pred.logits[known_design][:, classes],
            matches[known_design].long().argmax(dim=-1),
            label_smoothing=self.label_smoothing,
        )
