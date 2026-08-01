r"""
OInvFold polymer training objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...errors import RequestRef, fail
from ..processing import model_log_probabilities
from ..targets import SequenceTarget
from .model import OInvFoldOutput


class OInvFoldLoss(nn.Module):
    r"""Average released polymer negative log likelihood over designable nodes."""

    def forward(self, pred: OInvFoldOutput, target: SequenceTarget, *, mask: Tensor | None = None) -> Tensor:
        r"""Score aligned sequence targets within an optional task domain.

        Args:
            pred: Predicted canonical-class logits.
            target: Aligned classes and their knownness mask.
            mask: Optional scoring domain, intersected with target knownness.

        Returns:
            Mean sequence loss over the selected rows.
        """
        mask = target.mask if mask is None else target.mask & mask.bool()
        sequence_tokens = target.tokens[mask]
        logits = pred.logits[mask]
        log_probabilities = model_log_probabilities(
            logits, torch.ones_like(sequence_tokens, dtype=torch.bool), feature="OInvFold scored logits"
        )
        scorable = torch.isfinite(log_probabilities.gather(-1, sequence_tokens.unsqueeze(-1))).all()
        if torch.compiler.is_compiling():
            torch._assert_async(scorable, "OInvFold scored classes must be classes the model predicts")
        elif not bool(scorable):
            fail("model_capacity", RequestRef("target", "a scored class the model does not predict"))
        return F.nll_loss(log_probabilities, sequence_tokens)
