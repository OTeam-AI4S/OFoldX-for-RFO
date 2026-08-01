r"""
Learned contact-class and distance-threshold pair conditioning.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .fourier import FrozenFourierEmbedding


class ContactConditioning(nn.Module):
    r"""Encode contact classes and distance thresholds as an additive pair bias.

    Contact class coefficients have shape ``[..., N_source, N_target, num_classes]`` and
    thresholds have shape ``[..., N_source, N_target]``. The first two classes select learned
    unspecified and unselected biases; the remaining classes and normalized
    threshold drive the learned constraint encoding. The caller owns the
    scientific meaning and selection of the active contact classes.
    """

    def __init__(
        self,
        pair_size: int,
        *,
        min_cutoff: float,
        max_cutoff: float,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff
        self.threshold_fourier = FrozenFourierEmbedding(pair_size)
        self.constraint_encoder = nn.Linear(pair_size + num_classes - 1, pair_size)
        self.unspecified_bias = nn.Parameter(torch.zeros(pair_size))
        self.unselected_bias = nn.Parameter(torch.zeros(pair_size))

    def forward(self, contact_class: Tensor, contact_threshold: Tensor) -> Tensor:
        r"""Return the encoded pair bias with shape ``[..., N_source, N_target, pair_size]``."""

        threshold = (contact_threshold - self.min_cutoff) / (self.max_cutoff - self.min_cutoff)
        encoded = self.constraint_encoder(
            torch.cat([contact_class[..., 2:], threshold.unsqueeze(-1), self.threshold_fourier(threshold)], dim=-1)
        )
        active = 1 - contact_class[..., :2].sum(dim=-1, keepdim=True)
        return (
            encoded * active
            + self.unspecified_bias * contact_class[..., :1]
            + self.unselected_bias * contact_class[..., 1:2]
        )
