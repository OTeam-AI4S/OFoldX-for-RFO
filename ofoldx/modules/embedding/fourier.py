r"""
Fourier feature embeddings.

Random Fourier features lift a scalar (e.g. a diffusion timestep) into a higher-dimensional
periodic embedding that a network can use for diffusion timesteps and other scalar conditions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenFourierEmbedding(nn.Module):
    r"""
    Frozen random Fourier features ``cos(2*pi * projection(x))``.

    A single ``1 -> embedding_size`` linear layer with standard-normal weight and bias is drawn once, frozen
    (``requires_grad_(False)``), and reused as a fixed random projection; taking the cosine of the
    scaled projection yields ``embedding_size`` periodic features per scalar input.

    Args:
        embedding_size: Number of Fourier feature channels.
    """

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, embedding_size)
        self.reset_parameters()
        self.projection.requires_grad_(False)

    def reset_parameters(self) -> None:
        r"""
        Re-draw the frozen projection weight and bias from a standard normal distribution.
        """

        nn.init.normal_(self.projection.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.projection.bias, mean=0.0, std=1.0)

    def forward(self, times: Tensor) -> Tensor:
        r"""
        Return Fourier features for scalar times.

        Args:
            times: ``[...]`` scalar values (e.g. diffusion timesteps) to embed; a trailing
                channel axis is added internally before the projection.

        Returns:
            ``[..., embedding_size]`` Fourier features ``cos(2*pi * projection(times))``.
        """

        projected = self.projection(times.to(self.projection.weight).unsqueeze(-1))
        return torch.cos(2 * torch.pi * projected)
