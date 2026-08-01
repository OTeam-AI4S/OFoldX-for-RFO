r"""
Pairwise distance prediction.
"""

from torch import Tensor, nn


class DistogramHead(nn.Module):
    r"""
    Project pair representations and add the transposed distance-bin logits.

    The projection's bias contributes twice. A checkpoint that projects the
    symmetrized pair representation maps its source bias to half this value.
    """

    def __init__(self, pair_size: int = 128, num_bins: int = 64, bias: bool = True) -> None:
        super().__init__()
        self.projection = nn.Linear(pair_size, num_bins, bias=bias)

    def forward(self, pair: Tensor) -> Tensor:
        logits = self.projection(pair)
        return logits + logits.transpose(-2, -3)
