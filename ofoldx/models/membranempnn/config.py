r"""MembraneMPNN architecture configuration."""

from __future__ import annotations

from chanfig import Config


class MembraneMpnnConfig(Config):
    r"""Configure the shared graph of the two membrane-label checkpoints.

    Attributes:
        hidden_size: Width of node, edge and decoder hidden states.
        num_encoder_blocks: Number of residue graph encoder blocks.
        num_decoder_blocks: Number of autoregressive decoder blocks.
        num_neighbors: Number of nearest C-alpha neighbours.
        dropout: Probability of encoder and decoder dropout.
    """

    hidden_size: int = 128
    num_encoder_blocks: int = 3
    num_decoder_blocks: int = 3
    num_neighbors: int = 48
    dropout: float = 0.0
