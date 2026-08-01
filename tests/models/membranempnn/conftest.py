from __future__ import annotations

import pytest
import torch

from ofoldx.models.membranempnn import (
    MembraneMpnnConfig,
    MembraneMpnnModel,
    MembraneMpnnPipeline,
    MembraneMpnnProcessor,
)


@pytest.fixture
def designer() -> MembraneMpnnPipeline:
    torch.manual_seed(83)
    config = MembraneMpnnConfig(hidden_size=8, num_encoder_blocks=1, num_decoder_blocks=1, num_neighbors=2)
    return MembraneMpnnPipeline(model=MembraneMpnnModel(config).eval(), processor=MembraneMpnnProcessor())
