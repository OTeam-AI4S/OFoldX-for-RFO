from __future__ import annotations

import pytest
import torch

from ofoldx.models.ligandmpnn import (
    LigandMpnnConfig,
    LigandMpnnModel,
    LigandMpnnPipeline,
    LigandMpnnProcessor,
    LigandMpnnProcessorConfig,
)


@pytest.fixture
def designer() -> LigandMpnnPipeline:
    torch.manual_seed(79)
    config = LigandMpnnConfig(hidden_size=8, num_encoder_blocks=1, num_decoder_blocks=1, num_neighbors=2)
    return LigandMpnnPipeline(
        model=LigandMpnnModel(config).eval(),
        processor=LigandMpnnProcessor(LigandMpnnProcessorConfig(num_context_atoms=2)),
    )
