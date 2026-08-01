import pytest

from ofoldx.models.mpnn_packer import (
    MpnnPackerConfig,
    MpnnPackerModel,
    MpnnPackerPipeline,
    MpnnPackerProcessor,
    MpnnPackerProcessorConfig,
)


@pytest.fixture
def small_packer():
    config = MpnnPackerConfig(
        hidden_size=8,
        edge_size=8,
        num_encoder_blocks=1,
        num_decoder_blocks=1,
        num_rbf=2,
        num_neighbors=1,
        num_context_atoms=1,
        num_mix=1,
        dropout=0.0,
    )
    return MpnnPackerPipeline(
        model=MpnnPackerModel(config).eval(),
        processor=MpnnPackerProcessor(MpnnPackerProcessorConfig(num_context_atoms=1)),
    )
