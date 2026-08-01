"""Builders shared by the ESMFold2 tests."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.msa import MSA


@pytest.fixture
def referenced_repository(tmp_path, esm_config):
    from multimolecule import EsmCConfig, EsmCModel

    from ofoldx.models.esmfold2 import EsmFold2Model, EsmFold2Pipeline, EsmFold2Processor, EsmFold2ProcessorConfig

    config = esm_config(False)
    backbone = EsmCModel(EsmCConfig(**config.esmc.dict()), add_pooling_layer=False)
    config.esmc_repository = str(tmp_path / "esmc")
    backbone.save_pretrained(config.esmc_repository)
    pipeline = EsmFold2Pipeline(
        model=EsmFold2Model(config, esmc=backbone).eval().set_backend("eager"),
        processor=EsmFold2Processor(EsmFold2ProcessorConfig(msa_conditioning=False)),
    )
    destination = tmp_path / "native"
    pipeline.save(destination)
    return destination, pipeline


@pytest.fixture
def msa_pool():
    """``msa_pool(query, *rows)``: the pool of the ``peptide`` fixture's chain ``A``, with zero deletions."""

    def build(query, *rows):
        unpaired = (query, *rows)
        counts = np.zeros((len(unpaired), len(query)), dtype=np.uint32)
        return {"A": MSA("P", unpaired, counts, (query,), np.zeros((1, len(query)), dtype=np.uint32), ("query",))}

    return build
