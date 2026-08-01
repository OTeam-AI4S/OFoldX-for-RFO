import pytest
import torch

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.models.nampnn import NaMpnnConfig, NaMpnnModel


@pytest.fixture
def complex_structure(library, placed):
    """Chain ``A``: RNA ``A A``; chain ``B``: protein ``ALA GLY``; placed."""
    entities = {
        "R": PolymerSequence("polyribonucleotide", ("A", "A")),
        "P": PolymerSequence("polypeptide(L)", ("ALA", "GLY")),
    }
    system, _, _ = build_system(Recipe(entities, {"A": "R", "B": "P"}), library)
    return placed(system)


@pytest.fixture
def model():
    torch.manual_seed(83)
    values = {"hidden_size": 8, "feature_size": 8, "num_encoder_blocks": 1, "num_decoder_blocks": 1}
    values.update(num_neighbors=3, dropout=0.0, positional_embedding_size=4, num_rbf=2, include_predicted_na_n=False)
    return NaMpnnModel(NaMpnnConfig(values)).eval()
