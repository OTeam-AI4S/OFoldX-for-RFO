"""Behavior of ofoldx.models.grnade.objective."""

import torch
import torch.nn.functional as F

from ofoldx.data.tokenization import TOKENS
from ofoldx.models.grnade import GrnadeLoss, GrnadeOutput, GrnadeProcessor
from ofoldx.models.grnade.model import GRNADE_CLASSES


def test_loss_on_predicted_classes(rna):
    structure = rna(2)
    prepared, target = GrnadeProcessor(dtype=torch.float64).prepare(structure, target=structure)
    logits = torch.full((2, len(TOKENS)), -torch.inf, dtype=torch.float64)
    logits[:, list(GRNADE_CLASSES)] = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, 1.0]], dtype=torch.float64)
    loss = GrnadeLoss()(GrnadeOutput(logits=logits, batch_size=[]), target, mask=prepared.features.design_mask)
    expected = F.cross_entropy(logits[:, list(GRNADE_CLASSES)], torch.zeros(2, dtype=torch.long), label_smoothing=0.05)
    torch.testing.assert_close(loss, expected)
