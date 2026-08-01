r"""Backbone and membrane-label preparation on ProteinMPNN's prepared rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import torch
from torch import Tensor

from ...data.crop import Cropper
from ...data.molecular import ComponentRef
from ...data.structure import Structure
from ...errors import RequestRef, fail
from ...model import Model
from ...processor import Processor
from ..conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..design import PositionSelection
from ..layout import LayoutTable
from ..preparation import PreparedInput
from ..proteinmpnn.model import ProteinMpnnInput
from ..proteinmpnn.processor import PROTEINMPNN_LAYOUT, ProteinMpnnProcessor


class MembraneMpnnProcessor(ProteinMpnnProcessor):
    r"""
    Prepare N, CA, C, O backbones and scalar or per-residue membrane labels.

    A scalar label is broadcast to every row; a mapping labels the named residues and leaves the others zero. The
    per-residue checkpoint reads zero = exposed or other, one = interface, two = buried; the global checkpoint zero =
    soluble, one = transmembrane. A tied group's members default to coefficient one (upstream LigandMPNN repository),
    not ProteinMPNN's ``1 / len(group)``.
    """

    config_class = None

    def __init__(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        Processor.__init__(self, device=device, dtype=dtype)

    @property
    def layout(self) -> LayoutTable:
        return PROTEINMPNN_LAYOUT

    def validate_model_input_layout(self, model: Model) -> None:
        pass

    def prepare_input(  # type: ignore[override]
        self,
        structure: Structure,
        *,
        membrane_labels: int | Mapping[ComponentRef, int] = 0,
        design: PositionSelection = PositionSelection(),
        sequence: str | Mapping[str, str] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[ProteinMpnnInput]:
        r"""Prepare ProteinMPNN rows with membrane labels.

        See `ProteinMpnnProcessor.prepare_input` for the remaining arguments.
        """
        prepared = super().prepare_input(
            structure,
            design=design,
            sequence=sequence,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            filter_by_occupancy=filter_by_occupancy,
            cropper=cropper,
            generator=generator,
        )
        system = structure.system
        input = prepared.features
        rows = input.component_index[0].cpu().numpy()
        if isinstance(membrane_labels, Mapping):
            named = system.component_index(list(membrane_labels))
            if (named < 0).any():
                reference = list(membrane_labels)[int(np.flatnonzero(named < 0)[0])]
                fail("request_invalid_positions", RequestRef("membrane_labels", str(reference)))
            labels = np.zeros(len(system.component_symbol), dtype=np.int64)
            labels[named] = list(membrane_labels.values())
            input.membrane_labels = torch.as_tensor(labels[rows], device=self.device).unsqueeze(0)
        else:
            input.membrane_labels = torch.full_like(input.sequence_tokens, membrane_labels)
        if tied_weights:
            row_of = {component: row for row, component in enumerate(rows.tolist())}
            weights = torch.ones_like(cast(Tensor, input.tied_weights))
            for component, weight in zip(system.component_index(list(tied_weights)).tolist(), tied_weights.values()):
                weights[0, row_of[component]] = weight
            input.tied_weights = weights
        return prepared
