r"""Copy canonical NumPy columns to model tensors.

Canonical arrays are read-only. Model tensors receive independent storage on the
requested device, so modifying a feature leaves the source value unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from ..data.molecular import MolecularSystem
from ..data.structure import Structure


def to_tensors(
    value: Structure | MolecularSystem, columns: Sequence[str], *, device: torch.device
) -> Mapping[str, torch.Tensor]:
    r"""Copy named canonical columns to PyTorch tensors on a device.

    Args:
        value: A molecular system or structure supplying the columns. A structure
            exposes both its own columns and its system's columns.
        columns: Names of the columns to copy.
        device: Destination device.

    Returns:
        Tensors indexed by column name, with storage independent of the source arrays.
        Memory use is proportional to the selected columns.
    """

    own = value.arrays()
    system = value.system.arrays() if isinstance(value, Structure) else own
    return {name: torch.from_numpy(np.array(own[name] if name in own else system[name])).to(device) for name in columns}
