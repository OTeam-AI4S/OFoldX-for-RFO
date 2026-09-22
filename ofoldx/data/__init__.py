r"""Molecular identities, coordinates, and evidence for structure workflows.

Immutable NumPy values support molecular selection, assembly, chemical edits, and
coordinate replacement. This layer is independent of PyTorch, RDKit, and source readers.
"""

from .assembly import batch, replicate, unbatch, union
from .editing import ChemicalEdit, apply_edit
from .molecular import AtomSite, BondKind, ComponentRef, EndpointState, MolecularSystem, MonomerType
from .projection import fold_hydrogens, project_atoms, project_instances, select
from .structure import CoordinateEvidence, CopyId, CopyKind, Structure, replace_coordinates
from .tokenization import TOKENS, TokenLayout, token_classes, tokenize

__all__ = [
    "TOKENS",
    "AtomSite",
    "BondKind",
    "ChemicalEdit",
    "ComponentRef",
    "CoordinateEvidence",
    "CopyId",
    "CopyKind",
    "EndpointState",
    "MolecularSystem",
    "MonomerType",
    "Structure",
    "TokenLayout",
    "apply_edit",
    "batch",
    "fold_hydrogens",
    "project_atoms",
    "project_instances",
    "replace_coordinates",
    "replicate",
    "select",
    "token_classes",
    "tokenize",
    "unbatch",
    "union",
]
