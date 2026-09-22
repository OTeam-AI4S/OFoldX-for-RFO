r"""
Sources to canonical values and back: mmCIF and PDB reading through one read path (``read_structure``), writing,
native artifacts and CIF syntax. Multiple sequence alignments live in ``ofoldx.msa``.
"""

from __future__ import annotations

from .cif import parse_cif, read_cif
from .native import load_definitions, load_structure, save_structure
from .read import read_structure
from .report import ReadReport
from .source import Source, parse_source, read_source
from .tables import DocumentMetadata
from .write import WriteReport, write_structure

__all__ = [
    "read_source",
    "parse_source",
    "Source",
    "read_structure",
    "ReadReport",
    "DocumentMetadata",
    "write_structure",
    "WriteReport",
    "save_structure",
    "load_structure",
    "load_definitions",
    "read_cif",
    "parse_cif",
]
