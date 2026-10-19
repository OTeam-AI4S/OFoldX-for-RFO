r"""
Multiple sequence alignments: reading A3M and aligned-FASTA files, indexed storage, pairing aligned sequences, and
projecting alignment rows onto tokens.
"""

from .alignment import MSA, MsaCollection, MsaRows, msa_rows, unique_rows
from .files import IndexedMsaLoader, load_msa_collection, read_msa, save_msa_collection
from .pairing import pair_msas

__all__ = [
    "MSA",
    "IndexedMsaLoader",
    "MsaCollection",
    "MsaRows",
    "load_msa_collection",
    "msa_rows",
    "pair_msas",
    "read_msa",
    "save_msa_collection",
    "unique_rows",
]
