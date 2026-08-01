"""Builder of the Boltz-2 tests: templates of a structure's chains."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ofoldx.data.structure import Structure, StructureArrays
from ofoldx.provenance import Basis, Resolution, state
from ofoldx.templates import Template, TemplateAlignment, chain_coordinates


@pytest.fixture
def template() -> Callable[..., Template]:
    """``template(structure, "A", (target_instance, ((0, 0), (1, 1))), ...)``: a template of chain ``A`` of
    ``structure`` (author chains named as the instances), aligned to entity ``P``."""

    def make(structure: Structure, chain: str, *alignments: tuple[str | None, tuple]) -> Template:
        arrays = structure.build()
        arrays.columns["instance_author_chain"][:] = structure.system.instance_symbol
        arrays.columns["instance_author_chain_state"][:] = state(Resolution.KNOWN, Basis.DECLARED)
        symbols = {**arrays.symbols, "chain": tuple(structure.system.symbols()["instance"])}
        source = Structure.from_arrays(structure.system, StructureArrays(symbols, arrays.columns))
        return Template(
            chain_coordinates(source, [chain]),
            tuple(TemplateAlignment("P", target, chain, pairs) for target, pairs in alignments),
        )

    return make
