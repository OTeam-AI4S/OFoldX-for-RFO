"""Behaviour of ofoldx.models.conditions: placements, symbolic sites and symbolic bonds."""

from __future__ import annotations

import pytest

from ofoldx.data import AtomSite, ComponentRef
from ofoldx.errors import InvalidRequestError
from ofoldx.models.conditions import (
    ExistingSpan,
    GeneratedSegment,
    SymbolicAtomSite,
    SymbolicBond,
    SymbolicResidueSite,
    placement_instance_of,
    resolve_bond_site,
)

A, B = AtomSite("chain_0", 0, "C"), AtomSite("chain_1", 0, "N")
PLACEMENT_MAP = {"binder": {"loop": (ComponentRef("D", 0), ComponentRef("D", 1))}}


class TestSymbolicBond:
    def test_undirected_ignores_order(self):
        bond = SymbolicBond(A, B, leaving_a=frozenset({"OXT"}))
        swapped = SymbolicBond(B, A, leaving_b=frozenset({"OXT"}))
        assert bond == swapped and hash(bond) == hash(swapped)

    def test_leaving_ids_follow_site(self):
        assert SymbolicBond(A, B, leaving_a=frozenset({"OXT"})) != SymbolicBond(B, A, leaving_a=frozenset({"OXT"}))

    def test_directed_compares_in_order(self):
        assert SymbolicBond(A, B, direction="a_to_b") != SymbolicBond(B, A, direction="a_to_b")

    def test_self_bond_is_invalid(self):
        with pytest.raises(InvalidRequestError):
            SymbolicBond(A, A)

    def test_resolve_names_realised_atom(self):
        site = SymbolicAtomSite(SymbolicResidueSite("loop", -1), "CA")
        resolved = SymbolicBond(A, site).resolve(PLACEMENT_MAP, {"loop": "binder"})
        assert resolved.site_b == AtomSite("D", 1, "CA") and resolved.site_a == A


class TestPlacements:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: GeneratedSegment("protein", 3, 2),
            lambda: ExistingSpan("A", 2, 1),
        ],
    )
    def test_inverted_bounds_are_invalid(self, make):
        with pytest.raises(InvalidRequestError):
            make()

    def test_placement_ids_are_unique(self):
        with pytest.raises(InvalidRequestError):
            placement_instance_of({"a": {"x": GeneratedSegment("protein", 1, 1)}, "b": {"x": ExistingSpan("A", 0, 1)}})

    def test_offset_past_placement_invalid(self):
        site = SymbolicAtomSite(SymbolicResidueSite("loop", 2), "CA")
        with pytest.raises(InvalidRequestError):
            resolve_bond_site(site, PLACEMENT_MAP, {"loop": "binder"})
