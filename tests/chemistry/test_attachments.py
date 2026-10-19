from __future__ import annotations

from ofoldx.chemistry.attachments import rule_l
from ofoldx.data.molecular import EndpointState
from ofoldx.errors import CODE_NAMES

from .builders import edge, problem

S = EndpointState


def names(component, atoms) -> set[str]:
    return {component.atom_names[a] for a in atoms}


def reason(outcome) -> str:
    return CODE_NAMES[outcome.reason]


class TestRuleL:
    def test_designated_group_leaves_whole(self, items, glycine, define):
        columns = glycine()
        columns["atoms"]["type_symbol"][9] = "C"  # HXT as a carbon: a two-heavy-atom group designated at C
        ester = define(items(**columns))
        outcome = rule_l(problem(ester, "C", [edge(1)], observed={"O", "CA"}), ester)
        assert (outcome.state, names(ester, outcome.atoms)) == (S.DERIVED_FROM_REFERENCE, {"OXT", "HXT"})
        assert outcome.phantom == ()

    def test_unknown_element_has_no_class(self, items, glycine, define):
        columns = glycine()
        columns["atoms"]["type_symbol"][0] = "?"  # N of unknown element: neither class A nor class B
        gly = define(items(**columns))
        outcome = rule_l(problem(gly, "N", [edge(1)], observed={"CA"}), gly)
        assert (outcome.state, reason(outcome)) == (S.UNKNOWN, "reference_incomplete_absent")
