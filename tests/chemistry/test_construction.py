from __future__ import annotations

import pytest

from ofoldx.chemistry.construction import (
    Admissibility,
    Molecule,
    Recipe,
    RecipeBond,
    build_system,
    instantiate_component,
    polymer_sequence,
    replacement_admissibility,
)
from ofoldx.chemistry.smiles import read_smiles
from ofoldx.data.molecular import AtomSite, BondKind, BondOrigin, MonomerType
from ofoldx.errors import CODE_NAMES, ContradictionError, InvalidRequestError


def names(system, component: int) -> set[str]:
    start, stop = system.component_atom_ptr[component : component + 2]
    return set(system.text("atom", system.atom_symbol[start:stop]).tolist())


@pytest.fixture
def dipeptide(archive, library):
    r"""``G-A`` built from the CCD subset definitions."""
    recipe = Recipe({"1": polymer_sequence("GA", MonomerType.PROTEIN)}, {"A": "1"})
    system, _, _ = build_system(recipe, library([archive("GLY"), archive("ALA")]))
    return system


class TestPolymerSequence:
    @pytest.mark.parametrize(
        ("text", "monomer", "components", "polymer_type"),
        [
            ("GUOX", MonomerType.PROTEIN, ("GLY", "SEC", "PYL", "UNK"), "polypeptide(L)"),
            ("ACTN", MonomerType.RNA, ("A", "C", "U", "N"), "polyribonucleotide"),
            ("ACUI", MonomerType.DNA, ("DA", "DC", "DT", "DI"), "polydeoxyribonucleotide"),
        ],
        ids=["protein", "rna", "dna"],
    )
    def test_symbols_to_components(self, text, monomer, components, polymer_type):
        sequence = polymer_sequence(text, monomer)
        assert (sequence.components, sequence.polymer_type) == (components, polymer_type)

    @pytest.mark.parametrize("text", ["AB", "A-", "Ag"])
    def test_ambiguous_symbol_refused(self, text):
        with pytest.raises(InvalidRequestError) as caught:
            polymer_sequence(text, MonomerType.PROTEIN)
        assert caught.value.code == "request_invalid_sequence"
        assert caught.value.detail["position"] == 1


class TestBuildSystem:
    def test_junction_removes_leaving_atoms(self, dipeptide):
        assert not {"OXT", "HXT"} & names(dipeptide, 0)
        assert "H2" not in names(dipeptide, 1)
        junction = dipeptide.bond_origin == BondOrigin.JUNCTION
        assert junction.sum() == 1
        assert dipeptide.component_topology_complete.all()

    def test_smiles_ligand_component(self, ligand):
        system = ligand("OCC")
        assert len(system.atom_symbol) == 9
        assert system.text("component", system.component_symbol).tolist() == ["LIG"]

    def test_recipe_bond_unknown_atom(self, library):
        component = read_smiles("CC", component_id="LIG").component
        recipe = Recipe(
            {"L": Molecule(("LIG", "LIG"))},
            {"L": "L"},
            (RecipeBond(AtomSite("L", 0, "C1"), AtomSite("L", 1, "X9"), 1, BondKind.COVALENT),),
        )
        with pytest.raises(InvalidRequestError) as caught:
            build_system(recipe, library([]), {"LIG": component})
        assert caught.value.code == "value_invalid"

    def test_conflicting_caller_refused(self, archive, library):  # F2: no component is left without its atoms
        component = read_smiles("N[C@@H](C)C(=O)O", component_id="ALA").component
        with pytest.raises(ContradictionError) as caught:
            build_system(Recipe({"L": Molecule(("ALA",))}, {"L": "L"}), library([archive("ALA")]), {"ALA": component})
        assert caught.value.code == "read_conflicts"


class TestInstantiateComponent:
    @pytest.mark.parametrize(
        ("linked", "absent"),
        [((False, False), set()), ((True, False), {"H2"}), ((True, True), {"H2", "OXT", "HXT"})],
        ids=["free", "previous", "both"],
    )
    def test_linked_ports_lose_leaving(self, archive, linked, absent):
        rows = instantiate_component(archive("ALA"), linked=linked)
        assert set(archive("ALA").atom_names) - set(rows.names) == absent
        assert rows.claims == (True, True)

    def test_unverified_port_refused(self):
        component = read_smiles("CC", component_id="LIG").component
        with pytest.raises(InvalidRequestError) as caught:
            instantiate_component(component, linked=(True, False))
        assert caught.value.code == "value_invalid"


class TestReplacementAdmissibility:
    def test_shared_port_admissible(self, dipeptide, archive):
        check = replacement_admissibility(dipeptide, 0, archive("ALA"))
        assert check.outcome.tolist() == [Admissibility.ADMISSIBLE]

    def test_absent_endpoint_inadmissible(self, dipeptide):
        check = replacement_admissibility(dipeptide, 0, read_smiles("CC", component_id="LIG").component)
        assert check.outcome.tolist() == [Admissibility.INADMISSIBLE]
        assert CODE_NAMES[check.reason[0]] == "replacement_atom_absent"
