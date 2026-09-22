"""Behavior of ofoldx.analysis.structure."""

from math import pi

import numpy as np
import pytest
from biotite.structure.info import vdw_radius_single

from ofoldx.analysis.structure import (
    assign_secondary_structure,
    structure_coordinates,
    structure_design_lddt,
    structure_rmsd_metrics,
    target_delta_sasa,
)
from ofoldx.chemistry.construction import Molecule, PolymerSequence, Recipe, build_system
from ofoldx.chemistry.smiles import read_smiles
from ofoldx.data.molecular import AtomSite, ComponentRef
from ofoldx.data.structure import CoordinateEvidence, Structure, replace_coordinates


@pytest.fixture
def place(library):
    """``place(recipe, sites, points, components)``: the recipe's system with only ``sites`` placed at ``points``."""

    def make(recipe: Recipe, sites: list[AtomSite], points: np.ndarray, components=None) -> Structure:
        system, _, _ = build_system(recipe, library, components or {})
        structure, _ = replace_coordinates(
            Structure.unobserved(system),
            system.atom_index(sites),
            np.asarray(points, dtype=np.float64).reshape(-1, 3),
            scope=np.ones(len(system.atom_symbol), dtype=bool),
            evidence=CoordinateEvidence.OBSERVED,
        )
        return structure

    return make


@pytest.fixture
def waters(place):
    """Two SMILES water ligands ``LIG`` (Biotite drops residues named ``HOH`` as solvent)."""
    recipe = Recipe({"W": Molecule(("LIG",))}, {"A": "W", "B": "W"})
    sites, points = [AtomSite("A", 0, "O1"), AtomSite("B", 0, "O1")], [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    return place(recipe, sites, points, {"LIG": read_smiles("O", component_id="LIG").component})


@pytest.fixture
def backbones(place):
    """Protein chains of ALA (PRO where named) with only N, CA, C, O placed from ``[residues, 4, 3]`` arrays."""

    def make(*chains, prolines=(), linked=None) -> Structure:
        entities, instances, sites = {}, {}, []
        for k, points in enumerate(chains):
            ids = tuple("PRO" if (k, p) in prolines else "ALA" for p in range(len(points)))
            entities[str(k)] = PolymerSequence("polypeptide(L)", ids, None if linked is None else linked[k])
            instances[f"chain{k}"] = str(k)
            sites += [AtomSite(f"chain{k}", p, name) for p in range(len(points)) for name in ("N", "CA", "C", "O")]
        return place(Recipe(entities, instances), sites, np.concatenate(chains))

    return make


class TestSiteMatching:
    def test_identical_metrics(self, peptide):
        structure = peptide("ALA", "GLY", "ALA")
        design, target = {ComponentRef("A", 1)}, {ComponentRef("A", 0), ComponentRef("A", 2)}

        metrics = structure_rmsd_metrics(structure, structure, designed_positions=design, target=target)

        assert all(value == pytest.approx(0.0, abs=1e-6) for value in metrics.values())
        assert structure_design_lddt(structure, structure, designed_positions=design) == 1.0

    def test_backbone_sites(self, peptide):
        structure = peptide("ALA", "GLY", "ALA")

        reference, predicted, sites = structure_coordinates(structure, structure, backbone_only=True)

        assert {site.atom for site in sites} == {"N", "CA", "C", "O"}
        assert len(sites) == 12
        np.testing.assert_array_equal(reference, predicted)

    def test_unplaced_sites_excluded(self, backbones, secondary_backbones):
        structure = backbones(secondary_backbones["helix"][:2])

        _, _, sites = structure_coordinates(structure, structure)

        assert {site.atom for site in sites} == {"N", "CA", "C", "O"}


class TestSasa:
    def test_isolated_water(self, waters):
        result = target_delta_sasa(waters, target={ComponentRef("A", 0)}, designed_positions={ComponentRef("B", 0)})

        assert result.delta == pytest.approx(0.0)
        assert result.target_unbound == pytest.approx(4 * pi * (vdw_radius_single("O") + 1.4) ** 2, rel=1e-6)

    def test_requires_heavy_coordinates(self, library):
        system, _, _ = build_system(Recipe({"W": Molecule(("HOH",))}, {"A": "W"}), library)
        unplaced = Structure.unobserved(system)

        with pytest.raises(ValueError, match="heavy"):
            target_delta_sasa(unplaced, target={ComponentRef("A", 0)}, designed_positions=())


class TestSecondaryStructure:
    def test_cross_chain_beta(self, backbones, secondary_backbones):
        backbone = secondary_backbones["beta"]

        assert assign_secondary_structure(backbones(backbone[:2], backbone[2:])) == {
            ComponentRef("chain0", 0): "L",
            ComponentRef("chain0", 1): "E",
            ComponentRef("chain1", 0): "L",
            ComponentRef("chain1", 1): "E",
        }
        proline = backbones(backbone[:2], backbone[2:], prolines=((0, 1),))
        assert set(assign_secondary_structure(proline).values()) == {"L"}

    def test_helix_context(self, backbones, secondary_backbones):
        backbone = secondary_backbones["helix"]
        selected = (ComponentRef("chain0", 1), ComponentRef("chain0", 2))

        assert assign_secondary_structure(backbones(backbone), selected) == dict.fromkeys(selected, "H")
        broken = backbones(backbone, linked=((True, True, True, False, True, True),))
        assert set(assign_secondary_structure(broken).values()) == {"L"}

    def test_requires_coordinates(self, place, secondary_backbones):
        recipe = Recipe({"0": PolymerSequence("polypeptide(L)", ("ALA", "ALA"))}, {"chain0": "0"})
        backbone = secondary_backbones["helix"][:2]
        sites = [AtomSite("chain0", p, name) for p in range(2) for name in ("N", "CA", "C", "O")]
        structure = place(recipe, sites[:-1], backbone.reshape(-1, 3)[:-1])

        with pytest.raises(ValueError, match="secondary structure requires"):
            assign_secondary_structure(structure, components={ComponentRef("chain0", 1)})
        assert set(assign_secondary_structure(structure, components={ComponentRef("chain0", 0)})) == {
            ComponentRef("chain0", 0)
        }
