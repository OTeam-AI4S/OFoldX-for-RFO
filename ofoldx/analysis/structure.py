r"""
Molecular-site matching, scoped structural analysis, and annotations.

This module contains coordinate, alignment, interaction, and secondary-structure
operations shared by real model families. It does not own candidate filtering,
ranking, diversity reduction, or protocol selection.

A component's kind (protein, nucleic acid, other) is the monomer type of its canonical token class
(``CLASS_MONOMER[token_classes(...)]``), so a modified residue counts with its parent's polymer.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection
from dataclasses import dataclass

import biotite.structure as struc
import hydride
import numpy as np
from biotite.structure.info import vdw_radius_single
from numpy.typing import NDArray

from ..data.molecular import ELEMENTS, AtomSite, BondKind, ComponentRef, MolecularSystem, MonomerType
from ..data.structure import Structure
from ..data.tokenization import CLASS_MONOMER, token_classes
from ..metrics import structure as structure_metrics
from ..provenance import known

_PROTEIN_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})
_DNA_BACKBONE_ATOMS = frozenset({"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"})
_RNA_BACKBONE_ATOMS = _DNA_BACKBONE_ATOMS | {"O2'"}
_BACKBONE_ATOMS = {
    MonomerType.PROTEIN: _PROTEIN_BACKBONE_ATOMS,
    MonomerType.RNA: _RNA_BACKBONE_ATOMS,
    MonomerType.DNA: _DNA_BACKBONE_ATOMS,
}
#: Relative RMSD alignment weight per monomer type (NONE, PROTEIN, RNA, DNA): protein 1, nucleic acid 6, other 11.
_RMSD_WEIGHTS = np.array([11.0, 1.0, 6.0, 6.0])
_HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP"})
_BOND_TYPES = {
    1: struc.BondType.SINGLE,
    2: struc.BondType.DOUBLE,
    3: struc.BondType.TRIPLE,
    4: struc.BondType.QUADRUPLE,
}

# Parameters from ProtOr (Tsai et al., 1999) and Biotite structure/info/radii.py,
# including MET/halogen extensions: (element, covalent neighbors including H, total H).
_PROTOR_GROUP_RADII = {
    ("C", 3, 0): 1.61,
    ("C", 3, 1): 1.76,
    ("C", 4, 1): 1.88,
    ("C", 4, 2): 1.88,
    ("C", 4, 3): 1.88,
    ("N", 3, 0): 1.64,
    ("N", 3, 1): 1.64,
    ("N", 3, 2): 1.64,
    ("N", 4, 3): 1.64,
    ("O", 1, 0): 1.42,
    ("O", 2, 1): 1.46,
    ("S", 1, 0): 1.77,
    ("S", 2, 0): 1.77,
    ("S", 2, 1): 1.77,
    ("F", 1, 0): 1.47,
    ("CL", 1, 0): 1.75,
    ("BR", 1, 0): 1.85,
    ("I", 1, 0): 1.98,
}


@dataclass(frozen=True)
class NoncovalentContacts:
    r"""
    Released design-to-target noncovalent contact counts.

    Attributes:
        num_hydrogen_bonds: Hydride/Biotite donor--acceptor hydrogen bonds.
        num_charged_pairs: Oppositely charged atom pairs at ``0.5 < d < 5.5`` Å.
    """

    num_hydrogen_bonds: int
    num_charged_pairs: int


@dataclass(frozen=True)
class TargetDeltaSasa:
    r"""
    Target burial caused by a designed component.

    Attributes:
        delta: Target SASA in isolation minus target SASA in the bound complex.
        target_unbound: Target SASA in isolation.
        target_bound: Target SASA in the bound target--design complex.
    """

    delta: float
    target_unbound: float
    target_bound: float


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _monomer(system: MolecularSystem) -> NDArray[np.uint8]:
    r"""Per component, the monomer type of its canonical token class."""
    return CLASS_MONOMER[token_classes(system).token]


def _component_instance(system: MolecularSystem) -> NDArray[np.int64]:
    return _owner(system.instance_component_ptr)


def _members(system: MolecularSystem, components: Collection[ComponentRef]) -> NDArray[np.int64]:
    r"""The component indices of ``components`` (references naming no component of ``system`` are ignored)."""
    index = system.component_index(list(components))
    return index[index >= 0]


def _atom_sites(structure: Structure) -> tuple[AtomSite, ...]:
    system = structure.system
    refs = system.component_refs(_owner(system.component_atom_ptr))
    names = system.text("atom", system.atom_symbol).tolist()
    return tuple(AtomSite(ref.instance, ref.component, name) for ref, name in zip(refs, names))


def _backbone_atoms(system: MolecularSystem) -> NDArray[np.bool_]:
    monomer = _monomer(system)[_owner(system.component_atom_ptr)]
    names = system.text("atom", system.atom_symbol)
    backbone = np.zeros(len(names), dtype=bool)
    for kind, atoms in _BACKBONE_ATOMS.items():
        backbone |= (monomer == kind) & np.isin(names, tuple(atoms))
    return backbone


def _matched(
    reference: Structure,
    predicted: Structure,
    components: Collection[ComponentRef] | None,
    backbone_only: bool,
) -> tuple[NDArray[np.int64], NDArray[np.int64], tuple[AtomSite, ...]]:
    r"""Reference atoms with coordinates whose exact site has coordinates in ``predicted``, and the predicted atoms."""
    sites = _atom_sites(reference)
    matched = predicted.system.atom_index(sites)
    keep = reference.coordinate_mask & (matched >= 0)
    keep[keep] = predicted.coordinate_mask[matched[keep]]
    if components is not None:
        owner = _owner(reference.system.component_atom_ptr)
        keep &= np.isin(owner, _members(reference.system, components))
    if backbone_only:
        keep &= _backbone_atoms(reference.system)
    rows = np.flatnonzero(keep)
    return rows, matched[rows], tuple(sites[row] for row in rows.tolist())


def structure_coordinates(
    reference: Structure,
    predicted: Structure,
    *,
    components: Collection[ComponentRef] | None = None,
    backbone_only: bool = False,
) -> tuple[NDArray, NDArray, tuple[AtomSite, ...]]:
    r"""
    Return the coordinates of exact atom sites present in both structures.

    The output order follows the reference canonical atom axis. ``components``
    selects exact component occurrences; ``backbone_only`` uses the shared
    protein and nucleic-acid backbone atom definitions.
    """

    rows, matched, sites = _matched(reference, predicted, components, backbone_only)
    return reference.coordinates[rows], predicted.coordinates[matched], sites


def structure_rmsd_metrics(
    original: Structure,
    refolded: Structure,
    *,
    designed_positions: Collection[ComponentRef],
    target: Collection[ComponentRef],
    backbone_only: bool = False,
) -> dict[str, float]:
    r"""
    Return the shared all/design/target subset RMSDs.

    Exact atom sites present in both structures are matched first.  Whole,
    design, and target alignments use the shared relative atom weights:
    protein ``1``, nucleic acid ``6``, and other components ``11``.  RMSDs measured
    on the design after whole- or target-only alignment are unweighted.
    ``design_rmsd`` and ``target_rmsd`` fit and measure their own subsets.
    ``complex_aligned_design_rmsd`` fits all matched sites and measures the
    exact designed positions; ``target_aligned_design_rmsd`` fits only target
    sites before measuring those same designed positions. A target alignment is
    supported when the target holds a polymer component or more than three sites.
    """

    rows, matched, _ = _matched(original, refolded, None, backbone_only)
    reference, predicted = original.coordinates[rows], refolded.coordinates[matched]
    system = original.system
    owner = _owner(system.component_atom_ptr)[rows]
    monomer = _monomer(system)
    targets = _members(system, target)
    design_mask = np.isin(owner, _members(system, designed_positions))
    target_mask = np.isin(owner, targets)
    all_mask = np.ones(len(rows), dtype=bool)
    weights = _RMSD_WEIGHTS[monomer[owner]]
    unweighted = np.ones(len(rows))
    target_alignment_supported = bool((monomer[targets] != MonomerType.NONE).any()) or int(target_mask.sum()) > 3

    def subset(align_mask: NDArray[np.bool_], measure_mask: NDArray[np.bool_], *, weighted: bool = True) -> float:
        if not align_mask.any() or not measure_mask.any():
            return float("nan")
        return structure_metrics.aligned_rmsd(
            reference,
            predicted,
            align_mask=align_mask,
            measure_mask=measure_mask,
            weights=weights,
            measure_weights=weights if weighted else unweighted,
        )

    metrics = {
        "rmsd": subset(all_mask, all_mask),
        "design_rmsd": subset(design_mask, design_mask),
        "target_rmsd": subset(target_mask, target_mask),
        "complex_aligned_design_rmsd": float("nan"),
        "target_aligned_design_rmsd": float("nan"),
    }
    if target_alignment_supported:
        metrics["complex_aligned_design_rmsd"] = subset(all_mask, design_mask, weighted=False)
        metrics["target_aligned_design_rmsd"] = subset(target_mask, design_mask, weighted=False)
    return metrics


def structure_design_lddt(
    original: Structure,
    refolded: Structure,
    *,
    designed_positions: Collection[ComponentRef],
    backbone_only: bool = False,
) -> float:
    r"""
    Return shared same-chain intra-design lDDT for one coherent refold.

    Exact atom sites present in both structures are paired.  Only pairs of
    designed atoms from the same molecular instance contribute.
    """

    rows, matched, _ = _matched(original, refolded, None, backbone_only)
    system = original.system
    owner = _owner(system.component_atom_ptr)[rows]
    is_design = np.isin(owner, _members(system, designed_positions))
    instance = _component_instance(system)[owner]
    pair_mask = is_design[:, None] & is_design[None, :] & (instance[:, None] == instance[None, :])
    return structure_metrics.lddt(original.coordinates[rows], refolded.coordinates[matched], pair_mask=pair_mask)


class _SecondaryStructurePremiseError(ValueError):
    """An unsupported local backbone, distinct from an unexpected implementation failure."""


def _adjacency(system: MolecularSystem) -> dict[int, list[tuple[int, int]]]:
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for edge, (first, second) in enumerate(system.edge_index.T.tolist()):
        adjacency.setdefault(first, []).append((second, edge))
        adjacency.setdefault(second, []).append((first, edge))
    return adjacency


def assign_secondary_structure(
    structure: Structure,
    components: Collection[ComponentRef] | None = None,
) -> dict[ComponentRef, str]:
    r"""Assign simplified DSSP classes in established peptide context.

    Selected backbones require N/CA/C/O coordinates, the expected
    elements and KNOWN bond orders, and closed local amide chemistry with KNOWN
    hydrogen counts. Virtual donor geometry is used only for an established
    N-H; substituted no-H nitrogen is not a donor. Missing coordinates or unsupported
    chemistry in unselected components interrupt context. No stereo projection is
    required. Neutral or protonated free terminal amines with local carbon
    substituents are supported without a virtual donor H. Other charged nitrogen
    forms are unsupported. Cross-chain beta contacts
    and actual cyclic peptide edges remain available.

    Args:
        structure: Canonical protein chemistry and coordinates.
        components: Exact output components, or all protein components when omitted.

    Returns:
        Simplified ``L/H/E`` assignments for the supported selected backbones.

    Raises:
        ValueError: A selected component lacks required coordinates or supported chemistry.
    """
    selected = None if components is None else frozenset(components)
    system = structure.system
    instances = _component_instance(system)
    refs = system.component_refs(np.arange(len(system.component_symbol)))
    adjacency = _adjacency(system)
    references: list[ComponentRef] = []
    backbone: list[NDArray] = []
    nitrogen_rows: dict[int, int] = {}
    carbon_rows: dict[int, int] = {}
    donors: list[bool] = []
    for component in np.flatnonzero(_monomer(system) == MonomerType.PROTEIN).tolist():
        reference = refs[component]
        try:
            atoms, donor = _dssp_backbone(structure, component, int(instances[component]), adjacency, reference)
        except _SecondaryStructurePremiseError:
            if selected is None or reference in selected:
                raise
            continue
        nitrogen, _, carbon, _ = atoms
        nitrogen_rows[nitrogen] = len(references)
        carbon_rows[carbon] = len(references)
        references.append(reference)
        backbone.append(structure.coordinates[list(atoms)])
        donors.append(donor)
    if not backbone:
        return {}
    peptide_edges = []
    for first, second in system.edge_index.T.tolist():
        if first in carbon_rows and second in nitrogen_rows:
            peptide_edges.append((carbon_rows[first], nitrogen_rows[second]))
        elif second in carbon_rows and first in nitrogen_rows:
            peptide_edges.append((carbon_rows[second], nitrogen_rows[first]))
    assignment = structure_metrics.backbone_secondary_structure(
        np.stack(backbone),
        np.asarray(peptide_edges, dtype=np.int64).reshape(-1, 2).T,
        np.asarray(donors, dtype=np.bool_),
    )
    return {
        reference: str(value)
        for reference, value in zip(references, assignment, strict=True)
        if selected is None or reference in selected
    }


def _dssp_backbone(
    structure: Structure,
    component: int,
    instance: int,
    adjacency: dict[int, list[tuple[int, int]]],
    reference: ComponentRef,
) -> tuple[tuple[int, int, int, int], bool]:
    system = structure.system

    def require(condition: bool, premise: str) -> None:
        if not condition:
            raise _SecondaryStructurePremiseError(f"secondary structure requires {premise} for {reference!r}")

    start, stop = (int(value) for value in system.component_atom_ptr[component : component + 2])
    names = system.text("atom", system.atom_symbol[start:stop]).tolist()
    by_name = {name: start + offset for offset, name in enumerate(names)}
    require(all(name in by_name for name in ("N", "CA", "C", "O")), "an identified N/CA/C/O backbone")
    nitrogen, alpha, carbon, oxygen = (by_name[name] for name in ("N", "CA", "C", "O"))
    atoms = nitrogen, alpha, carbon, oxygen
    require(
        bool(structure.coordinate_mask[list(atoms)].all() and np.isfinite(structure.coordinates[list(atoms)]).all()),
        "complete backbone coordinates",
    )
    require(tuple(system.atomic_number[list(atoms)].tolist()) == (7, 6, 6, 8), "backbone N/C/C/O elements")
    require(
        bool(system.instance_connections_complete[instance])
        and bool(system.component_topology_complete[component])
        and bool(system.component_heavy_atoms_complete[component])
        and bool(system.component_hydrogens_complete[component]),
        "closed backbone connectivity and hydrogen inventory",
    )

    def order(edge: int) -> int:
        require(
            int(system.bond_kind[edge]) == BondKind.COVALENT
            and bool(known(system.bond_kind_state[edge]))
            and bool(known(system.bond_order_state[edge]))
            and bool(known(system.bond_aromatic_state[edge]))
            and not bool(system.bond_aromatic[edge]),
            "known nonaromatic covalent backbone bonds",
        )
        return int(system.bond_order[edge])

    incident: dict[int, dict[int, int]] = {}
    hydrogen_counts: dict[int, int] = {}
    for atom in (nitrogen, carbon, oxygen):
        require(
            bool(known(system.formal_charge_state[atom]))
            and int(system.formal_charge[atom]) in ((0, 1) if atom == nitrogen else (0,))
            and bool(known(system.atom_aromatic_state[atom]))
            and not bool(system.atom_aromatic[atom]),
            "supported known backbone chemistry",
        )
        require(bool(known(system.implicit_hydrogens_state[atom])), "known backbone attached hydrogen counts")
        bonds = {neighbor: order(edge) for neighbor, edge in adjacency.get(atom, ())}
        incident[atom] = bonds
        represented_hydrogens = [neighbor for neighbor in bonds if int(system.atomic_number[neighbor]) == 1]
        for hydrogen in represented_hydrogens:
            require(
                bonds[hydrogen] == 1
                and len(adjacency[hydrogen]) == 1
                and bool(known(system.formal_charge_state[hydrogen]))
                and int(system.formal_charge[hydrogen]) == 0,
                "neutral singly attached backbone hydrogens",
            )
        hydrogen_counts[atom] = int(system.implicit_hydrogens[atom]) + len(represented_hydrogens)
    require(incident[nitrogen].get(alpha) == 1, "a single N-CA bond")
    require(incident[carbon].get(alpha) == 1 and incident[carbon].get(oxygen) == 2, "CA-C and carbonyl C=O bonds")
    require(
        hydrogen_counts[carbon] == 0
        and hydrogen_counts[oxygen] == 0
        and incident[oxygen] == {carbon: 2}
        and len(incident[carbon]) == 3
        and sum(incident[carbon].values()) == 4
        and all(
            neighbor in (alpha, oxygen) or (int(system.atomic_number[neighbor]) in (7, 8) and bond_order == 1)
            for neighbor, bond_order in incident[carbon].items()
        ),
        "a supported peptide carbonyl",
    )
    nitrogen_heavy = [neighbor for neighbor in incident[nitrogen] if int(system.atomic_number[neighbor]) != 1]
    terminal_amine = all(start <= neighbor < stop and neighbor != carbon for neighbor in nitrogen_heavy)
    nitrogen_charge = int(system.formal_charge[nitrogen])
    require(
        all(
            int(system.atomic_number[neighbor]) == 6 and incident[nitrogen][neighbor] == 1
            for neighbor in nitrogen_heavy
        )
        and len(nitrogen_heavy) + hydrogen_counts[nitrogen] == 3 + nitrogen_charge
        and (terminal_amine or nitrogen_charge == 0),
        "a neutral internal amide or supported free terminal amine",
    )
    if nitrogen_charge == 1:
        require(
            all(order(edge) == 1 for neighbor in nitrogen_heavy for _, edge in adjacency.get(neighbor, ())),
            "a free terminal amine with saturated carbon substituents",
        )
    return atoms, not terminal_amine and hydrogen_counts[nitrogen] > 0


def count_noncovalent_contacts(
    structure: Structure,
    *,
    designed_positions: Collection[ComponentRef],
    connected_designed_positions: Collection[ComponentRef],
) -> NoncovalentContacts:
    r"""
    Count shared design-to-target hydrogen bonds and charged atom pairs.

    Hydrogens are added with Hydride and hydrogen bonds are assigned with
    Biotite. Charged pairs use the strict distance interval ``0.5 < d < 5.5`` Å.
    Complete heavy-atom coordinates and closed, KNOWN covalent
    chemistry are required; missing facts raise ``ValueError``.
    """

    expected_hydrogens = _contact_hydrogen_inventory(structure)
    atom_array, _ = _biotite_atom_array(
        structure,
        designed_positions=designed_positions,
        connected_designed_positions=connected_designed_positions,
        chemistry=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atom_array, original_mask = hydride.add_hydrogen(atom_array)
        actual_hydrogens = [
            sum(atom_array.element[int(neighbor)] == "H" for neighbor in atom_array.bonds.get_bonds(int(index))[0])
            for index in np.flatnonzero(original_mask)
        ]
        if actual_hydrogens != expected_hydrogens:
            raise ValueError("Hydride hydrogen completion disagrees with the canonical hydrogen inventory")
        hydrogen_bonds = struc.hbond(atom_array)

    donors = hydrogen_bonds[:, 0]
    acceptors = hydrogen_bonds[:, 2]
    charge = atom_array.get_annotation("charge")
    is_design = atom_array.get_annotation("is_design")
    is_chain_design = atom_array.get_annotation("is_chain_design")
    design_donors = is_design[donors] & ~is_chain_design[acceptors]
    design_acceptors = ~is_chain_design[donors] & is_design[acceptors]
    num_hydrogen_bonds = int(np.sum(design_donors) + np.sum(design_acceptors))

    positive = np.flatnonzero(charge > 0)
    negative = np.flatnonzero(charge < 0)
    num_charged_pairs = 0
    if len(positive) and len(negative):
        distances = np.linalg.norm(atom_array.coord[positive, None] - atom_array.coord[None, negative], axis=-1)
        positive_index, negative_index = np.where((distances > 0.5) & (distances < 5.5))
        positive_design = is_design[positive[positive_index]] & ~is_chain_design[negative[negative_index]]
        negative_design = ~is_chain_design[positive[positive_index]] & is_design[negative[negative_index]]
        num_charged_pairs = int(np.sum(positive_design) + np.sum(negative_design))
    return NoncovalentContacts(num_hydrogen_bonds, num_charged_pairs)


def _contact_hydrogen_inventory(structure: Structure) -> list[int]:
    r"""Per heavy atom in canonical order, its total attached hydrogens (represented atoms plus the implicit count)."""
    system = structure.system
    if not (system.instance_connections_complete.all() and system.component_topology_complete.all()):
        raise ValueError("contact analysis requires closed molecular connectivity")
    if not system.component_hydrogens_complete.all():
        raise ValueError("contact analysis requires complete hydrogen inventory")
    if not known(system.formal_charge_state).all():
        raise ValueError("contact analysis requires known formal charges")
    if not (known(system.bond_kind_state).all() and (system.bond_kind == BondKind.COVALENT).all()):
        raise ValueError("Hydride contact analysis requires covalent bond definitions")
    heavy = system.atomic_number != 1
    first, second = system.edge_index.astype(np.int64)
    ends, partners = np.concatenate((first, second)), np.concatenate((second, first))
    hydrogen = ~heavy
    attached = hydrogen[ends]
    degree = np.bincount(ends[attached], minlength=len(heavy))
    if (
        (degree[hydrogen] != 1).any()
        or (system.atomic_number[partners[attached]] == 1).any()
        or (system.formal_charge[hydrogen] != 0).any()
    ):
        raise ValueError("Hydride contact analysis requires neutral hydrogens attached to heavy atoms")
    counts = system.implicit_hydrogens.astype(np.int64) + np.bincount(partners[attached], minlength=len(heavy))
    return counts[heavy].tolist()


def target_delta_sasa(
    structure: Structure,
    *,
    target: Collection[ComponentRef],
    designed_positions: Collection[ComponentRef],
) -> TargetDeltaSasa:
    r"""
    Return target delta-SASA with the shared 1.4 Å/960-point/ProtOr operation.

    Each selected component needs complete heavy-atom inventory and coordinates.
    ProtOr united-atom radii use actual element, covalent neighbor count and total
    attached H, including implicit counts, when local topology, connection and
    H inventories are closed. Other groups use the known elemental radius.
    No residue/atom-name lookup, charge or CIP inference is used. Missing required
    coordinates or elemental radii raise ``ValueError``.

    Args:
        structure: Canonical graph and heavy-atom coordinates.
        target: Exact target component occurrences.
        designed_positions: Components occluding the target in the bound calculation.

    Returns:
        Target unbound/bound accessible areas and their difference, in square Å.

    Raises:
        ValueError: Required heavy atoms, coordinates or elemental radii are unavailable.
    """

    target = frozenset(target)
    bound_array, bound_atoms = _biotite_atom_array(structure, components=target | frozenset(designed_positions))
    bound_sasa = _atom_sasa(structure, bound_array, bound_atoms)
    owner = _owner(structure.system.component_atom_ptr)
    target_in_bound = np.isin(owner[bound_atoms], _members(structure.system, target))
    target_bound = float(np.sum(bound_sasa[target_in_bound]))

    target_array, target_atoms = _biotite_atom_array(structure, components=target)
    target_unbound = float(np.sum(_atom_sasa(structure, target_array, target_atoms)))
    return TargetDeltaSasa(target_unbound - target_bound, target_unbound, target_bound)


def largest_hydrophobic_patch_area(
    structure: Structure,
    *,
    components: Collection[ComponentRef] | None = None,
    distance_cutoff: float = 6.0,
) -> float:
    r"""
    Return the largest connected exposed hydrophobic-carbon patch area.

    Exposed carbon atoms in the shared hydrophobic residue set are connected
    transitively when their distance is at most ``distance_cutoff`` Å.
    Complete selected heavy-atom inventory and coordinates are required.
    SASA uses the same actual-group ProtOr approximation and elemental fallback as
    ``target_delta_sasa``; missing coordinates or elemental radii raise ``ValueError``.

    Args:
        structure: Canonical graph and heavy-atom coordinates.
        components: Exact component occurrences, or the complete structure.
        distance_cutoff: Maximum distance in Å joining exposed eligible carbons.

    Returns:
        Largest connected patch area in square Å, or NaN when no eligible carbon
        has exposed area. Residue eligibility uses the existing named residue set;
        carbon membership uses actual elements.

    Raises:
        ValueError: Required heavy atoms, coordinates or elemental radii are unavailable.
    """

    atom_array, atoms = _biotite_atom_array(structure, components=components)
    atom_sasa = _atom_sasa(structure, atom_array, atoms)
    mask = (
        np.isin(atom_array.res_name, tuple(_HYDROPHOBIC_RESIDUES))
        & (np.char.upper(atom_array.element) == "C")
        & (atom_sasa > 0)
    )
    coordinates = atom_array.coord[mask]
    areas = atom_sasa[mask]
    if not len(coordinates):
        return float("nan")

    adjacent = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1) <= distance_cutoff
    unseen = set(range(len(coordinates)))
    largest = 0.0
    while unseen:
        frontier = [unseen.pop()]
        component = []
        while frontier:
            index = frontier.pop()
            component.append(index)
            neighbors = {int(neighbor) for neighbor in np.flatnonzero(adjacent[index])} & unseen
            unseen.difference_update(neighbors)
            frontier.extend(neighbors)
        largest = max(largest, float(np.sum(areas[component])))
    return largest


def _biotite_atom_array(
    structure: Structure,
    *,
    components: Collection[ComponentRef] | None = None,
    designed_positions: Collection[ComponentRef] = (),
    connected_designed_positions: Collection[ComponentRef] = (),
    chemistry: bool = False,
) -> tuple[struc.AtomArray, NDArray[np.int64]]:
    r"""The heavy atoms of the selected components as a Biotite array, and their canonical atom indices."""
    system = structure.system
    if components is None:
        selected = np.arange(len(system.component_symbol))
    else:
        selected = system.component_index(list(components))
        if (selected < 0).any():
            raise ValueError("analysis components must name components of the structure")
    refs = system.component_refs(selected)
    for component, reference in zip(selected.tolist(), refs):
        if not system.component_heavy_atoms_complete[component]:
            raise ValueError(f"analysis requires complete heavy-atom inventory for {reference!r}")
    owner = _owner(system.component_atom_ptr)
    atoms = np.flatnonzero(np.isin(owner, selected) & (system.atomic_number != 1))
    missing = (
        (system.atomic_number[atoms] == 0)
        | ~structure.coordinate_mask[atoms]
        | ~np.isfinite(structure.coordinates[atoms]).all(axis=1)
    )
    if missing.any():
        (reference,) = system.component_refs(owner[atoms[missing][:1]])
        raise ValueError(f"analysis requires identified heavy atoms with complete coordinates for {reference!r}")
    components_of = owner[atoms]
    instance = _component_instance(system)
    atom_array: struc.AtomArray = struc.AtomArray(len(atoms))
    atom_array.coord = structure.coordinates[atoms]
    atom_array.chain_id = instance[components_of].astype(str)
    atom_array.res_id = components_of - system.instance_component_ptr[instance[components_of]] + 1
    atom_array.ins_code = np.full(len(atoms), "")
    atom_array.atom_name = system.text("atom", system.atom_symbol[atoms])
    atom_array.res_name = system.text("component", system.component_symbol[components_of])
    atom_array.element = np.asarray(ELEMENTS)[system.atomic_number[atoms]]
    atom_array.hetero = _monomer(system)[components_of] == MonomerType.NONE
    if chemistry:
        if not known(system.formal_charge_state[atoms]).all():
            raise ValueError("contact analysis requires known formal charges")
        atom_array.set_annotation("charge", system.formal_charge[atoms].astype(np.int64))
    atom_array.set_annotation("is_design", np.isin(components_of, _members(system, designed_positions)))
    atom_array.set_annotation("is_chain_design", np.isin(components_of, _members(system, connected_designed_positions)))

    if chemistry:
        local = np.full(len(system.atom_symbol), -1, dtype=np.int64)
        local[atoms] = np.arange(len(atoms))
        first, second = system.edge_index.astype(np.int64)
        inside = np.flatnonzero((local[first] >= 0) & (local[second] >= 0))
        bonds: list[tuple[int, int, int]] = []
        for edge in inside.tolist():
            if (
                system.bond_kind[edge] != BondKind.COVALENT
                or not known(system.bond_kind_state[edge])
                or not known(system.bond_order_state[edge])
                or not known(system.bond_aromatic_state[edge])
            ):
                raise ValueError("contact analysis requires known covalent bond order and aromaticity")
            bond_type = (
                struc.BondType.AROMATIC if system.bond_aromatic[edge] else _BOND_TYPES[int(system.bond_order[edge])]
            )
            bonds.append((int(local[first[edge]]), int(local[second[edge]]), int(bond_type)))
        atom_array.bonds = struc.BondList(len(atom_array), np.asarray(bonds, dtype=np.uint32).reshape(-1, 3))
    return atom_array, atoms


def _atom_sasa(structure: Structure, atom_array: struc.AtomArray, atoms: NDArray[np.int64]) -> NDArray:
    system = structure.system
    elements = system.atomic_number.astype(np.int64)
    first, second = system.edge_index.astype(np.int64)
    settled = known(system.bond_kind_state)
    covalent = settled & (system.bond_kind == BondKind.COVALENT)
    ends, partners = np.concatenate((first, second)), np.concatenate((second, first))
    unsettled_edges = np.concatenate((~settled, ~settled))
    covalent_ends = np.concatenate((covalent, covalent))
    count = len(elements)
    degree = np.bincount(ends[covalent_ends], minlength=count)
    vertex_hydrogens = np.bincount(ends[covalent_ends & (elements[partners] == 1)], minlength=count)
    unsettled = np.bincount(ends[unsettled_edges | (covalent_ends & (elements[partners] == 0))], minlength=count) > 0
    owner = _owner(system.component_atom_ptr)
    instance = _component_instance(system)[owner]
    closed = (
        ~unsettled
        & system.component_topology_complete[owner]
        & system.instance_connections_complete[instance]
        & system.component_heavy_atoms_complete[owner]
        & system.component_hydrogens_complete[owner]
        & known(system.implicit_hydrogens_state)
    )
    implicit = system.implicit_hydrogens.astype(np.int64)
    radii: list[float] = []
    for atom, element in zip(atoms.tolist(), atom_array.element, strict=True):
        radius = None
        if closed[atom]:
            radius = _PROTOR_GROUP_RADII.get(
                (str(element).upper(), int(degree[atom] + implicit[atom]), int(vertex_hydrogens[atom] + implicit[atom]))
            )
        if radius is None:
            radius = vdw_radius_single(str(element))
        if radius is None:
            raise ValueError(f"SASA has no van der Waals radius for element {element!r}")
        radii.append(radius)
    return struc.sasa(
        atom_array,
        probe_radius=1.4,
        point_number=960,
        vdw_radii=np.asarray(radii, dtype=np.float64),
    )
