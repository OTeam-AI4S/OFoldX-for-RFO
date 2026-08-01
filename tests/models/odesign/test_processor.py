"""Behavior of ofoldx.models.odesign.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data import Structure
from ofoldx.data.molecular import ComponentRef, MolecularSystem
from ofoldx.data.projection import project_arrays, project_instances
from ofoldx.data.structure import CoordinateEvidence
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import InvalidRequestError, UnsupportedError
from ofoldx.models.design import PositionSelection
from ofoldx.models.odesign import ODesignLigandProcessor, ODesignProcessor, ODesignProcessorConfig
from ofoldx.msa import MSA
from ofoldx.provenance import Basis, Resolution, state

FIRST = PositionSelection(components=(ComponentRef("A", 0),))
SECOND = PositionSelection(components=(ComponentRef("A", 1),))


def slot_names(structure, prepared) -> list[str]:
    system = structure.system
    return system.text("atom", system.atom_symbol[prepared.context.slots.atom]).tolist()


def one_hot(prepared, token: str, position: int, plane: int = 0) -> float:
    return float(prepared.features.token_features[0, position, plane * len(TOKENS) + TOKEN_INDEX[token]])


@pytest.fixture
def ligand_truth(peptide):
    target = peptide("ALA", "GLY", ligands=("ACT", "ACT"))
    context, _ = project_instances(target, np.array([True, False, False]))
    return context, target


class TestPrepare:
    def test_generated_identity_hidden(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = ODesignProcessor().prepare_input(structure)
        assert [one_hot(prepared, "X", position) for position in (0, 1)] == [1.0, 1.0]
        assert [one_hot(prepared, "-", position, plane=1) for position in (0, 1)] == [1.0, 1.0]
        assert prepared.features.design_features[0].tolist() == [[1, 0, 0, 0], [1, 0, 0, 0]]
        assert set(slot_names(structure, prepared)) == {"N", "CA", "C", "O"}

    def test_context_kept_whole(self, peptide):
        structure = peptide("ALA", "GLY")
        prepared = ODesignProcessor().prepare_input(structure, design=SECOND)
        assert one_hot(prepared, "A", 0) == 1.0
        assert "CB" in slot_names(structure, prepared)
        assert prepared.context.scope.rule == "atoms"

    def test_flexible_receptor_scope(self, peptide):
        structure = peptide("ALA", "GLY")
        atoms = structure.system.component_atom_ptr
        rigid = ODesignProcessor(ODesignProcessorConfig(artifact="protein_rigid"))
        designed = slice(int(atoms[0]), int(atoms[1]))
        context = slice(int(atoms[1]), int(atoms[2]))
        assert ODesignProcessor().prepare_input(structure, design=FIRST).context.scope.atoms[context].all()
        assert not rigid.prepare_input(structure, design=FIRST).context.scope.atoms[context].any()
        assert rigid.prepare_input(structure, design=FIRST).context.scope.atoms[designed].all()

    def test_msa_profile_unpaired(self, peptide):
        pool = MSA(
            entity_id="P",
            unpaired_sequences=("AG", "AA"),
            unpaired_deletion_counts=np.zeros((2, 2), dtype=np.uint32),
            paired_sequences=("AG",),
            paired_deletion_counts=np.zeros((1, 2), dtype=np.uint32),
        )
        structure = peptide("ALA", "GLY", ligands=("ACT",))
        prepared = ODesignProcessor().prepare_input(structure, design=FIRST, msa={"A": pool})
        assert one_hot(prepared, "A", 1, plane=1) == pytest.approx(0.5)
        assert one_hot(prepared, "G", 1, plane=1) == pytest.approx(0.5)
        assert one_hot(prepared, "-", 0, plane=1) == 1.0
        assert one_hot(prepared, "OTHER", 2, plane=1) == 1.0  # no pool covers it: its own class, not the pool's depth
        assert bool((prepared.features.msa_tokens[0, :, 0] == TOKEN_INDEX["-"]).all())

    def test_artifact_capacity(self, peptide):
        processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
        with pytest.raises(UnsupportedError, match="model_capacity"):
            processor.prepare_input(peptide("ALA", "GLY"))

    def test_crop_needs_generator(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_generator"):
            ODesignProcessor().prepare_input(peptide("ALA"), cropper=lambda *args, **kwargs: None)

    def test_training_target(self, peptide):
        processor = ODesignProcessor()
        structure = peptide("ALA", "GLY")
        prepared, target = processor.prepare(structure, target=structure, design=FIRST)
        tensors = target
        assert tensors.coordinates.shape == (1, len(prepared.context.slots.atom), 3)
        assert tensors.coordinate_mask.all()
        assert bool(tensors.covalent_pair_mask.all())

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
    def test_distinct_truth(self, peptide, dtype):
        structure = peptide("ALA", "GLY")
        source = structure.build()
        source.columns["coordinates"] += [0.123456789123, -0.987654321987, 0.314159265359]
        structure = Structure.from_arrays(structure.system, source)
        arrays = structure.build()
        arrays.columns["coordinates"] += [15, -4, 8]
        target = Structure.from_arrays(structure.system, arrays)
        processor = ODesignProcessor(ODesignProcessorConfig(artifact="protein_rigid"), dtype=dtype)
        prepared, truth = processor.prepare(structure, target=target, design=FIRST)
        tensors = truth
        fixed = prepared.features.fixed_coordinate_mask[0]
        torch.testing.assert_close(
            tensors.coordinates[0, fixed],
            torch.tensor(structure.coordinates[prepared.context.slots.atom[fixed.numpy()]], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            tensors.coordinates[0, ~fixed],
            torch.tensor(target.coordinates[prepared.context.slots.atom[~fixed]], dtype=torch.float64),
        )

    @pytest.mark.parametrize("filter_by_occupancy", [True, False])
    def test_separate_target_reuses_observation_policy(self, peptide, filter_by_occupancy):
        structure = peptide("ALA", "GLY")
        arrays = structure.build()
        arrays.columns["occupancy"][0] = 0
        arrays.columns["occupancy_state"][0] = state(Resolution.KNOWN, Basis.DECLARED)
        target = Structure.from_arrays(structure.system, arrays)
        processor = ODesignProcessor()
        combined, labels = processor.prepare(
            structure, target=target, design=FIRST, filter_by_occupancy=filter_by_occupancy
        )
        prepared = processor.prepare_input(structure, design=FIRST, filter_by_occupancy=filter_by_occupancy)
        separate = processor.prepare_target(target, prepared=prepared)
        torch.testing.assert_close(separate.to_dict(), labels.to_dict())
        slot = int(np.flatnonzero(combined.context.slots.atom == 0)[0])
        assert bool(separate.coordinate_mask[0, slot]) is not filter_by_occupancy


class TestLigandPrepare:
    def test_generated_slots_unbound(self, peptide):
        prepared = ODesignLigandProcessor().prepare_input(peptide("ALA", "GLY"), generated_atoms=3)
        assert prepared.context.slots.atom[-3:].tolist() == [-1, -1, -1]
        assert int(prepared.features.generated_atom_mask.sum()) == 3
        assert prepared.context.scope.rule == "none"
        assert one_hot(prepared, "OTHER", -1) == 1.0

    def test_empty_context_generates(self, polymers):
        prepared = ODesignLigandProcessor().prepare_input(Structure.unobserved(polymers()), generated_atoms=3)
        assert prepared.context.slots.atom.tolist() == [-1, -1, -1]

    def test_sampled_count_needs_generator(self, peptide):
        with pytest.raises(InvalidRequestError, match="request_invalid_generator"):
            ODesignLigandProcessor().prepare_input(peptide("ALA"), generated_atoms=range(2, 4))

    @pytest.mark.parametrize("filter_by_occupancy", [True, False])
    def test_training_truth(self, ligand_truth, filter_by_occupancy):
        context, original = ligand_truth
        system, arrays, mapping = project_arrays(
            original.system.build(), original.build(), np.array([2, 0, 1]), np.ones(len(original.coordinates), bool)
        )
        assert arrays is not None
        generated = mapping.atom[context.system.component_atom_ptr[-1] :]
        rows = np.concatenate((generated[len(generated) // 2 :], generated[: len(generated) // 2]))
        arrays.columns["coordinates"] += [20, -7, 5]
        arrays.columns["occupancy"][rows[0]] = 0
        arrays.columns["occupancy_state"][rows[0]] = state(Resolution.KNOWN, Basis.DECLARED)
        arrays.columns["coordinate_evidence"][rows[-1]] = CoordinateEvidence.NONE
        arrays.columns["coordinates"][rows[-1]] = 0
        target = Structure.from_arrays(MolecularSystem.from_arrays(system), arrays)
        processor = ODesignLigandProcessor()
        prepared, truth = processor.prepare(
            context,
            target=target,
            generated_target_instance_ids=("C", "B"),
            generated_atoms=range(len(rows), len(rows) + 1),
            filter_by_occupancy=filter_by_occupancy,
        )
        separate_input = processor.prepare_input(
            context, generated_atoms=len(rows), filter_by_occupancy=filter_by_occupancy
        )
        separate_truth = processor.prepare_target(
            target, prepared=separate_input, generated_target_instance_ids=("C", "B")
        )
        torch.testing.assert_close(separate_truth.to_dict(), truth.to_dict())
        tensors = truth
        count = len(prepared.context.slots.atom) - len(rows)
        assert prepared.context.structure is context
        assert (prepared.context.slots.atom[-len(rows) :] == -1).all()
        np.testing.assert_allclose(tensors.coordinates[0, :count], prepared.features.observed_positions[0, :count])
        assert bool(tensors.coordinate_mask[0, count]) is not filter_by_occupancy
        assert not tensors.coordinate_mask[0, -1]
        np.testing.assert_allclose(tensors.coordinates[0, count + 1 :], target.coordinates[rows[1:]])
        slot_of = {int(atom): count + slot for slot, atom in enumerate(rows)}
        expected = {
            (slot_of[int(first)], slot_of[int(second)])
            for first, second in target.system.edge_index.T
            if int(first) in slot_of and int(second) in slot_of
        }
        actual = {tuple(edge) for edge in tensors.covalent_bond_index[0].tolist() if min(edge) >= count}
        assert expected == actual and expected
        assert tensors.covalent_pair_mask[0, count:, count:].all()

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
    def test_fixed_condition_preserves_source_precision(self, ligand_truth, dtype):
        context, target = ligand_truth
        source = context.build()
        source.columns["coordinates"] += [0.123456789123, -0.987654321987, 0.314159265359]
        context = Structure.from_arrays(context.system, source)
        changed = target.build()
        changed.columns["coordinates"] += [13, 17, 23]
        target = Structure.from_arrays(target.system, changed)
        processor = ODesignLigandProcessor(dtype=dtype)
        prepared, truth = processor.prepare(context, target=target, generated_target_instance_ids=("B",))
        fixed = prepared.features.fixed_coordinate_mask[0]
        assert fixed.any()
        torch.testing.assert_close(
            truth.coordinates[0, fixed],
            torch.tensor(context.coordinates[prepared.context.slots.atom[fixed.numpy()]], dtype=torch.float64),
            rtol=0,
            atol=0,
        )

    def test_truth_stays_separate(self, ligand_truth):
        context, target = ligand_truth
        processor = ODesignLigandProcessor()
        prepared, truth = processor.prepare(context, target=target, generated_target_instance_ids=("B",))
        arrays = target.build()
        arrays.columns["coordinates"] += [50, -11, 3]
        changed = Structure.from_arrays(target.system, arrays)
        other, other_truth = processor.prepare(context, target=changed, generated_target_instance_ids=("B",))
        for name, value in prepared.features.items():
            torch.testing.assert_close(value, getattr(other.features, name))
        generated = prepared.features.generated_atom_mask[0].numpy()
        np.testing.assert_allclose(
            other_truth.coordinates[0, generated] - truth.coordinates[0, generated],
            np.tile([50, -11, 3], (int(generated.sum()), 1)),
        )

    @pytest.mark.parametrize("generated_atoms", [1, range(1, 3)])
    def test_training_count(self, ligand_truth, generated_atoms):
        context, target = ligand_truth
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            ODesignLigandProcessor().prepare(
                context, target=target, generated_target_instance_ids=("B",), generated_atoms=generated_atoms
            )


class TestLigandPreparationInterface:
    def test_composition_reuses_input_task_size(self, ligand_truth):
        context, truth = ligand_truth
        processor = ODesignLigandProcessor()
        combined, labels = processor.prepare(context, target=truth, generated_target_instance_ids=("B",))
        count = int(combined.features.generated_atom_mask.sum())
        prepared = processor.prepare_input(context, generated_atoms=count)
        target = processor.prepare_target(truth, prepared=prepared, generated_target_instance_ids=("B",))
        torch.testing.assert_close(prepared.features.to_dict(), combined.features.to_dict())
        torch.testing.assert_close(target.to_dict(), labels.to_dict())
        input_only, no_target = processor.prepare(context, generated_atoms=count)
        torch.testing.assert_close(input_only.features.to_dict(), prepared.features.to_dict())
        assert no_target is None
        wrong = processor.prepare_input(context, generated_atoms=count + 1)
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            processor.prepare_target(truth, prepared=wrong, generated_target_instance_ids=("B",))
