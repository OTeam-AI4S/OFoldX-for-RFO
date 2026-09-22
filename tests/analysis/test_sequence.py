"""Behavior of sequence composition annotations."""

import pytest

from ofoldx.analysis.sequence import liability_metrics, sequence_liabilities


class TestComposition:
    @pytest.mark.parametrize("operation", [sequence_liabilities, liability_metrics])
    @pytest.mark.parametrize(
        "sequence,options",
        [("", {}), ("A", {"modality": "unknown"}), ("A", {"peptide_type": "unknown"})],
    )
    def test_invalid_request(self, operation, sequence, options):
        with pytest.raises(ValueError):
            operation(sequence, **options)

    @pytest.mark.parametrize(
        ("modality", "peptide_type"),
        [("antibody", "linear"), ("peptide", "linear"), ("peptide", "cyclic")],
    )
    @pytest.mark.parametrize("sequence", ["C", "CAC", "CACAC"])
    def test_cysteine_count(self, modality, peptide_type, sequence):
        annotations = sequence_liabilities(sequence, modality=modality, peptide_type=peptide_type)
        counts = [item for item in annotations if item.motif == "OddCysteineCount"]
        if sequence.count("C") % 2:
            assert len(counts) == 1
            assert counts[0].position is None
            assert counts[0].details == f"{sequence.count('C')} cysteines"
        else:
            assert not counts
        assert all(item.motif != "UnpairedCys" for item in annotations)

    @pytest.mark.parametrize("sequence,excess", [("KKRDE", 1), ("KKRD", 2), ("H", 0)])
    def test_basic_residue_excess(self, sequence, excess):
        annotations = sequence_liabilities(sequence)
        basic = [item for item in annotations if item.motif == "BasicResidueExcess"]
        if excess > 1:
            assert len(basic) == 1
            assert basic[0].details == f"K+R-D-E={excess}"
        else:
            assert not basic
        assert all(item.motif != "HighNetCharge" for item in annotations)

    def test_report_composition(self):
        report = liability_metrics("CKK")
        assert report["liability_OddCysteineCount_count"] == 1
        assert report["liability_OddCysteineCount_position"] == -1
        assert report["liability_BasicResidueExcess_details"] == "K+R-D-E=2"
