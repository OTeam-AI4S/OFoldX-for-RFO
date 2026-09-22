"""Behaviour of ofoldx.io.tables: the selected model's checks raise, the other models' are recorded."""

import pytest

from ofoldx.errors import ContradictionError
from ofoldx.io.source import parse_source


class TestModelView:
    @pytest.fixture
    def source(self, mmcif_text):
        # model 2's zinc names an asym _struct_asym does not declare (1ts6)
        return parse_source(mmcif_text.replace("HETATM 6 ZN ZN . ZN B", "HETATM 6 ZN ZN . ZN C"), format="mmcif")

    def test_other_model_recorded(self, source):
        records = source.tables(1).records
        detail = {"model": 2, "check": "asym_undeclared", "asym": "C"}
        assert (records.counts(), records[0].detail) == ({"entry_level_violation": 1}, detail)

    def test_selected_model_refused(self, source):
        with pytest.raises(ContradictionError, match="asym_undeclared"):
            source.tables(2)
