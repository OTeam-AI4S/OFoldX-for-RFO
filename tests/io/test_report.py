from __future__ import annotations

import pickle

import pytest

from ofoldx.errors import ContradictionError
from ofoldx.io import parse_source, read_structure


class TestReadReport:
    def test_summary_counts_bulk(self, library, dipeptide):
        _, report = read_structure(parse_source(dipeptide(), format="mmcif"), library=library)
        assert report.summary()["bulk.standard_junction_endpoint"] == 2

    def test_pickles_with_its_refusal(self, library, disulfide):  # a process pool returns both
        source = parse_source(disulfide, format="mmcif")
        _, report = read_structure(source, library=library, on_conflict="record")
        assert pickle.loads(pickle.dumps(report)).summary() == report.summary()
        with pytest.raises(ContradictionError) as raised:
            read_structure(source, library=library)
        again = pickle.loads(pickle.dumps(raised.value))
        assert again.code == "read_conflicts" and again.detail["report"].summary() == report.summary()
