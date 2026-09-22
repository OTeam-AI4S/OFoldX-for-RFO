"""Behaviour of ofoldx.io.source: format dispatch and per-model views of one decoded source."""

import pytest

from ofoldx.errors import InsufficientInformationError, InvalidRequestError
from ofoldx.io.source import parse_source, read_source


class TestSource:
    def test_detects_format(self, tmp_path, mmcif_text, pdb_text):
        (tmp_path / "a.cif").write_text(mmcif_text)
        (tmp_path / "b.pdb").write_text(pdb_text)
        assert read_source(tmp_path / "a.cif").format == "mmcif"
        assert read_source(tmp_path / "b.pdb").format == "pdb"

    def test_unknown_format_refused(self, tmp_path, pdb_text):  # "PDB" is not read as CIF
        (tmp_path / "b.pdb").write_text(pdb_text)
        with pytest.raises(InvalidRequestError, match="format_unknown"):
            read_source(tmp_path / "b.pdb", format="PDB")
        with pytest.raises(InvalidRequestError, match="format_unknown"):
            parse_source(pdb_text, format="cif")

    def test_byte_order_mark_skipped(self, tmp_path, mmcif_text, pdb_text):
        for name, text in (("a.cif", mmcif_text), ("b.pdb", pdb_text)):
            (tmp_path / name).write_bytes(b"\xef\xbb\xbf" + text.encode())
        assert read_source(tmp_path / "a.cif").format == "mmcif"
        assert read_source(tmp_path / "b.pdb").format == "pdb"

    def test_several_blocks_refused(self):
        text = "data_a\n_x.v 1\ndata_b\n_x.v 2\n"
        with pytest.raises(InsufficientInformationError, match="multiple_data_blocks"):
            parse_source(text, format="mmcif")
        assert parse_source(text, format="mmcif", block="b").block == "b"

    def test_models_in_file_order(self, mmcif_text):
        swapped = mmcif_text.replace(" P N 1\n", " P N 9\n").replace(" P CA 1\n", " P CA 9\n")
        assert parse_source(swapped.replace(" P ZN 1\n", " P ZN 9\n"), format="mmcif").models == (9, 2)

    def test_views_share_tables(self, mmcif_text):
        source = parse_source(mmcif_text, format="mmcif")
        assert source.tables(1).atoms is source.tables(2).atoms

    def test_unknown_model(self, mmcif_text):
        with pytest.raises(InvalidRequestError, match="model_unknown"):
            parse_source(mmcif_text, format="mmcif").tables(3)
