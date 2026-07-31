import pytest

import ofoldx


class TestPackage:
    def test_version(self):
        assert isinstance(ofoldx.__version__, str)
        assert ofoldx.__version__
        assert ofoldx.version == ofoldx.__version__

    def test_unknown_attribute(self):
        with pytest.raises(AttributeError):
            _ = ofoldx.missing_attribute
