r"""
Module tests.
"""

import pytest


def require_oops(*operators: str) -> None:
    """Skip an operator-specific case only when its OOps exports are unavailable."""

    oops = pytest.importorskip("oops")
    missing = tuple(name for name in operators if not callable(getattr(oops, name, None)))
    if missing:
        pytest.skip(f"installed OOps does not provide: {', '.join(missing)}")
