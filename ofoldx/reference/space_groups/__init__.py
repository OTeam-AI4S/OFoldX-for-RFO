r"""Bundled crystallographic space-group reference data."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, cast

__all__ = ["load_table"]


def load_table() -> dict[str, Any]:
    r"""Load the bundled space-group table and its provenance."""

    text = (
        resources.files("ofoldx.reference.space_groups")
        .joinpath("space_group_operators.json")
        .read_text(encoding="utf-8")
    )
    return cast(dict[str, Any], json.loads(text))
