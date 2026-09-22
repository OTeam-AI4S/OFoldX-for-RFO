r"""
Open biomolecular structure models.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._version import __version__ as __version__  # noqa: F401
    from ._version import version
    from .chemistry import PolymerSequence
    from .data import MolecularSystem, Structure
    from .modules.backend import Backend
    from .msa import MSA
    from .results import (
        AutoregressiveProbabilityResult,
        DesignAndPackResult,
        DesignCandidate,
        DesignResult,
        FoldPrediction,
        FoldResult,
        Metric,
        MetricReference,
        PackPrediction,
        PackResult,
        PairMetric,
        Score,
        Scored,
        ScoreResult,
        best,
    )
    from .templates import Template


_EXPORTS = {
    "__version__": ("._version", "__version__"),
    "version": ("._version", "version"),
    "MSA": (".msa", "MSA"),
    "MolecularSystem": (".data", "MolecularSystem"),
    "PolymerSequence": (".chemistry", "PolymerSequence"),
    "Structure": (".data", "Structure"),
    "Template": (".templates", "Template"),
    "Backend": (".modules.backend", "Backend"),
    "AutoregressiveProbabilityResult": (".results", "AutoregressiveProbabilityResult"),
    "DesignAndPackResult": (".results", "DesignAndPackResult"),
    "DesignCandidate": (".results", "DesignCandidate"),
    "DesignResult": (".results", "DesignResult"),
    "FoldPrediction": (".results", "FoldPrediction"),
    "FoldResult": (".results", "FoldResult"),
    "Metric": (".results", "Metric"),
    "MetricReference": (".results", "MetricReference"),
    "PackPrediction": (".results", "PackPrediction"),
    "PackResult": (".results", "PackResult"),
    "PairMetric": (".results", "PairMetric"),
    "Score": (".results", "Score"),
    "Scored": (".results", "Scored"),
    "ScoreResult": (".results", "ScoreResult"),
    "best": (".results", "best"),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
