r"""
Open biomolecular structure models.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._version import __version__ as __version__  # noqa: F401
    from ._version import version
    from .api import pack, pipeline, score
    from .artifacts import save_model
    from .auto import (
        AutoModel,
        AutoPipelineForDesign,
        AutoPipelineForFold,
        AutoPipelineForPack,
        AutoPipelineForScore,
        AutoProcessor,
    )
    from .chemistry import PolymerSequence
    from .data import MolecularSystem, Structure
    from .model import Model
    from .models.conditions import Hotspot, SymbolicBond
    from .modules.backend import Backend
    from .msa import MSA
    from .pipelines import Pipeline
    from .processor import Processor
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
    from .runners import Runner
    from .templates import Template


_EXPORTS = {
    "__version__": ("._version", "__version__"),
    "version": ("._version", "version"),
    "pack": (".api", "pack"),
    "pipeline": (".api", "pipeline"),
    "score": (".api", "score"),
    "save_model": (".artifacts", "save_model"),
    "AutoPipelineForFold": (".auto", "AutoPipelineForFold"),
    "AutoPipelineForDesign": (".auto", "AutoPipelineForDesign"),
    "AutoPipelineForScore": (".auto", "AutoPipelineForScore"),
    "AutoPipelineForPack": (".auto", "AutoPipelineForPack"),
    "AutoModel": (".auto", "AutoModel"),
    "AutoProcessor": (".auto", "AutoProcessor"),
    "MSA": (".msa", "MSA"),
    "Hotspot": (".models.conditions", "Hotspot"),
    "MolecularSystem": (".data", "MolecularSystem"),
    "PolymerSequence": (".chemistry", "PolymerSequence"),
    "Structure": (".data", "Structure"),
    "SymbolicBond": (".models.conditions", "SymbolicBond"),
    "Template": (".templates", "Template"),
    "Pipeline": (".pipelines", "Pipeline"),
    "Model": (".model", "Model"),
    "Backend": (".modules.backend", "Backend"),
    "Processor": (".processor", "Processor"),
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
    "Runner": (".runners", "Runner"),
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
