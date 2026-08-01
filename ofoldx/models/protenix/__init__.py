r"""
Protenix structure-prediction implementations.
"""

from __future__ import annotations

from ...pipelines import PIPELINES
from ..registry import MODELS
from .config import (
    ProtenixConfidenceConfig,
    ProtenixConfig,
    ProtenixConstraintConfig,
    ProtenixEsmConfig,
    ProtenixProcessorConfig,
    ProtenixStructureConditionConfig,
    ProtenixTemplateConfig,
    ProtenixTemplateModelConfig,
    ProtenixTrunkConfig,
    ProtenixVariant,
)
from .model import (
    ProtenixAtomContact,
    ProtenixChiralAtomPotential,
    ProtenixChiralAtomTerm,
    ProtenixConstraintModel,
    ProtenixExperimentalTorsionPotential,
    ProtenixExperimentalTorsionTerm,
    ProtenixExponentialInterpolation,
    ProtenixInput,
    ProtenixInterchainBondPotential,
    ProtenixInterchainBondTerm,
    ProtenixLinearBondPotential,
    ProtenixLinearBondTerm,
    ProtenixModel,
    ProtenixNoisyStructureEmbedder,
    ProtenixOutput,
    ProtenixPairwiseDistancePotential,
    ProtenixPairwiseDistanceTerm,
    ProtenixPlanarImproperPotential,
    ProtenixPlanarImproperTerm,
    ProtenixPocket,
    ProtenixSamplingOutput,
    ProtenixStereoBondPotential,
    ProtenixStereoBondTerm,
    ProtenixStructureConditionModel,
    ProtenixStructureTarget,
    ProtenixTemplateFeatures,
    ProtenixTemplateModel,
    ProtenixTfgFeatures,
    ProtenixTfgGuidance,
    ProtenixTfgPotential,
    ProtenixTokenContact,
    ProtenixVinaStericPotential,
    ProtenixVinaStericTerm,
    protenix_confidence_summary,
    protenix_contact_probability,
)
from .pipeline import (
    ProtenixConstraintPipeline,
    ProtenixPipeline,
    ProtenixPipelineInput,
    ProtenixStructureConditionPipeline,
    ProtenixTemplatePipeline,
)
from .processor import (
    ProtenixConstraintProcessor,
    ProtenixProcessor,
    ProtenixStructureConditionProcessor,
    ProtenixTemplateProcessor,
)

MODELS.register(
    "protenix",
    model=ProtenixModel,
    processor=ProtenixProcessor,
)
MODELS.register(
    "protenix_template",
    model=ProtenixTemplateModel,
    processor=ProtenixTemplateProcessor,
)
MODELS.register(
    "protenix_structure_condition",
    model=ProtenixStructureConditionModel,
    processor=ProtenixStructureConditionProcessor,
)
MODELS.register(
    "protenix_constraint",
    model=ProtenixConstraintModel,
    processor=ProtenixConstraintProcessor,
)
PIPELINES.register(
    "protenix",
    pipeline=ProtenixPipeline,
    model_architecture=ProtenixPipeline.model_architecture,
)
PIPELINES.register(
    "protenix_template",
    pipeline=ProtenixTemplatePipeline,
    model_architecture=ProtenixTemplatePipeline.model_architecture,
)
PIPELINES.register(
    "protenix_structure_condition",
    pipeline=ProtenixStructureConditionPipeline,
    model_architecture=ProtenixStructureConditionPipeline.model_architecture,
)
PIPELINES.register(
    "protenix_constraint",
    pipeline=ProtenixConstraintPipeline,
    model_architecture=ProtenixConstraintPipeline.model_architecture,
)
