r"""
Public in-memory scientific analysis over generated structures and sequences.
"""

from __future__ import annotations

from .diversity import (
    ConditionedVendiScores,
    DiversityAggregate,
    DiversityMetrics,
    DiversityResult,
    DiversitySample,
    FoldseekMatch,
    FoldseekNoveltyOperation,
    StructureVendiScores,
    analyze_diversity,
    deduplicate_sequences,
    foldseek_novelty,
    greedy_diversity_selection,
    structure_vendi_scores,
)
from .ranking import AnalysisValue, CandidateFilter, CandidateRanking, candidate_filter_results, rank_candidates
from .sequence import (
    Liability,
    LiabilityModality,
    PeptideType,
    liability_metrics,
    sequence_liabilities,
)
from .structure import (
    NoncovalentContacts,
    TargetDeltaSasa,
    assign_secondary_structure,
    count_noncovalent_contacts,
    largest_hydrophobic_patch_area,
    structure_coordinates,
    structure_design_lddt,
    structure_rmsd_metrics,
    target_delta_sasa,
)
