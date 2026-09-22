from __future__ import annotations

import pytest

from ofoldx.results.values import (
    DesignAndPackResult,
    DesignCandidate,
    DesignResult,
    FoldPrediction,
    FoldResult,
    PackResult,
    Score,
)


@pytest.fixture
def structure(peptide):
    return peptide("ALA")


class TestFoldResult:
    def test_best_by(self, structure) -> None:
        first = FoldPrediction(structure, scores={"quality": Score(1.0, preferred="higher")})
        second = FoldPrediction(structure, scores={"quality": Score(2.0, preferred="higher")})
        tied = FoldPrediction(structure, scores={"quality": Score(2.0, preferred="higher")})

        assert FoldResult((first, second, tied)).best_by("quality") is second

    def test_direction(self, structure) -> None:
        higher = FoldPrediction(structure, scores={"quality": Score(1.0, preferred="higher")})
        lower = FoldPrediction(structure, scores={"quality": Score(2.0, preferred="lower")})

        with pytest.raises(ValueError):
            FoldResult((higher, lower)).best_by("quality")


class TestDesignResult:
    def test_best_by(self, structure) -> None:
        first = DesignCandidate(structure, scores={"quality": Score(2.0, preferred="lower")})
        second = DesignCandidate(structure, scores={"quality": Score(1.0, preferred="lower")})

        assert DesignResult((first, second)).best_by("quality") is second

    def test_candidates(self, structure) -> None:
        candidate = DesignCandidate(structure)
        result = DesignAndPackResult(DesignResult((candidate,)), (PackResult(()),))

        assert result.candidates == (candidate,)
