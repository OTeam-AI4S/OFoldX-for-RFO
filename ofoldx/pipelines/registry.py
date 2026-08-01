r"""
Concrete Pipeline registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from ..models.registry import import_builtin

if TYPE_CHECKING:
    from .base import Pipeline

PipelineTask = str


class PipelineRegistration(NamedTuple):
    r"""One concrete executable Pipeline and its biological task."""

    id: str
    pipeline: type[Pipeline]
    task: PipelineTask
    model_architecture: str | None


class PipelineRegistry:
    r"""
    Registry of exact Pipeline IDs and classes.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, PipelineRegistration] = {}
        self._pipelines: dict[type[object], PipelineRegistration] = {}
        self._models: dict[tuple[str, PipelineTask], PipelineRegistration] = {}

    def register(
        self,
        id: str,
        *,
        pipeline: type[Pipeline],
        model_architecture: str | None = None,
    ) -> None:
        r"""
        Register one concrete Pipeline class with its declared biological task.

        Args:
            id: Stable Pipeline identifier.
            pipeline: Concrete Pipeline class.
            model_architecture: Architecture for which this is the default task workflow.
        """

        if id == "":
            raise ValueError("pipeline id must be nonempty")

        self._import_builtin(id)
        if id in self._registrations:
            raise ValueError(f"pipeline id is already registered: {id!r}")
        if pipeline in self._pipelines:
            raise ValueError(f"pipeline class is already registered as {self._pipelines[pipeline].id!r}")

        task = pipeline.task
        if model_architecture is not None and (model_architecture, task) in self._models:
            raise ValueError(f"default Pipeline already registered for {(model_architecture, task)!r}")
        registration = PipelineRegistration(id, pipeline, task, model_architecture)
        self._registrations[id] = registration
        self._pipelines[pipeline] = registration
        if model_architecture is not None:
            self._models[model_architecture, task] = registration

    def _import_builtin(self, id: str) -> None:
        if id not in self._registrations:
            import_builtin(id)

    def lookup_pipeline(self, id: str) -> PipelineRegistration:
        r"""
        Return the Pipeline registration for an exact ID.
        """

        self._import_builtin(id)
        return self._registrations[id]

    def id_for(self, pipeline: type[object]) -> str:
        r"""
        Return the exact ID registered for a Pipeline class.
        """

        return self._pipelines[pipeline].id

    def for_model(self, architecture: str, task: PipelineTask) -> PipelineRegistration:
        r"""Return the explicitly registered default for a model and task."""

        self._import_builtin(architecture)
        try:
            return self._models[architecture, task]
        except KeyError:
            raise ValueError(f"model {architecture!r} does not support {task}") from None


#: Global registry of OFoldX task Pipelines.
PIPELINES = PipelineRegistry()
