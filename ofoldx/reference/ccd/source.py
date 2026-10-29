r"""Resolve CCD artifacts from one Hugging Face dataset snapshot."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from ...errors import RequestRef, fail

ARTIFACT_FILENAME = "components.safetensors"
PROVENANCE_FILENAME = f"{ARTIFACT_FILENAME}.provenance.json"
DEFAULT_REPOSITORY = "oteam/ccd"
DEFAULT_REVISION = "main"


def resolve_snapshot(
    repo_id: str = DEFAULT_REPOSITORY,
    *,
    revision: str = DEFAULT_REVISION,
    token: str | None = None,
    local_files_only: bool = False,
) -> Path:
    r"""Resolve the artifact and provenance sidecar from the same dataset snapshot."""

    if not revision.strip():
        fail("ccd_revision_invalid", RequestRef("revision", revision))
    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=[ARTIFACT_FILENAME, PROVENANCE_FILENAME],
        )
    )
