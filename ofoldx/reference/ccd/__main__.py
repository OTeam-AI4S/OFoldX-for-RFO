r"""Build, download and explicitly upload CCD datasets: ``python -m ofoldx.reference.ccd``."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any

CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    builder = commands.add_parser("build", help="Build a local dataset directory")
    builder.add_argument("--source", default=CCD_URL, help="Local .cif[.gz] or HTTP(S) URL (default: wwPDB CCD)")
    builder.add_argument("--content-version", required=True, help="Identity of the source snapshot")
    builder.add_argument("--output", required=True, type=Path, help="New output directory")

    downloader = commands.add_parser("download", help="Download a CCD dataset into the Hub cache")
    downloader.add_argument("--repo-id", help="Hub dataset (default: oteam/ccd)")
    downloader.add_argument("--revision", help="Dataset tag or commit (default: main)")
    downloader.add_argument("--local-files-only", action="store_true", help="Use cached files without network access")

    publisher = commands.add_parser("upload", help="Validate and publish a built dataset directory")
    publisher.add_argument("directory", type=Path)
    publisher.add_argument("--repo-id", required=True, help="Destination Hub dataset repository")
    publisher.add_argument("--private", action="store_true", help="Create a private repository")

    arguments = parser.parse_args(argv)
    if arguments.command == "build":
        build_artifact = import_module(".build", __package__).build_artifact
        path = build_artifact(arguments.source, arguments.output, content_version=arguments.content_version)
        print(path)
    elif arguments.command == "download":
        download = import_module(".hub", __package__).download
        options: dict[str, Any] = {"local_files_only": arguments.local_files_only}
        if arguments.repo_id is not None:
            options["repo_id"] = arguments.repo_id
        if arguments.revision is not None:
            options["revision"] = arguments.revision
        print(download(**options))
    else:
        upload = import_module(".hub", __package__).upload
        info = upload(arguments.directory, arguments.repo_id, private=arguments.private)
        print(info.oid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
