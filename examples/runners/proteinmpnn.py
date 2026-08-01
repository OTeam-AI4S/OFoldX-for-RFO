from __future__ import annotations

from pathlib import Path

from ofoldx import pipeline, save_model
from ofoldx.io import read_structure, save_structure
from ofoldx.runners import Runner

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "tripeptide.pdb"


def main() -> None:
    # training datasets read native artifacts only, so the bundled PDB file is read and saved once beforehand
    structure, _ = read_structure(STRUCTURE)
    artifact = Path("runs/proteinmpnn/tripeptide.safetensors")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_structure(structure, artifact)
    runner = Runner(
        {
            "steps": 1,
            "model": {"architecture": "proteinmpnn"},
            "data": {"train": str(artifact)},
            "optim": {"type": "adamw", "lr": 1e-4},
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "logging": {"enabled": False},
            "workspace": {"root": "runs/proteinmpnn"},
            "ckpt": {"enabled": False},
        }
    )
    try:
        runner.train()
        repository = Path("runs/proteinmpnn/native")
        save_model(repository, model=runner.unwrap(runner.model), processor=runner.processor)
        scorer = pipeline("score", model=repository)
        result = scorer(structure, seed=7)
        print(result.scores["designed_sequence_nll"].value)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
