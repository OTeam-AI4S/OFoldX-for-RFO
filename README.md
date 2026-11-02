# [OFoldX](https://ofoldx.valhalla.technology/)

> [!TIP]
> Fold molecules. Unfold possibilities.

OFoldX is a PyTorch library for biomolecular structure prediction and design.
Use pretrained models to predict structures, design sequences and backbones,
score candidates, and pack side chains for proteins, nucleic acids, ligands,
and their complexes.

> [!IMPORTANT]
> This fork is maintained for integration with the standalone RFO project. Its
> `rfo-integration` branch contains the OFoldX changes required by RFO. Install
> this branch in the same Python environment as RFO; the PyPI release and the
> upstream OFoldX `main` branch may not contain these integration changes.

## Get Started

Install the RFO-compatible OFoldX branch:

```shell
pip install "git+https://github.com/OTeam-AI4S/OFoldX-for-RFO.git@rfo-integration"
```

Activate the Python environment used by RFO before installing OFoldX. If you
manage the environment with uv, use `uv pip install` with the same Git URL.

Score a protein sequence with ProteinMPNN:

```python
from ofoldx import pipeline
from ofoldx.io import read_structure

structure, _ = read_structure("backbone.cif")
scorer = pipeline("score", model="oteam/proteinmpnn")
result = scorer(structure, seed=7)
nll = result.scores["designed_sequence_nll"]
print(nll.value, nll.unit)
```

The result reports sequence negative log-likelihood in `nat/residue`. See the
[model](ofoldx/models) and [pipeline](ofoldx/pipelines) guides for structure
prediction, molecular design, scoring, and side-chain packing. See the
[runner guide](ofoldx/runners) for training.

For general OFoldX use outside the RFO integration, install the upstream
package from PyPI:

```shell
pip install ofoldx
```

## Community

- [GitHub Issues](https://github.com/OTeam-AI4S/OFoldX/issues): bugs, usage questions, and feature requests.
- [Hugging Face](https://huggingface.co/oteam): model checkpoints and datasets.

## Citation

> [!NOTE]
> The artifacts distributed in this repository are part of the OFoldX project.
> If OFoldX supports your research, please cite the OFoldX project as follows:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## License

We believe openness is the Foundation of Research.

OFoldX is licensed under the [GNU Affero General Public License](license.md).

For additional terms and clarifications, please refer to our [License FAQ](license-faq.md).

Please join us in building an open research community.

`SPDX-License-Identifier: AGPL-3.0-or-later`
