# Third-party notices

The repository includes the components described below. Their copyright,
attribution, and license notices remain applicable to the corresponding code.
These notices supplement the [Apache License 2.0](LICENSE) and do not replace
the licenses of individual components or separately installed dependencies.

## Protenix

- Project: [ByteDance Protenix](https://github.com/bytedance/Protenix).
- Copyright 2024 ByteDance and/or its affiliates.
- License: [Apache License 2.0](LICENSE).

The Protenix implementation is included under `protenix/`, with configuration
and entry points under `configs/`, `runner/`, and `scripts/`. ATMOS adapts
Protenix model, loss, dataset, data-loader, and runner code under `atmos/`;
those files identify their source and the ATMOS modifications in their headers.
Modified configuration and utility files also carry modification notices.

The ByteDance copyright notice in ATMOS files identifies the portions derived
from Protenix. It does not attribute ATMOS-specific additions or modifications
to ByteDance.

Protenix includes implementations based on
[OpenFold](https://github.com/aqlaboratory/openfold),
[OneFlow](https://github.com/Oneflow-Inc/oneflow), and
[FastFold](https://github.com/hpcaitech/FastFold), released under Apache 2.0.
Their attribution is retained in the source files and in [LICENSE](LICENSE).

## MISATO trajectory conversion

- Project: [MISATO](https://github.com/t7morgen/misato-dataset).
- Source: [`src/data/processing/h5_to_traj.py`](https://github.com/t7morgen/misato-dataset/blob/7b06d532e2ed0719411fcc1b3ac39743db4ca10d/src/data/processing/h5_to_traj.py).
- Included adaptation: [`atmos/data/preprocessing/misato/step_0_misato_h5_to_traj.py`](atmos/data/preprocessing/misato/step_0_misato_h5_to_traj.py).
- Copyright (C) 2023 Till Siebenmorgen, Sabrina Benassou, Filipe Menezes,
  and Erinç Merdivan.
- License: GNU Lesser General Public License, version 2.1 or, at your option,
  any later version (`LGPL-2.1-or-later`). The complete
  [LGPL 2.1 text](LICENSES/LGPL-2.1-or-later.txt) is included.

The conversion script retains the MISATO copyright and license notice and
identifies the ATMOS modifications. Its source is distributed in this
repository under LGPL-2.1-or-later, including those modifications. The
repository's Apache 2.0 license does not replace this script's license.

## Data and model assets

The licenses of datasets and model weights are separate from source-code
licenses. See the [asset licensing and attribution notes](docs/assets.md#licenses-and-attribution)
for the dataset sources and the scope of the code license.
