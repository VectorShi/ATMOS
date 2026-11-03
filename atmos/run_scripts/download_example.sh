#!/usr/bin/env bash
set -euo pipefail

if (( $# > 0 )); then
  if [[ $# == 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    echo "Usage: bash atmos/run_scripts/download_example.sh"
    echo "Downloads mdCATH inference weights, test trajectories, MSA, and common data."
    echo "Writes a single-protein index to ATMOS_ROOT_DIR/examples/mdcath.csv."
    echo "Requires the installed ATMOS environment, wget, and GNU tar."
    echo "ATMOS_ROOT_DIR defaults to repository atmos_oss/; PROTENIX_ROOT_DIR to af3-dev/."
    exit 0
  fi
  echo "Usage: bash atmos/run_scripts/download_example.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export ATMOS_ROOT_DIR="${ATMOS_ROOT_DIR:-${REPO_DIR}/atmos_oss}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-${REPO_DIR}/af3-dev}"
for tool in python ms-hub wget tar; do
  command -v "${tool}" >/dev/null || { echo "Missing command: ${tool}" >&2; exit 1; }
done
mkdir -p "${ATMOS_ROOT_DIR}" "${PROTENIX_ROOT_DIR}/common"

ms-hub --endpoint https://modelscope.cn download SmartVector/ATMOS \
  --repo-type model --local-dir "${ATMOS_ROOT_DIR}" \
  --include atmos_checkpoints/mdCATH_i20.pt atmos_checkpoints/mdCATH_i20_config.yaml \
            atmos_checkpoints/mdCATH_i1.pt atmos_checkpoints/mdCATH_i1_config.yaml
ms-hub --endpoint https://modelscope.cn download SmartVector/ATMOS-dataset \
  --repo-type dataset --local-dir "${ATMOS_ROOT_DIR}" \
  --include mdCATH_tempo/mdCATH_tempo_test.csv mdCATH_tempo/mdCATH_tempo_test.tar \
            mdCATH_tempo/mdcath_msa/seq_to_pdb_index.json \
            mdCATH_tempo/mdcath_msa/msa_template.tar.gz
tar -xf "${ATMOS_ROOT_DIR}/mdCATH_tempo/mdCATH_tempo_test.tar" -C "${ATMOS_ROOT_DIR}"
tar -xf "${ATMOS_ROOT_DIR}/mdCATH_tempo/mdcath_msa/msa_template.tar.gz" -C "${ATMOS_ROOT_DIR}"

download_file() {
  local url="$1" destination="$2"
  if [[ ! -s "${destination}" ]]; then
    wget -c -O "${destination}.part" "${url}"
    mv "${destination}.part" "${destination}"
  fi
}

for name in clusters-by-entity-40.txt obsolete_release_date.csv \
            obsolete_to_successor.json release_date_cache.json seq_to_pdb_index.json; do
  if [[ ! -s "${PROTENIX_ROOT_DIR}/common/${name}" ]]; then
    download_file https://protenix.tos-cn-beijing.volces.com/protenix.20260101/common.tar.gz \
      "${PROTENIX_ROOT_DIR}/common.tar.gz"
    tar -xzf "${PROTENIX_ROOT_DIR}/common.tar.gz" -C "${PROTENIX_ROOT_DIR}"
    break
  fi
done
for name in components.v20240608.cif components.v20240608.cif.rdkit_mol.pkl; do
  download_file "https://af3-dev.tos-cn-beijing.volces.com/release_data/${name}" \
    "${PROTENIX_ROOT_DIR}/common/${name}"
done

python - <<'PY'
import csv
import os
from pathlib import Path

root = Path(os.environ["ATMOS_ROOT_DIR"])
target = "mdcath_dataset_2hsna01_0"
with (root / "mdCATH_tempo/mdCATH_tempo_test.csv").open(newline="") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = [row for row in reader if row["pdb_id"] == target]
if not rows:
    raise SystemExit(f"Example target {target} is missing from the mdCATH index.")
trajectory = root / "mdCATH_tempo/mdCATH_tempo_test" / f"{target}.pkl.gz"
if not trajectory.is_file():
    raise SystemExit(f"Missing example trajectory: {trajectory}")
output = root / "examples/mdcath.csv"
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"Example ready: {target} ({rows[0]['num_tokens']} tokens)")
print(f"SOURCE_INDICES_CSV={output}")
PY
