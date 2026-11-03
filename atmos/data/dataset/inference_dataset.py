# This file includes portions adapted from Protenix protenix/data/pipeline/dataset.py.
# The following upstream copyright notice applies to those portions:
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for ATMOS: trajectory inputs and inference dataset construction.

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn.functional as F
from biotite.structure import AtomArrayStack
from ml_collections.config_dict import ConfigDict

from protenix.data.msa.msa_featurizer import MSAFeaturizer
from protenix.data.pipeline.dataset import (
    BaseSingleDataset,
    get_msa_featurizer,
    get_sample_weights,
    get_template_featurizer,
)
from protenix.data.template.template_featurizer import TemplateFeaturizer
from protenix.utils.logger import get_logger
logger = get_logger(__name__)


class MDInferenceDataset(BaseSingleDataset):
    """
    Dataset class for MD Inference.
    """

    @staticmethod
    def read_pdb_list(pdb_list):
        """Normalize inclusion-list IDs to the lowercase IDs in trajectory indices."""
        entries = BaseSingleDataset.read_pdb_list(pdb_list)
        return None if entries is None else [entry.lower() for entry in entries]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if "mdcath" not in self.name.lower():
            return

        n_before = len(self.indices_list)
        if isinstance(self.indices_list, list):
            self.indices_list = [
                indices
                for indices in self.indices_list
                if str(indices.iloc[0]["pdb_id"]).endswith("_0")
            ]
        else:
            self.indices_list = self.indices_list[
                self.indices_list["pdb_id"].astype(str).str.endswith("_0")
            ].reset_index(drop=True)

        self.check_indices_list(self.indices_list, "mdCATH inference _0 filtering")
        logger.info(
            f"[{self.name}] mdCATH inference _0 filtering: "
            f"{n_before} -> {len(self.indices_list)}"
        )
    
    def process_one(
        self, idx: int, return_atom_token_array: bool = True
    ) -> dict[str, dict]:
        (
            sample_indice,
            bioassembly_dict,
            bioassembly_dict_fpath,
        ) = self._get_bioassembly_data(idx=idx)

        # Trajectory pickles store frames in atom_array_stack or as an atom_array list.
        trajectory = bioassembly_dict.get("atom_array_stack", bioassembly_dict.get("atom_array"))
        if not isinstance(trajectory, (list, tuple, AtomArrayStack)) or len(trajectory) == 0:
            raise ValueError(f"No trajectory frames in {bioassembly_dict_fpath}")
        prefix_batch_atom_array_list = [trajectory]

        if self.shuffle_mols or self.shuffle_sym_ids or self.reassign_continuous_chain_ids or self.use_reference_chains_only:
            raise NotImplementedError

        representative_atom_array = deepcopy(prefix_batch_atom_array_list[0][0])

        bioassembly_dict["atom_array"] = representative_atom_array

        max_entity_mol_id = bioassembly_dict["atom_array"].entity_mol_id.max()

        (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
            reference_token_index,
        ) = self.crop(
            sample_indice=sample_indice,
            bioassembly_dict=bioassembly_dict,
            **self.cropping_configs,
        )

        feat, label, label_full = self.get_feature_and_label(
            idx=idx,
            token_array=cropped_token_array,
            atom_array=cropped_atom_array,
            msa_features=cropped_msa_features,
            template_features=cropped_template_features,
            full_atom_array=bioassembly_dict["atom_array"],
            is_spatial_crop="spatial" in crop_method.lower(),
            max_entity_mol_id=max_entity_mol_id,
        )

        cropped_ref_space_uid = cropped_atom_array.ref_space_uid
        coordinate_list_batch = []
        coordinate_mask_list_batch = []
        for sampled_atom_array_list in prefix_batch_atom_array_list:
            coordinate_list = []
            coordinate_mask_list = []
            for atom_array in sampled_atom_array_list:
                cropped_frame = atom_array[
                    np.isin(atom_array.ref_space_uid, cropped_ref_space_uid)
                ]

                coordinate = torch.tensor(cropped_frame.coord, dtype=torch.float32)
                coordinate_mask = torch.tensor(
                    np.any(cropped_frame.coord != 0, axis=-1),
                    dtype=torch.float32,
                ).contiguous()
                coordinate_list.append(coordinate)
                coordinate_mask_list.append(coordinate_mask)

            coordinate_list_batch.append(coordinate_list)
            coordinate_mask_list_batch.append(coordinate_mask_list)

        sub_sequence_length_max = max(
            len(coordinate_list) for coordinate_list in coordinate_list_batch
        )
        padding_mask_batch = []
        for batch_i in range(len(coordinate_list_batch)):
            coordinate_i = torch.stack(coordinate_list_batch[batch_i], dim=0)
            coordinate_mask_i = torch.stack(coordinate_mask_list_batch[batch_i], dim=0)

            need_pad_length = sub_sequence_length_max - coordinate_i.shape[0]
            coordinate_i = F.pad(
                coordinate_i, (0, 0, 0, 0, need_pad_length, 0), value=0.0
            )
            coordinate_mask_i = F.pad(
                coordinate_mask_i, (0, 0, need_pad_length, 0), value=0.0
            )

            padding_mask_i = torch.ones(
                (sub_sequence_length_max,), dtype=torch.float32
            )
            padding_mask_i[:need_pad_length] = 0.0

            coordinate_list_batch[batch_i] = coordinate_i
            coordinate_mask_list_batch[batch_i] = coordinate_mask_i
            padding_mask_batch.append(padding_mask_i)

        coordinate_batch = torch.stack(coordinate_list_batch, dim=0)
        coordinate_mask_batch = torch.stack(coordinate_mask_list_batch, dim=0)
        padding_mask_batch = torch.stack(padding_mask_batch, dim=0)

        feat["coordinate"] = coordinate_batch[:, :1]
        feat["coordinate_mask"] = coordinate_mask_batch[:, :1]
        feat["prefix_batch_padding_mask"] = None

        label["coordinate"] = coordinate_batch[:, :]
        label["coordinate_mask"] = coordinate_mask_batch[:, :]

        basic_info = {
            "pdb_id": (
                bioassembly_dict["pdb_id"]
                if self.is_distillation is False
                else sample_indice["pdb_id"]
            ),
            "N_asym": torch.tensor([len(torch.unique(feat["asym_id"]))]),
            "N_token": torch.tensor([feat["token_index"].shape[0]]),
            "N_atom": torch.tensor([feat["atom_to_token_idx"].shape[0]]),
            "N_msa": torch.tensor([feat["msa"].shape[0]]),
            "bioassembly_dict_fpath": bioassembly_dict_fpath,
            "N_msa_prot_pair": torch.tensor([feat["prot_pair_num_alignments"]]),
            "N_msa_prot_unpair": torch.tensor([feat["prot_unpair_num_alignments"]]),
            "N_msa_rna_pair": torch.tensor([feat["rna_pair_num_alignments"]]),
            "N_msa_rna_unpair": torch.tensor([feat["rna_unpair_num_alignments"]]),
            "reference_token_index": reference_token_index,
        }

        for mol_type in ("protein", "ligand", "rna", "dna"):
            abbr = {"protein": "prot", "ligand": "lig"}
            abbr_type = abbr.get(mol_type, mol_type)
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            n_atom = int(mol_type_mask.sum(dim=-1).item())
            n_token = len(torch.unique(feat["atom_to_token_idx"][mol_type_mask]))
            basic_info[f"N_{abbr_type}_atom"] = torch.tensor([n_atom])
            basic_info[f"N_{abbr_type}_token"] = torch.tensor([n_token])

        asymn_id_to_chain_id = {
            atom.asym_id_int: atom.chain_id for atom in cropped_atom_array
        }
        chain_id_list = [
            asymn_id_to_chain_id[asymn_id_int]
            for asymn_id_int in sorted(asymn_id_to_chain_id.keys())
        ]
        basic_info["chain_id"] = chain_id_list

        data = {
            "input_feature_dict": feat,
            "label_dict": label,
            "label_full_dict": label_full,
            "basic": basic_info,
        }

        if return_atom_token_array:
            data["cropped_atom_array"] = cropped_atom_array
            data["cropped_token_array"] = cropped_token_array
        return data


def get_inference_datasets(
    configs: ConfigDict, error_dir: Optional[str]
) -> dict[str, MDInferenceDataset]:
    """
    Build the inference datasets listed in ``configs.data.test_sets``.

    Args:
        configs: A ConfigDict containing the dataset configurations.
        error_dir: The directory where error logs will be saved.

    Returns:
        A mapping from dataset names to MD inference datasets.
    """

    def _get_dataset_param(config_dict, dataset_name: str, stage: str):
        return {
            "name": dataset_name,
            **config_dict["base_info"],
            "cropping_configs": config_dict["cropping_configs"],
            "error_dir": error_dir,
            "msa_featurizer": get_msa_featurizer(configs, dataset_name, stage),
            "template_featurizer": get_template_featurizer(
                configs, dataset_name, stage
            ),
            "lig_atom_rename": config_dict.get("lig_atom_rename", False),
            "shuffle_mols": config_dict.get("shuffle_mols", False),
            "shuffle_sym_ids": config_dict.get("shuffle_sym_ids", False),
            "constraint": config_dict.get("constraint", {}),
            "prefix_batch_size": configs.get("prefix_batch_size", 6)
        }

    data_config = configs.data

    test_datasets = {}
    test_sets = data_config.test_sets
    for test_name in test_sets:
        config_dict = data_config[test_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=test_name, stage="test"
        )
        dataset_param["ref_pos_augment"] = data_config.get("test_ref_pos_augment", True)
        test_dataset = MDInferenceDataset(**dataset_param)
        test_datasets[test_name] = test_dataset
    return test_datasets
