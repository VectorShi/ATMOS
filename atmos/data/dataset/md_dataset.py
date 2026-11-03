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

# Modified for ATMOS: trajectory subsequence sampling and dataset construction.

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn.functional as F
from ml_collections.config_dict import ConfigDict

from protenix.data.msa.msa_featurizer import MSAFeaturizer
from protenix.data.pipeline.dataset import (
    BaseSingleDataset,
    WeightedMultiDataset,
    get_msa_featurizer,
    get_sample_weights,
    get_template_featurizer,
)
from protenix.data.template.template_featurizer import TemplateFeaturizer
from protenix.utils.logger import get_logger
logger = get_logger(__name__)


class RandomSubsequenceFeaturizer:
    """
    Encapsulates the MD random subsequence sampling logic so it can be
    constructed once and injected into the dataset, similar to MSAFeaturizer.
    """

    def __init__(
        self,
        dataset_name: str = "",
        min_len: int = 2,
        max_len: int = 10,
        enable_non_equal_interval: bool = False,
        interval_min: int = 1,
        interval_max: int = 10,
        w_destination: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.min_len = min_len
        self.max_len = max_len
        self.enable_non_equal_interval = enable_non_equal_interval
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.w_destination = w_destination

    @classmethod
    def from_config(
        cls,
        config: ConfigDict,
        dataset_name: str = "",
    ) -> Optional["RandomSubsequenceFeaturizer"]:
        return cls(
            dataset_name=dataset_name,
            min_len=config.get("min_len"),
            max_len=config.get("max_len"),
            enable_non_equal_interval=config.get("enable_non_equal_interval", False),
            interval_min=config.get("interval_min"),
            interval_max=config.get("interval_max"),
            w_destination=config.get("w_destination", False),
        )

    @staticmethod
    def _sample_random_subsequence(
        lst: Sequence[Any],
        min_len: int,
        max_len: int,
        enable_non_equal_interval: bool,
        interval_min: int,
        interval_max: int,
    ) -> tuple[list[Any], bool, list[int]]:
        n = len(lst)

        if not 2 <= min_len <= n:
            raise ValueError("min_len must be between 2 and the list length")
        if min_len > max_len:
            raise ValueError("min_len must not be larger than max_len")
        if interval_min < 1 or interval_min > interval_max:
            raise ValueError("interval_min must satisfy 1 <= interval_min <= interval_max")
        if n - 1 - (min_len - 1) * interval_min < 0:
            raise ValueError("No valid subsequence exists under the current constraints")

        is_equal_interval = (
            random.choice([True, False]) if enable_non_equal_interval else True
        )

        start_idx = random.randint(0, n - 1 - (min_len - 1) * interval_min)
        possible_max_len = (n - 1 - start_idx) // interval_min + 1
        max_len = min(max_len, possible_max_len)

        preset_subseq_len = random.randint(min_len, max_len)
        subsequence = [lst[start_idx]]
        current_idx = start_idx
        intervals: list[int] = []

        if is_equal_interval:
            possible_max_interval = (n - 1 - start_idx) // (preset_subseq_len - 1)
            interval_max = min(possible_max_interval, interval_max)
            fixed_interval = random.randint(interval_min, interval_max)
            for _ in range(preset_subseq_len - 1):
                current_idx += fixed_interval
                subsequence.append(lst[current_idx])
                intervals.append(fixed_interval)
        else:
            remain_length = preset_subseq_len - 1
            for _ in range(preset_subseq_len - 1):
                possible_max_interval = (
                    n - 1 - current_idx - interval_min * (remain_length - 1)
                )
                max_gap = min(interval_max, possible_max_interval)
                interval = random.randint(interval_min, max_gap)
                current_idx += interval
                subsequence.append(lst[current_idx])
                intervals.append(interval)
                remain_length -= 1

        return subsequence, is_equal_interval, intervals

    def _sample_random_subsequence_w_destination(
        self,
        lst: Sequence[Any],
    ) -> tuple[list[Any], bool, list[int]]:
        if self.interval_min != self.interval_max:
            raise ValueError(
                "w_destination only supports a fixed interval because the destination "
                "frame is prepended to the sampled prefix"
            )

        interval = self.interval_min
        if interval != 1:
            raise ValueError(
                "w_destination requires interval_min == interval_max == 1 "
                "for destination-conditioned sampling"
            )

        subsequence, is_equal_interval, intervals = self._sample_random_subsequence(
            lst=lst,
            min_len=self.min_len + 1,
            max_len=self.max_len + 1,
            enable_non_equal_interval=False,
            interval_min=interval,
            interval_max=interval,
        )
        subsequence_length = random.randint(self.min_len, len(subsequence) - 1)

        new_subsequence = [subsequence[-1], *subsequence[:subsequence_length]]
        new_intervals = [
            (len(subsequence) - 1) * interval,
            *intervals[: subsequence_length - 1],
        ]
        return new_subsequence, is_equal_interval, new_intervals

    def make_random_subsequence(
        self,
        lst: Sequence[Any],
    ) -> tuple[list[Any], bool, list[int]]:
        if self.w_destination:
            return self._sample_random_subsequence_w_destination(lst)
        return self._sample_random_subsequence(
            lst=lst,
            min_len=self.min_len,
            max_len=self.max_len,
            enable_non_equal_interval=self.enable_non_equal_interval,
            interval_min=self.interval_min,
            interval_max=self.interval_max,
        )

    def __call__(
        self,
        lst: Sequence[Any],
    ) -> tuple[list[Any], bool, list[int]]:
        return self.make_random_subsequence(lst)


def get_subsequence_featurizer(
    configs: ConfigDict,
    dataset_name: str,
) -> RandomSubsequenceFeaturizer:
    dataset_config = configs["data"][dataset_name]
    if "sample_subsequence_configs" not in dataset_config:
        raise ValueError(
            f"Dataset {dataset_name} must define sample_subsequence_configs"
        )

    return RandomSubsequenceFeaturizer.from_config(
        dataset_config["sample_subsequence_configs"],
        dataset_name=dataset_name,
    )


class MDDataset(BaseSingleDataset):
    """
    Dataset class for MD data with an injectable subsequence sampler.
    """

    def __init__(
        self,
        mmcif_dir: Union[str, Path],
        bioassembly_dict_dir: Optional[Union[str, Path]],
        indices_fpath: Union[str, Path],
        cropping_configs: dict[str, Any],
        msa_featurizer: Optional[MSAFeaturizer] = None,
        template_featurizer: Optional[TemplateFeaturizer] = None,
        subsequence_featurizer: Optional[RandomSubsequenceFeaturizer] = None,
        name: Optional[str] = None,
        prefix_batch_size: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(
            mmcif_dir=mmcif_dir,
            bioassembly_dict_dir=bioassembly_dict_dir,
            indices_fpath=indices_fpath,
            cropping_configs=cropping_configs,
            msa_featurizer=msa_featurizer,
            template_featurizer=template_featurizer,
            name=name,
            **kwargs,
        )

        if subsequence_featurizer is None:
            raise ValueError(
                "subsequence_featurizer must be provided when initializing MDDataset"
            )
        self.subsequence_featurizer = subsequence_featurizer
        self.prefix_batch_size = prefix_batch_size

    def random_subsequence(
        self,
        lst: Sequence[Any],
    ) -> tuple[list[Any], bool, list[int]]:
        return self.subsequence_featurizer(lst)
    
    def process_one(
        self, idx: int, return_atom_token_array: bool = True
    ) -> dict[str, dict]:
        (
            sample_indice,
            bioassembly_dict,
            bioassembly_dict_fpath,
        ) = self._get_bioassembly_data(idx=idx)

        atom_array_stack = bioassembly_dict["atom_array_stack"]

        prefix_batch_atom_array_list = []
        prefix_batch_intervals = []
        for _ in range(self.prefix_batch_size):
            sampled_atom_array_list, _, intervals = self.random_subsequence(
                atom_array_stack
            )
            prefix_batch_atom_array_list.append(
                [deepcopy(atom_array) for atom_array in sampled_atom_array_list]
            )
            prefix_batch_intervals.append(
                torch.as_tensor(intervals, dtype=torch.float32)
            )

        if self.shuffle_mols or self.shuffle_sym_ids or self.reassign_continuous_chain_ids or self.use_reference_chains_only:
            raise NotImplementedError

        sample_traj_i = random.randrange(len(prefix_batch_atom_array_list))
        sample_pos_j = random.randrange(len(prefix_batch_atom_array_list[sample_traj_i]))
        representative_atom_array = deepcopy(
            prefix_batch_atom_array_list[sample_traj_i][sample_pos_j]
        )
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

        if self.subsequence_featurizer.w_destination:
            destinations = []
            destination_masks = []
            destination_intervals = []
            for batch_i in range(len(coordinate_list_batch)):
                destinations.append(coordinate_list_batch[batch_i][0])
                destination_masks.append(coordinate_mask_list_batch[batch_i][0])
                destination_intervals.append(prefix_batch_intervals[batch_i][0])

                coordinate_list_batch[batch_i] = coordinate_list_batch[batch_i][1:]
                coordinate_mask_list_batch[batch_i] = coordinate_mask_list_batch[
                    batch_i
                ][1:]
                prefix_batch_intervals[batch_i] = prefix_batch_intervals[batch_i][1:]

            destinations = torch.stack(destinations, dim=0)
            destination_masks = torch.stack(destination_masks, dim=0)
            destination_intervals = torch.stack(destination_intervals, dim=0)

        sub_sequence_length_max = max(
            len(coordinate_list) for coordinate_list in coordinate_list_batch
        )
        padding_mask_batch = []
        for batch_i in range(len(coordinate_list_batch)):
            coordinate_i = torch.stack(coordinate_list_batch[batch_i], dim=0)
            coordinate_mask_i = torch.stack(coordinate_mask_list_batch[batch_i], dim=0)
            prefix_intervals_i = prefix_batch_intervals[batch_i]

            need_pad_length = sub_sequence_length_max - coordinate_i.shape[0]
            coordinate_i = F.pad(
                coordinate_i, (0, 0, 0, 0, need_pad_length, 0), value=0.0
            )
            coordinate_mask_i = F.pad(
                coordinate_mask_i, (0, 0, need_pad_length, 0), value=0.0
            )
            prefix_intervals_i = F.pad(
                prefix_intervals_i, (need_pad_length, 0), value=0.0
            )

            padding_mask_i = torch.ones(
                (sub_sequence_length_max,), dtype=torch.float32
            )
            padding_mask_i[:need_pad_length] = 0.0

            coordinate_list_batch[batch_i] = coordinate_i
            coordinate_mask_list_batch[batch_i] = coordinate_mask_i
            prefix_batch_intervals[batch_i] = prefix_intervals_i
            padding_mask_batch.append(padding_mask_i)

        coordinate_batch = torch.stack(coordinate_list_batch, dim=0)
        coordinate_mask_batch = torch.stack(coordinate_mask_list_batch, dim=0)
        padding_mask_batch = torch.stack(padding_mask_batch, dim=0)
        intervals = torch.stack(prefix_batch_intervals, dim=0)

        if self.subsequence_featurizer.w_destination:
            coordinate_batch = torch.cat(
                [destinations.unsqueeze(1), coordinate_batch], dim=1
            )
            coordinate_mask_batch = torch.cat(
                [destination_masks.unsqueeze(1), coordinate_mask_batch], dim=1
            )
            padding_mask_batch = torch.cat(
                [padding_mask_batch[:, -1:], padding_mask_batch], dim=1
            )
            intervals = torch.cat(
                [destination_intervals.unsqueeze(1), intervals], dim=1
            )

        feat["coordinate"] = coordinate_batch[:, :-1]
        feat["coordinate_mask"] = coordinate_mask_batch[:, :-1]
        feat["prefix_batch_padding_mask"] = padding_mask_batch[:, :-1]
        feat["intervals"] = intervals

        label["coordinate"] = coordinate_batch[:, -1]
        label["coordinate_mask"] = coordinate_mask_batch[:, -1]

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


def get_datasets(
    configs: ConfigDict, error_dir: Optional[str]
) -> tuple[WeightedMultiDataset, dict[str, BaseSingleDataset]]:
    """
    Get training and testing datasets given configs

    Args:
        configs: A ConfigDict containing the dataset configurations.
        error_dir: The directory where error logs will be saved.

    Returns:
        A tuple containing the training dataset and a dictionary of testing datasets.
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
            "subsequence_featurizer": get_subsequence_featurizer(
                configs, dataset_name
            ),
            "lig_atom_rename": config_dict.get("lig_atom_rename", False),
            "shuffle_mols": config_dict.get("shuffle_mols", False),
            "shuffle_sym_ids": config_dict.get("shuffle_sym_ids", False),
            "constraint": config_dict.get("constraint", {}),
            "prefix_batch_size": configs.get("prefix_batch_size", 6)
        }

    data_config = configs.data
    logger.info(f"Using train sets {data_config.train_sets}")
    assert len(data_config.train_sets) == len(
        data_config.train_sampler.train_sample_weights
    )
    train_datasets = []
    datapoint_weights = []
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=train_name, stage="train"
        )
        dataset_param["ref_pos_augment"] = data_config.get(
            "train_ref_pos_augment", True
        )
        dataset_param["limits"] = data_config.get("limits", -1)
        train_dataset = MDDataset(**dataset_param)
        train_datasets.append(train_dataset)
        datapoint_weights.append(
            get_sample_weights(
                **data_config[train_name]["sampler_configs"],
                indices_df=train_dataset.indices_list,
            )
        )
    train_dataset = WeightedMultiDataset(
        datasets=train_datasets,
        dataset_names=data_config.train_sets,
        datapoint_weights=datapoint_weights,
        dataset_sample_weights=data_config.train_sampler.train_sample_weights,
    )

    test_datasets = {}
    test_sets = data_config.test_sets
    for test_name in test_sets:
        config_dict = data_config[test_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=test_name, stage="test"
        )
        dataset_param["ref_pos_augment"] = data_config.get("test_ref_pos_augment", True)
        test_dataset = MDDataset(**dataset_param)
        test_datasets[test_name] = test_dataset
    return train_dataset, test_datasets
