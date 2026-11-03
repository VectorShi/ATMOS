# This file includes portions adapted from Protenix protenix/data/pipeline/dataloader.py.
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

# Modified for ATMOS: training and inference loaders for trajectory datasets.

from typing import Optional

from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader

from atmos.data.dataset.md_dataset import get_datasets
from atmos.data.dataset.inference_dataset import get_inference_datasets
from protenix.data.pipeline.dataloader import (
    DistributedDataLoader,
    DistributedWeightedSampler,
    IterDataLoader,
    KeySumBalancedSampler,
    WeightedSampler,
)
from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import collate_fn_first

logger = get_logger(__name__)


def get_dataloaders(
    configs: ConfigDict, world_size: int, seed: int, error_dir: Optional[str] = None
):
    """
    Generate data loaders using the MD dataset implementation.
    """
    train_dataset, test_datasets = get_datasets(configs, error_dir)
    if world_size > 1:
        train_sampler = DistributedWeightedSampler(
            train_dataset,
            train_dataset.merged_datapoint_weights,
            num_samples=configs.data.epoch_size,
            replacement=True,
            seed=seed,
        )
        train_dl = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=configs.data.num_dl_workers,
            collate_fn=collate_fn_first,
            sampler=train_sampler,
        )
    else:
        train_sampler = WeightedSampler(
            weights=train_dataset.merged_datapoint_weights,
            num_samples=configs.data.epoch_size,
            replacement=True,
            seed=seed,
        )
        train_dl = IterDataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=configs.data.num_dl_workers,
            collate_fn=collate_fn_first,
            sampler=train_sampler,
        )

    test_dls = {}
    test_dataset_sizes = {}
    for test_name, test_dataset in test_datasets.items():
        test_dataset_sizes[test_name] = len(test_dataset)
        test_sampler = (
            KeySumBalancedSampler(test_dataset, key="num_tokens", seed=configs.seed)
            if world_size > 1
            else None
        )
        test_dls[test_name] = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=configs.data.num_dl_workers,
            sampler=test_sampler,
            collate_fn=collate_fn_first,
        )
    logger.info(
        f"train data size: {len(train_dataset)}, test size: {test_dataset_sizes}"
    )
    return train_dl, test_dls


def get_inference_dataloaders(
    configs: ConfigDict, world_size: int, seed: int, error_dir: Optional[str] = None
):
    """
    Build inference loaders for the datasets listed in ``configs.data.test_sets``.
    """
    test_datasets = get_inference_datasets(configs, error_dir)
    test_dls = {}
    test_dataset_sizes = {}
    for test_name, test_dataset in test_datasets.items():
        test_dataset_sizes[test_name] = len(test_dataset)
        test_sampler = (
            KeySumBalancedSampler(test_dataset, key="num_tokens", seed=configs.seed)
            if world_size > 1
            else None
        )
        test_dls[test_name] = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=configs.data.num_dl_workers,
            sampler=test_sampler,
            collate_fn=collate_fn_first,
        )
    logger.info(
        f"test size: {test_dataset_sizes}"
    )
    return test_dls
