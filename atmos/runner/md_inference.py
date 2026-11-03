# This file includes portions adapted from Protenix runner/train.py and runner/inference.py.
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

# Modified for ATMOS: distributed trajectory inference and staged model loading.

import copy
import datetime
import hashlib
import logging
import os
import time
from argparse import Namespace
from contextlib import nullcontext
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from atmos.data.dataset.dataloader import get_inference_dataloaders
from atmos.model.atmos import AtmosProtenix
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config.config import load_config, parse_configs, parse_sys_args, save_config
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.file_io import dump_gzip_pickle
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device

os.environ["WANDB_CONSOLE"] = "off"

torch.serialization.add_safe_globals([Namespace])


def _config_mod_to_arg_str(config_mod: str) -> str:
    if not config_mod:
        return ""
    parts = [part.strip() for part in config_mod.split(",") if part.strip()]
    assert len(parts) % 2 == 0, f"Invalid load_config_mod: {config_mod}"
    for index in range(0, len(parts), 2):
        parts[index] = "--" + parts[index]
    return " ".join(parts)


def load_stage_config(config_path: str, config_mod: str = ""):
    return parse_configs(
        load_config(config_path),
        arg_str=_config_mod_to_arg_str(config_mod),
    )


class AtmosInferenceRunner:

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.try_load_checkpoint()
        self.init_data()

    def init_basics(self) -> None:
        run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.prediction_dir = f"{self.run_dir}/predictions"
        self.error_dir = f"{self.run_dir}/errors"

        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.prediction_dir, exist_ok=True)
            os.makedirs(self.error_dir, exist_ok=True)
            save_config(self.configs, os.path.join(self.run_dir, "config.yaml"))
        self.print(
            f"Using run name: {self.run_name}, run dir: {self.run_dir}, "
            f"prediction_dir: {self.prediction_dir}, error_dir: {self.error_dir}"
        )

    def init_env(self) -> None:
        """
        Initialize the PyTorch and CUDA environment for inference.
        Set up distributed inference if world_size > 1 and set random seeds.
        """
        logging.info(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if DIST_WRAPPER.world_size > 1:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
            )

        if not self.configs.deterministic_seed:
            # use rank-specific seed
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},init_seed)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed

        seed_everything(
            seed=rank_seed,
            deterministic=self.configs.deterministic,
        )  # Different DDP processes get different seeds

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            print(f"env: {env}")
            assert env is not None, (
                "If use deepspeed (ds4sci), set CUTLASS_PATH env as per "
                "https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
        logging.info("Finished environment initialization.")

    def init_model(self) -> None:
        assert self.configs.load_checkpoint_path
        assert self.configs.load_config_path
        load_checkpoint_paths = self._split_stage_arg(self.configs.load_checkpoint_path)
        load_config_paths = self._split_stage_arg(self.configs.load_config_path)
        load_config_mods = self._split_stage_arg(
            self.configs.load_config_mod,
            expected_len=len(load_config_paths),
            fill_value="",
        )
        assert len(load_checkpoint_paths) == len(load_config_paths)

        self.models = []
        self.model_configs = []
        for model_idx, config_path in enumerate(load_config_paths):
            stage_config = load_stage_config(config_path, load_config_mods[model_idx])
            self.print(
                f"Model {model_idx} inference: N_cycle={stage_config.model.N_cycle}, "
                f"N_step={stage_config.sample_diffusion.N_step}, "
                f"model_max_track_length={stage_config.model_max_track_length}"
            )
            self.model_configs.append(stage_config)
            model = AtmosProtenix(stage_config).to(self.device)
            model.eval()
            self.models.append(model)

        for model_idx, model in enumerate(self.models):
            self.print(f"Model {model_idx} Parameters: {self.count_parameters(model):.2f}M")
        torch.cuda.empty_cache()

    def init_data(self) -> None:
        self.test_dls = get_inference_dataloaders(
            self.configs,
            DIST_WRAPPER.world_size,
            seed=self.configs.seed,
            error_dir=self.error_dir,
        )

    def try_load_checkpoint(self) -> None:
        load_checkpoint_paths = self._split_stage_arg(self.configs.load_checkpoint_path)
        for model_idx, checkpoint_path in enumerate(load_checkpoint_paths):
            self._load_checkpoint(checkpoint_path, model_idx)

    def _load_checkpoint(self, checkpoint_path: str, model_idx: int) -> None:
        """Load a state dict or a checkpoint containing it under ``model``."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Given checkpoint path does not exist: [{checkpoint_path}]")
        load_strict = bool(self.configs.get("load_strict", False))
        self.print(f"Loading model {model_idx} from {checkpoint_path}, strict: {load_strict}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint.get("model", checkpoint)
        sample_key = next(iter(state_dict.keys()))
        if sample_key.startswith("module."):
            state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
        load_result = self.models[model_idx].load_state_dict(
            state_dict=state_dict,
            strict=load_strict,
        )
        self._print_missing_checkpoint_keys(model_idx, load_result.missing_keys)
        self.print(f"Finished loading model {model_idx}")

    def _print_missing_checkpoint_keys(
        self,
        model_idx: int,
        missing_keys: Sequence[str],
    ) -> None:
        if not missing_keys:
            self.print(f"Model {model_idx} keys without checkpoint weights: none")
            return
        self.print(
            f"Model {model_idx} keys without checkpoint weights ({len(missing_keys)}):\n  "
            + "\n  ".join(missing_keys)
        )

    @staticmethod
    def _split_stage_arg(value: str, expected_len: int | None = None, fill_value: str = "") -> list[str]:
        if value:
            parts = value.split(";")
        elif expected_len is not None:
            parts = [fill_value for _ in range(expected_len)]
        else:
            parts = []
        if expected_len is not None:
            assert len(parts) == expected_len, f"Expected {expected_len} items, got {len(parts)}: {value}"
        return parts

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> float:
        return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000.0

    def print(self, msg: str) -> None:
        """
        Print message to log only on the master rank (rank 0).

        Args:
            msg (str): The message to log.
        """
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)

    def get_model_attr(self, model_idx: int, attr_name: str) -> Any:
        model = self.models[model_idx]
        if isinstance(model, DDP):
            model = model.module
        return getattr(model, attr_name)

    def parse_infer_trajectory_schedule(self) -> tuple[list[int], list[int]]:
        schedule = self.configs.infer_trajectory_schedule.split(";")
        infer_length_list = []
        infer_interval_list = []
        for scale_idx, scale in enumerate(schedule):
            parts = scale.split(":")
            assert len(parts) == 2, f"Invalid infer_trajectory_schedule scale: {scale}"
            infer_length, infer_interval = int(parts[0]), int(parts[1])
            if scale_idx > 0:
                assert infer_length * infer_interval == infer_interval_list[-1]
            infer_length_list.append(infer_length)
            infer_interval_list.append(infer_interval)
        return infer_length_list, infer_interval_list

    def prepare_inference_batch_inputs(
        self,
        batch: dict[str, Any],
        intervals: torch.Tensor,
        coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
    ) -> None:
        input_feature_dict = batch["input_feature_dict"]
        input_feature_dict["intervals"] = intervals
        input_feature_dict["coordinate"] = coordinate
        input_feature_dict["coordinate_mask"] = coordinate_mask
        input_feature_dict["prefix_batch_padding_mask"] = None

    def model_forward(
        self,
        batch: dict[str, Any],
        model_idx: int = 0,
        infer_trajectory_max_length: int = 10,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model = self.models[model_idx]
        with torch.no_grad():
            batch["pred_dict"], batch["label_dict"], log_dict = model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                label_full_dict=batch["label_full_dict"],
                mode="inference",
                current_step=None,
                infer_trajectory_max_length=infer_trajectory_max_length,
            )
        return batch, log_dict

    def _constant_intervals(self, infer_length: int, infer_interval: int) -> torch.Tensor:
        return torch.as_tensor(
            [infer_interval for _ in range(infer_length - 1)],
            dtype=torch.float32,
            device=self.device,
        )

    def _append_pred_frames(
        self,
        pred_coordinate: torch.Tensor,
        structures: list[torch.Tensor],
    ) -> None:
        batched_pred_coordinate = pred_coordinate.reshape(
            *((-1, 1) + pred_coordinate.shape[-2:])
        )
        for frame_idx in range(batched_pred_coordinate.shape[0]):
            structures.append(batched_pred_coordinate[frame_idx : frame_idx + 1])

    def get_ground_truth_coarse_structures(
        self,
        batch: dict[str, Any],
        infer_interval: int,
        num_structures: int,
    ) -> list[torch.Tensor]:
        coarse_structures = []
        ground_truth_coordinate = batch["label_dict"]["coordinate"]
        for frame_idx in range(num_structures):
            if (
                frame_idx == num_structures - 1
                and ground_truth_coordinate.shape[1] <= frame_idx * infer_interval
            ):
                coarse_structures.append(
                    ground_truth_coordinate[:, frame_idx * infer_interval - 1 : frame_idx * infer_interval]
                )
            else:
                coarse_structures.append(
                    ground_truth_coordinate[:, frame_idx * infer_interval : frame_idx * infer_interval + 1]
                )
        return coarse_structures

    def inference_long_traj_single_model_single_interval(self, batch: dict[str, Any]) -> dict[str, Any]:
        infer_length_list, infer_interval_list = self.parse_infer_trajectory_schedule()
        assert len(infer_length_list) == 1 and len(infer_interval_list) == 1
        assert len(self.models) == 1
        assert not self.get_model_attr(0, "destination_enable")

        infer_length, infer_interval = infer_length_list[0], infer_interval_list[0]
        intervals = self._constant_intervals(infer_length, infer_interval).view(1, -1)
        self.prepare_inference_batch_inputs(
            batch=batch,
            intervals=intervals,
            coordinate=batch["input_feature_dict"]["coordinate"],
            coordinate_mask=batch["input_feature_dict"]["coordinate_mask"],
        )
        batch, _ = self.model_forward(
            batch,
            model_idx=0,
            infer_trajectory_max_length=infer_length,
        )
        return batch

    def inference_long_traj_multiple_model_multiple_interval(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._inference_multiple_model_multiple_interval(
            batch=batch,
            use_ground_truth_coarse=False,
            use_stitching=False,
        )

    def inference_fine_grained_upsampling(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._inference_multiple_model_multiple_interval(
            batch=batch,
            use_ground_truth_coarse=True,
            use_stitching=False,
        )

    def inference_long_traj_multiple_model_multiple_interval_w_stitching(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._inference_multiple_model_multiple_interval(
            batch=batch,
            use_ground_truth_coarse=False,
            use_stitching=True,
        )

    def inference_fine_grained_upsampling_w_stitching(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._inference_multiple_model_multiple_interval(
            batch=batch,
            use_ground_truth_coarse=True,
            use_stitching=True,
        )

    def _run_first_scale(
        self,
        batch: dict[str, Any],
        infer_length: int,
        infer_interval: int,
        use_ground_truth_coarse: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        assert not self.get_model_attr(0, "destination_enable")
        coarse_infer_length = infer_length + 1
        intervals = self._constant_intervals(coarse_infer_length, infer_interval).view(1, -1)
        self.prepare_inference_batch_inputs(
            batch=batch,
            intervals=intervals,
            coordinate=batch["input_feature_dict"]["coordinate"],
            coordinate_mask=batch["input_feature_dict"]["coordinate_mask"],
        )
        batch, _ = self.model_forward(
            batch,
            model_idx=0,
            infer_trajectory_max_length=coarse_infer_length,
        )

        new_start_structures: list[torch.Tensor] = []
        if use_ground_truth_coarse:
            pred_coordinate = batch["pred_dict"]["coordinate"]
            num_structures = pred_coordinate.reshape(*((-1, 1) + pred_coordinate.shape[-2:])).shape[0]
            new_start_structures = self.get_ground_truth_coarse_structures(
                batch=batch,
                infer_interval=infer_interval,
                num_structures=num_structures,
            )
        else:
            self._append_pred_frames(batch["pred_dict"]["coordinate"], new_start_structures)
        return new_start_structures[:-1], new_start_structures[1:]

    def _inference_multiple_model_multiple_interval(
        self,
        batch: dict[str, Any],
        use_ground_truth_coarse: bool,
        use_stitching: bool,
    ) -> dict[str, Any]:
        infer_length_list, infer_interval_list = self.parse_infer_trajectory_schedule()
        assert len(self.models) == len(infer_length_list)
        assert len(self.models) > 1

        coordinate_mask = batch["input_feature_dict"]["coordinate_mask"]
        start_structures, end_structures = self._run_first_scale(
            batch=batch,
            infer_length=infer_length_list[0],
            infer_interval=infer_interval_list[0],
            use_ground_truth_coarse=use_ground_truth_coarse,
        )

        for scale_idx in range(1, len(infer_length_list)):
            if use_stitching:
                start_structures, end_structures = self._run_stitched_scale(
                    batch=batch,
                    scale_idx=scale_idx,
                    infer_length_list=infer_length_list,
                    infer_interval_list=infer_interval_list,
                    start_structures=start_structures,
                    end_structures=end_structures,
                    coordinate_mask=coordinate_mask,
                )
            else:
                start_structures, end_structures = self._run_parallel_scale(
                    batch=batch,
                    scale_idx=scale_idx,
                    infer_length_list=infer_length_list,
                    infer_interval_list=infer_interval_list,
                    start_structures=start_structures,
                    end_structures=end_structures,
                    coordinate_mask=coordinate_mask,
                )

        batch["pred_dict"]["coordinate"] = torch.cat(start_structures, dim=-3)
        batch["input_feature_dict"]["coordinate_mask"] = coordinate_mask
        return batch

    def _run_parallel_scale(
        self,
        batch: dict[str, Any],
        scale_idx: int,
        infer_length_list: Sequence[int],
        infer_interval_list: Sequence[int],
        start_structures: list[torch.Tensor],
        end_structures: list[torch.Tensor],
        coordinate_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        infer_length = infer_length_list[scale_idx]
        infer_interval = infer_interval_list[scale_idx]
        destination_enable = self.get_model_attr(scale_idx, "destination_enable")
        intervals = [infer_interval for _ in range(infer_length - 1)]
        if destination_enable:
            intervals = [infer_interval_list[scale_idx - 1]] + intervals
        intervals_tensor = torch.as_tensor(intervals, dtype=torch.float32, device=self.device)

        new_start_structures: list[torch.Tensor] = []
        largest_batch_latent_num = self.configs.largest_batch_latent_num
        num_chunks = len(start_structures) // largest_batch_latent_num + (
            len(start_structures) % largest_batch_latent_num != 0
        )
        for chunk_idx in range(num_chunks):
            range_left = chunk_idx * largest_batch_latent_num
            range_right = (chunk_idx + 1) * largest_batch_latent_num
            input_coordinates = start_structures[range_left:range_right]
            batch_latent_num = len(input_coordinates)
            coordinate = torch.cat(input_coordinates, dim=-4)
            local_coordinate_mask = coordinate_mask.expand(batch_latent_num, -1, -1)

            if destination_enable:
                destination_coordinate = torch.cat(end_structures[range_left:range_right], dim=-4)
                coordinate = torch.cat([destination_coordinate, coordinate], dim=-3)
                local_coordinate_mask = coordinate_mask.expand(batch_latent_num, 2, -1)

            self.prepare_inference_batch_inputs(
                batch=batch,
                intervals=intervals_tensor.unsqueeze(0).expand(batch_latent_num, -1),
                coordinate=coordinate,
                coordinate_mask=local_coordinate_mask,
            )
            batch, _ = self.model_forward(
                batch,
                model_idx=scale_idx,
                infer_trajectory_max_length=infer_length,
            )
            self._append_pred_frames(batch["pred_dict"]["coordinate"], new_start_structures)

        return new_start_structures, new_start_structures[1:] + end_structures[-1:]

    def _run_stitched_scale(
        self,
        batch: dict[str, Any],
        scale_idx: int,
        infer_length_list: Sequence[int],
        infer_interval_list: Sequence[int],
        start_structures: list[torch.Tensor],
        end_structures: list[torch.Tensor],
        coordinate_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        infer_length = infer_length_list[scale_idx]
        infer_interval = infer_interval_list[scale_idx]
        destination_enable = self.get_model_attr(scale_idx, "destination_enable")
        model_max_track_length = self.get_model_attr(scale_idx, "model_max_track_length")
        stitching_prefix_num = self.configs.infer_trajectory_stitching_prefix_num
        assert stitching_prefix_num > 0
        assert stitching_prefix_num < infer_length
        assert stitching_prefix_num + 1 < model_max_track_length

        new_start_structures: list[torch.Tensor] = []
        stitched_prefix: list[torch.Tensor] = []
        for segment_idx, start_coordinate in enumerate(start_structures):
            prefix_coordinates = stitched_prefix + [start_coordinate]
            prefix_length = len(prefix_coordinates)
            local_infer_length = infer_length + prefix_length - 1
            intervals = [infer_interval for _ in range(local_infer_length - 1)]
            if destination_enable:
                destination_interval = infer_interval_list[scale_idx - 1] + (prefix_length - 1) * infer_interval
                intervals = [destination_interval] + intervals

            coordinate = torch.cat(prefix_coordinates, dim=-3)
            local_coordinate_mask = coordinate_mask.expand(1, prefix_length, -1)
            if destination_enable:
                coordinate = torch.cat([end_structures[segment_idx], coordinate], dim=-3)
                local_coordinate_mask = coordinate_mask.expand(1, prefix_length + 1, -1)

            self.prepare_inference_batch_inputs(
                batch=batch,
                intervals=torch.as_tensor(intervals, dtype=torch.float32, device=self.device).view(1, -1),
                coordinate=coordinate,
                coordinate_mask=local_coordinate_mask,
            )
            batch, _ = self.model_forward(
                batch,
                model_idx=scale_idx,
                infer_trajectory_max_length=local_infer_length,
            )

            segment_coordinate = batch["pred_dict"]["coordinate"][:, prefix_length - 1 :]
            self._append_pred_frames(segment_coordinate, new_start_structures)

            stitched_prefix = []
            stitched_tail = segment_coordinate[:, -stitching_prefix_num:]
            for frame_idx in range(stitched_tail.shape[1]):
                stitched_prefix.append(stitched_tail[:, frame_idx : frame_idx + 1])

        return new_start_structures, new_start_structures[1:] + end_structures[-1:]

    def _select_inference_forward(self, mode: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        if mode == "long_traj:single_model_single_interval":
            assert len(self.models) == 1
            return self.inference_long_traj_single_model_single_interval
        if mode == "long_traj:multiple_model_multiple_interval":
            assert len(self.models) > 1
            return self.inference_long_traj_multiple_model_multiple_interval
        if mode == "long_traj:inference_fine_grained_upsampling":
            assert len(self.models) > 1
            return self.inference_fine_grained_upsampling
        if mode == "long_traj:multiple_model_multiple_interval_w_stitching":
            assert len(self.models) > 1
            return self.inference_long_traj_multiple_model_multiple_interval_w_stitching
        if mode == "long_traj:inference_fine_grained_upsampling_w_stitching":
            assert len(self.models) > 1
            return self.inference_fine_grained_upsampling_w_stitching
        raise NotImplementedError(f"Unsupported infer_trajectory_mode: {mode}")

    @torch.no_grad()
    def run(self) -> None:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        for model in self.models:
            model.eval()

        inference_forward = self._select_inference_forward(self.configs.infer_trajectory_mode)
        for test_name, test_dl in self.test_dls.items():
            self.print(f"Running inference on {test_name}")
            for _, batch in enumerate(tqdm(test_dl)):
                batch = to_device(batch, self.device)
                pid = batch["basic"]["pdb_id"]
                with enable_amp:
                    batch = inference_forward(batch)

                batch = to_device(batch, "cpu")
                output_path = os.path.join(
                    self.prediction_dir,
                    f"rank_{DIST_WRAPPER.rank}_{pid}.pkl.gz",
                )
                dump_gzip_pickle(batch, output_path)
                del batch
                torch.cuda.empty_cache()


def main() -> None:
    """
    Parse configurations and start the inference process.
    """
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "cuequivariance"
    )
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance"
    )
    arg_str = parse_sys_args()
    configs = {**configs_base, **{"data": data_configs}}
    # 1. First pass to get model_name
    configs = parse_configs(
        configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = configs.model_name

    # 2. Get model specifics and merge into base defaults
    base_configs = {**configs_base, **{"data": data_configs}}
    model_specfics_configs = model_configs[model_name]

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specfics_configs)

    # 3. Second pass to apply sys_args with higher priority
    configs = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    logging.info(
        f"Using params for model {model_name}: "
        f"cycle={configs.model.N_cycle}, step={configs.sample_diffusion.N_step}"
    )
    print(f"Run Name: {configs.run_name}")
    runner = AtmosInferenceRunner(configs)
    runner.run()


if __name__ == "__main__":
    main()
