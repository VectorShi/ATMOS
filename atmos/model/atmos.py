# This file includes portions adapted from Protenix protenix/model/protenix.py.
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

# Modified for ATMOS: trajectory conditioning, fine-tuning, and autoregressive rollout.

import random
import time
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch

from atmos.model.modules import DynaSSMAA
from atmos.model.utils import append_batch_dim
from protenix.model.generator import sample_diffusion_training
from protenix.model.protenix import Protenix, update_input_feature_dict
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator


class AtmosProtenix(Protenix):
    
    def __init__(self, configs: Any) -> None:
        super().__init__(configs)
        dyna_configs = self._get_dyna_config(configs)
        self.dyna_ssm = DynaSSMAA(**dyna_configs)
        self.freeze_pairformer = configs.freeze_pairformer
        self.model_max_track_length = configs.model_max_track_length
        self.overlap_structure_num = configs.overlap_structure_num
        self.destination_enable = dyna_configs["destination_enable"]
        self._freeze_unused_modules()
        if self.freeze_pairformer:
            self._freeze_static_conditioning_modules()

    @staticmethod
    def _freeze_module(module: torch.nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad_(False)

    def _freeze_unused_modules(self) -> None:
        self._freeze_module(self.confidence_head)

    def _freeze_static_conditioning_modules(self) -> None:
        for module in (
            self.input_embedder,
            self.relative_position_encoding,
            self.template_embedder,
            self.msa_module,
            self.constraint_embedder,
            self.pairformer_stack,
            self.linear_no_bias_sinit,
            self.linear_no_bias_zinit1,
            self.linear_no_bias_zinit2,
            self.linear_no_bias_token_bond,
            self.layernorm_z_cycle,
            self.linear_no_bias_z_cycle,
            self.layernorm_s,
            self.linear_no_bias_s,
        ):
            self._freeze_module(module)

    def _get_dyna_config(self, configs: Any) -> dict[str, Any]:
        dyna_ssm = configs.model.dyna_ssm
        return {
            "c_token": configs.c_s,
            "c_tokenpair": configs.c_z,
            "pairformer_n_block": dyna_ssm.dyna_pairformer_n_block,
            "blocks_per_ckpt": configs.blocks_per_ckpt,
            "triangle_multiplicative": configs.triangle_multiplicative,
            "triangle_attention": configs.triangle_attention,
            "max_temperature": dyna_ssm.max_tempurtare,
            "min_temperature": dyna_ssm.min_tempurtare,
            "prefix_distogram_min_bin": dyna_ssm.prefix_distogram_min_bin,
            "prefix_distogram_max_bin": dyna_ssm.prefix_distogram_max_bin,
            "prefix_distogram_no_bins": dyna_ssm.prefix_distogram_no_bins,
            "destination_enable": configs.get("destination_enable", False),
            "prefix_fine_distogram_min_bin": dyna_ssm.prefix_fine_distogram_min_bin,
            "prefix_fine_distogram_max_bin": dyna_ssm.prefix_fine_distogram_max_bin,
            "prefix_fine_distogram_no_bins": dyna_ssm.prefix_fine_distogram_no_bins,
        }

    def get_static_conditioning(
        self,
        input_feature_dict: dict[str, Any],
        n_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )

        s_init = self.linear_no_bias_sinit(s_inputs)
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )
        z_init = z_init + self.relative_position_encoding(input_feature_dict["relp"])
        z_init = z_init + self.linear_no_bias_token_bond(
            input_feature_dict["token_bonds"].unsqueeze(dim=-1)
        )

        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        for cycle_no in range(n_cycle):
            with torch.set_grad_enabled(
                self.training
                and cycle_no == (n_cycle - 1)
                and not self.freeze_pairformer
            ):
                if cycle_no == n_cycle - 1 and torch.is_autocast_enabled():
                    # Sidestep AMP bug (PyTorch issue #65766)
                    torch.clear_autocast_cache()
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if self.template_embedder.n_blocks > 0:
                    z = z + self.template_embedder(
                        input_feature_dict,
                        z,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                z = self.msa_module(
                    input_feature_dict,
                    z,
                    s_inputs,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        return s_inputs, s_init, z_init, s, z

    def get_dyna_prefix_features(
        self, input_feature_dict: dict[str, Any]
    ) -> tuple[Any, ...]:
        frame_feat = (
            input_feature_dict["frame_atom_index"],
            input_feature_dict["pae_rep_atom_mask"],
            input_feature_dict["has_frame"],
        )
        return (
            input_feature_dict["distogram_rep_atom_mask"],
            frame_feat,
            input_feature_dict["intervals"],
            input_feature_dict["coordinate"],
            input_feature_dict["coordinate_mask"],
            input_feature_dict.get("prefix_batch_padding_mask", None),
        )

    def get_atom_features(
        self, input_feature_dict: dict[str, Any]
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        batch_shape = input_feature_dict["ref_pos"].shape[:-2]
        n_atom = input_feature_dict["ref_pos"].shape[-2]
        atom_feat = (
            input_feature_dict["ref_pos"],
            input_feature_dict["ref_charge"],
            input_feature_dict["ref_mask"],
            input_feature_dict["ref_element"],
            input_feature_dict["ref_atom_name_chars"],
            input_feature_dict["ref_space_uid"],
        )
        atom_feat = tuple(
            entry.reshape(*batch_shape, n_atom, -1)
            for entry in atom_feat
        )
        return atom_feat, input_feature_dict["atom_to_token_idx"]

    def get_dyna_static_features(
        self, input_feature_dict: dict[str, Any]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return (
            input_feature_dict["distogram_rep_atom_mask"],
            (
                input_feature_dict["frame_atom_index"],
                input_feature_dict["pae_rep_atom_mask"],
                input_feature_dict["has_frame"],
            ),
        )

    def get_dynamic_temperature(
        self, intervals: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        return 320 * torch.ones(
            intervals.shape[:-1], device=intervals.device, dtype=dtype
        )

    def run_dyna_ssm(
        self,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        distogram_rep_atom_mask: torch.Tensor,
        frame_feat: tuple[torch.Tensor, ...],
        intervals: torch.Tensor,
        temperature: torch.Tensor,
        coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        prefix_batch_padding_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dyna_ssm(
            s_init=s_init,
            z_init=z_init,
            s=s,
            z=z,
            distogram_rep_atom_mask=distogram_rep_atom_mask,
            frame_feat=frame_feat,
            prefix_batch_padding_mask=prefix_batch_padding_mask,
            intervals=intervals,
            temperature=temperature,
            coordinate=coordinate,
            coordinate_mask=coordinate_mask,
            chunk_size=chunk_size,
        )

    def run_dyna_ssm_one_cycle(
        self,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        distogram_rep_atom_mask: torch.Tensor,
        frame_feat: tuple[torch.Tensor, ...],
        intervals: torch.Tensor,
        temperature: torch.Tensor,
        coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        s_next: Optional[torch.Tensor] = None,
        z_next: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dyna_ssm.one_cycle_forward(
            s_init=s_init,
            z_init=z_init,
            s=s,
            z=z,
            distogram_rep_atom_mask=distogram_rep_atom_mask,
            frame_feat=frame_feat,
            intervals=intervals,
            temperature=temperature,
            coordinate=coordinate,
            coordinate_mask=coordinate_mask,
            s_next=s_next,
            z_next=z_next,
            chunk_size=chunk_size,
        )

    def get_dynamic_conditioning(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            distogram_rep_atom_mask,
            frame_feat,
            intervals,
            coordinate,
            coordinate_mask,
            prefix_batch_padding_mask,
        ) = self.get_dyna_prefix_features(input_feature_dict)
        temperature = self.get_dynamic_temperature(intervals, s_init.dtype)

        s_next, z_next = self.run_dyna_ssm(
            s_init=s_init,
            z_init=z_init,
            s=s,
            z=z,
            distogram_rep_atom_mask=distogram_rep_atom_mask,
            frame_feat=frame_feat,
            intervals=intervals,
            temperature=temperature,
            coordinate=coordinate,
            coordinate_mask=coordinate_mask,
            prefix_batch_padding_mask=prefix_batch_padding_mask,
            chunk_size=chunk_size,
        )
        return append_batch_dim(s_inputs, s_next.shape[:-2]), s_next, z_next

    def _expand_relp_for_batch(
        self, input_feature_dict: dict[str, Any], batch_shape: tuple[int, ...]
    ) -> dict[str, Any]:
        if len(batch_shape) == 0:
            return input_feature_dict
        expanded = dict(input_feature_dict)
        expanded["relp"] = append_batch_dim(input_feature_dict["relp"], batch_shape)
        return expanded

    def prepare_diffusion_cache(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        inplace_safe: bool = False,
    ) -> dict[str, Any]:
        batch_shape = z.shape[:-3]
        diffusion_input_feature_dict = self._expand_relp_for_batch(
            input_feature_dict, batch_shape
        )
        return {"diffusion_input_feature_dict": diffusion_input_feature_dict}

    def sample_dynamic_diffusion(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        cache: dict[str, Any],
        n_sample: int,
        n_step: int,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        noise_schedule = self.inference_noise_scheduler(
            N_step=n_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        return self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=cache["diffusion_input_feature_dict"],
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_z=None,
            p_lm=None,
            c_l=None,
            N_sample=n_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )

    def train_forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        n_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward training pass.
        """
        disable_static_gradient = (
            torch.no_grad()
            if self.freeze_pairformer
            else nullcontext()
        )
        with disable_static_gradient:
            s_inputs, s_init, z_init, s, z = self.get_static_conditioning(
                input_feature_dict=input_feature_dict,
                n_cycle=n_cycle,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        s_inputs, s_next, z_next = self.get_dynamic_conditioning(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_init=s_init,
            z_init=z_init,
            s=s,
            z=z,
            chunk_size=chunk_size,
        )
        cache = self.prepare_diffusion_cache(input_feature_dict, z_next, inplace_safe)
        pred_dict = {}
        log_dict = {}

        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=cache["diffusion_input_feature_dict"],
            s_inputs=s_inputs,
            s_trunk=s_next,
            z_trunk=z_next,
            pair_z=None,
            p_lm=None,
            c_l=None,
            N_sample=self.diffusion_batch_size,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            use_conditioning=True,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(True)(self.distogram_head)(
                    z_next
                ),
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )
        return pred_dict, label_dict, log_dict

    def evaluate_forward(
        self,
        input_feature_dict: dict[str, Any],
        n_cycle: int,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        step_st = time.time()
        pred_dict = {}
        log_dict = {}
        time_tracker = {}
        s_inputs, s_init, z_init, s, z = self.get_static_conditioning(
            input_feature_dict=input_feature_dict,
            n_cycle=n_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        s_inputs, s_next, z_next = self.get_dynamic_conditioning(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_init=s_init,
            z_init=z_init,
            s=s,
            z=z,
            chunk_size=chunk_size,
        )
        time_tracker["pairformer"] = time.time() - step_st
        cache = self.prepare_diffusion_cache(input_feature_dict, z_next, inplace_safe)
        pred_dict["coordinate"] = self.sample_dynamic_diffusion(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s=s_next,
            z=z_next,
            cache=cache,
            n_sample=self.configs.sample_diffusion["N_sample"],
            n_step=self.configs.sample_diffusion["N_step"],
            inplace_safe=inplace_safe,
        )
        step_diffusion = time.time()
        time_tracker["diffusion"] = step_diffusion - step_st - time_tracker["pairformer"]
        pred_dict["distogram"] = autocasting_disable_decorator(True)(self.distogram_head)(
            z_next
        )
        time_tracker["model_forward"] = time.time() - step_st
        return pred_dict, log_dict, time_tracker

    def rollout_dynamic_trajectory(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        infer_trajectory_max_length: int,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        (
            distogram_rep_atom_mask,
            frame_feat,
            intervals,
            coordinate,
            coordinate_mask,
            _,
        ) = self.get_dyna_prefix_features(input_feature_dict)
        temperature = self.get_dynamic_temperature(intervals, s_init.dtype)

        if self.destination_enable:
            assert coordinate.shape[-3] >= 2
            # The destination horizon decreases by one per generated frame.
            # The first interval is the horizon, not a transition step.
            if torch.any(intervals[..., 1:] != 1):
                raise ValueError(
                    "Destination-conditioned inference requires unit transition "
                    "intervals (intervals[..., 1:] == 1); the first interval "
                    "specifies the destination horizon."
                )
            destination_intervals = intervals[..., :1]
            destination_coordinate = coordinate[..., :1, :, :]
            destination_coordinate_mask = coordinate_mask[..., :1, :]
            intervals = intervals[..., 1:]
            coordinate = coordinate[..., 1:, :, :]
            coordinate_mask = coordinate_mask[..., 1:, :]
        else:
            destination_intervals = None
            destination_coordinate = None
            destination_coordinate_mask = None

        assert intervals.shape[-1] == infer_trajectory_max_length - 1
        input_prefix_num = coordinate.shape[-3]
        assert input_prefix_num >= 1
        assert input_prefix_num < infer_trajectory_max_length
        assert input_prefix_num < self.model_max_track_length

        pred_coordinates = [coordinate]
        input_intervals = intervals[..., :input_prefix_num]
        input_coordinate = coordinate
        input_coordinate_mask = coordinate_mask
        if self.destination_enable:
            input_intervals = torch.cat(
                [destination_intervals, input_intervals], dim=-1
            )
            input_coordinate = torch.cat(
                [destination_coordinate, input_coordinate], dim=-3
            )
            input_coordinate_mask = torch.cat(
                [destination_coordinate_mask, input_coordinate_mask], dim=-2
            )

        s_next, z_next = self.run_dyna_ssm(
            s_init=s_init,
            z_init=z_init,
            s=s,
            z=z,
            distogram_rep_atom_mask=distogram_rep_atom_mask,
            frame_feat=frame_feat,
            intervals=input_intervals,
            temperature=temperature,
            coordinate=input_coordinate,
            coordinate_mask=input_coordinate_mask,
            prefix_batch_padding_mask=None,
            chunk_size=chunk_size,
        )
        if self.destination_enable:
            destination_intervals = destination_intervals - input_prefix_num

        for str_no in range(input_prefix_num, infer_trajectory_max_length):
            batch_shape = s_next.shape[:-2]
            cache = self.prepare_diffusion_cache(input_feature_dict, z_next, inplace_safe)
            pred_coordinate = self.sample_dynamic_diffusion(
                input_feature_dict=input_feature_dict,
                s_inputs=append_batch_dim(s_inputs, batch_shape),
                s=s_next,
                z=z_next,
                cache=cache,
                n_sample=self.configs.sample_diffusion["N_sample"],
                n_step=self.configs.sample_diffusion["N_step"],
                inplace_safe=inplace_safe,
            )
            assert self.configs.sample_diffusion["N_sample"] == 1

            pred_coordinates.append(pred_coordinate)
            coordinate = torch.cat([coordinate, pred_coordinate], dim=-3)
            coordinate_mask = torch.cat(
                [coordinate_mask, coordinate_mask[..., -1:, :]], dim=-2
            )

            if str_no == infer_trajectory_max_length - 1:
                break

            if input_prefix_num + 1 >= self.model_max_track_length:
                input_intervals = intervals[
                    ..., str_no - self.overlap_structure_num + 1 : str_no + 1
                ]
                input_coordinate = coordinate[..., -self.overlap_structure_num :, :, :]
                input_coordinate_mask = coordinate_mask[
                    ..., -self.overlap_structure_num :, :
                ]

                if self.destination_enable:
                    input_intervals = torch.cat(
                        [
                            destination_intervals + self.overlap_structure_num - 1,
                            input_intervals,
                        ],
                        dim=-1,
                    )
                    input_coordinate = torch.cat(
                        [destination_coordinate, input_coordinate], dim=-3
                    )
                    input_coordinate_mask = torch.cat(
                        [destination_coordinate_mask, input_coordinate_mask], dim=-2
                    )

                s_next, z_next = self.run_dyna_ssm(
                    s_init=s_init,
                    z_init=z_init,
                    s=s,
                    z=z,
                    distogram_rep_atom_mask=distogram_rep_atom_mask,
                    frame_feat=frame_feat,
                    intervals=input_intervals,
                    temperature=temperature,
                    coordinate=input_coordinate,
                    coordinate_mask=input_coordinate_mask,
                    prefix_batch_padding_mask=None,
                    chunk_size=chunk_size,
                )
                input_prefix_num = self.overlap_structure_num
            else:
                input_intervals = intervals[..., str_no : str_no + 1]
                input_coordinate = coordinate[..., str_no : str_no + 1, :, :]
                input_coordinate_mask = coordinate_mask[..., str_no : str_no + 1, :]

                assert coordinate.shape[-3] == str_no + 1
                assert coordinate_mask.shape[-2] == str_no + 1

                if self.destination_enable:
                    input_intervals = torch.cat(
                        [destination_intervals, input_intervals], dim=-1
                    )
                    input_coordinate = torch.cat(
                        [destination_coordinate, input_coordinate], dim=-3
                    )
                    input_coordinate_mask = torch.cat(
                        [destination_coordinate_mask, input_coordinate_mask], dim=-2
                    )

                s_next, z_next = self.run_dyna_ssm_one_cycle(
                    s_init=s_init,
                    z_init=z_init,
                    s=s,
                    z=z,
                    distogram_rep_atom_mask=distogram_rep_atom_mask,
                    frame_feat=frame_feat,
                    intervals=input_intervals,
                    temperature=temperature,
                    coordinate=input_coordinate,
                    coordinate_mask=input_coordinate_mask,
                    s_next=s_next,
                    z_next=z_next,
                    chunk_size=chunk_size,
                )
                input_prefix_num += 1

            if self.destination_enable:
                destination_intervals = destination_intervals - 1

        return torch.cat(pred_coordinates, dim=-3)

    def inference_forward(
        self,
        input_feature_dict: dict[str, Any],
        n_cycle: int,
        infer_trajectory_max_length: int,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        self.eval()
        s_inputs, s_init, z_init, s, z = self.get_static_conditioning(
            input_feature_dict=input_feature_dict,
            n_cycle=n_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        coordinates = self.rollout_dynamic_trajectory(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_init=s_init,
            z_init=z_init,
            s=s,
            z=z,
            infer_trajectory_max_length=infer_trajectory_max_length,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        pred_dict = {
            "coordinate": coordinates,
        }
        return pred_dict, {}, {}

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any] = None,
        label_dict: dict[str, Any] = None,
        mode: str = "inference",
        current_step: Optional[int] = None,
        disable_inplace: bool = False,
        infer_trajectory_max_length: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Atmos model.
        """
        assert mode in ["train", "eval", "inference"]
        not_use_gradient = not (self.training or torch.is_grad_enabled())
        inplace_safe = not_use_gradient and (not disable_inplace)
        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        if mode == "train":
            assert self.training
            assert label_dict is not None
            rng = np.random.RandomState(current_step)
            n_cycle = rng.randint(1, self.N_cycle + 1)
            pred_dict, label_dict, log_dict = self.train_forward(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                n_cycle=n_cycle,
                inplace_safe=inplace_safe,
                chunk_size=None,
            )
            log_dict["N_cycle"] = n_cycle
        else:
            n_token = input_feature_dict["residue_index"].shape[-1]
            chunk_size = self.configs.infer_setting.chunk_size
            if (
                hasattr(self.configs.infer_setting, "dynamic_chunk_size")
                and self.configs.infer_setting.dynamic_chunk_size
            ):
                chunk_size = self._get_dynamic_chunk_size(n_token)
            if mode == "inference":
                assert infer_trajectory_max_length is not None
                pred_dict, log_dict, time_tracker = self.inference_forward(
                    input_feature_dict=input_feature_dict,
                    n_cycle=self.N_cycle,
                    infer_trajectory_max_length=infer_trajectory_max_length,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            else:
                pred_dict, log_dict, time_tracker = self.evaluate_forward(
                    input_feature_dict=input_feature_dict,
                    n_cycle=self.N_cycle,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            log_dict["time"] = time_tracker
        return pred_dict, label_dict, log_dict


Atmos = AtmosProtenix
