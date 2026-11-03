# This file includes portions adapted from Protenix protenix/model/loss.py.
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

# Modified for ATMOS: trajectory losses and optional loss terms.

from typing import Any, Callable

import torch
import torch.nn.functional as F

from protenix.model.loss import ProtenixLoss, logger


class AtmosLoss(ProtenixLoss):

    def _loss_enabled(self, loss_name: str) -> bool:
        weight = self.loss_weight[loss_name]
        if isinstance(weight, torch.Tensor):
            return bool((weight != 0).any().item())
        return float(weight) != 0.0

    @staticmethod
    def _zero_loss_like(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _distogram_loss(
        self,
        logits: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        rep_atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        boundaries = torch.linspace(
            start=self.distogram_loss.min_bin,
            end=self.distogram_loss.max_bin,
            steps=self.distogram_loss.no_bins - 1,
            device=true_coordinate.device,
        )
        rep_atom_mask = rep_atom_mask.bool()
        token_coordinate = true_coordinate[..., rep_atom_mask, :]
        true_distance = torch.cdist(
            token_coordinate,
            token_coordinate,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        labels = torch.sum(
            (true_distance.unsqueeze(dim=-1) > boundaries).long(), dim=-1
        )

        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = (token_mask[..., None] * token_mask[..., None, :]).bool()
        if labels[pair_mask].numel() == 0:
            return logits.sum() * 0.0
        return F.cross_entropy(logits[pair_mask, :], labels[pair_mask])

    def aggregate_losses(
        self,
        loss_fns: dict[str, Callable],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Aggregates multiple loss functions and their respective metrics.

        Args:
            loss_fns (dict[str, Callable]): Dictionary of loss functions to be aggregated.
            reference (torch.Tensor): Reference tensor used for device/dtype when all losses are disabled.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - all_metrics (dict[str, torch.Tensor]): Dictionary containing all metrics.
        """
        cum_loss = None
        all_metrics = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss_weight[loss_name]
            if not self._loss_enabled(loss_name):
                continue

            loss_outputs = loss_fn()
            if isinstance(loss_outputs, tuple):
                loss, metrics = loss_outputs
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics = loss_outputs, {}

            all_metrics.update(
                {f"{loss_name}/{key}": val for key, val in metrics.items()}
            )
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"{loss_name} loss is NaN. Skipping...")

            all_metrics[loss_name] = loss.detach().clone()
            all_metrics[f"weighted_{loss_name}"] = weight * loss.detach().clone()

            weighted_loss = weight * loss
            cum_loss = weighted_loss if cum_loss is None else cum_loss + weighted_loss

        if cum_loss is None:
            cum_loss = self._zero_loss_like(reference)
        all_metrics["loss"] = cum_loss.detach().clone()

        return cum_loss, all_metrics

    def calculate_losses(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate the cumulative loss and aggregated metrics for the given predictions and labels.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - metrics (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        assert mode in ["train", "eval"]
        if mode == "train" and self._loss_enabled("mse_loss"):
            # Scale diffusion loss with noise-level
            diffusion_per_sample_scale = (
                pred_dict["noise_level"] ** 2 + self.configs.sigma_data**2
            ) / (self.configs.sigma_data * pred_dict["noise_level"]) ** 2

        else:
            # No scale is required
            diffusion_per_sample_scale = None

        loss_fns = {}
        if self._loss_enabled("mse_loss"):
            loss_fns.update(
                {
                    "mse_loss": lambda: self.mse_loss(
                        pred_coordinate=pred_dict["coordinate"],
                        true_coordinate=label_dict["coordinate"],
                        coordinate_mask=label_dict["coordinate_mask"],
                        is_rna=feat_dict["is_rna"],
                        is_dna=feat_dict["is_dna"],
                        is_ligand=feat_dict["is_ligand"],
                        per_sample_scale=diffusion_per_sample_scale,
                    ),
                }
            )
        # Distogram Loss
        if "distogram" in pred_dict and self._loss_enabled("distogram_loss"):
            loss_fns.update(
                {
                    "distogram_loss": lambda: self._distogram_loss(
                        logits=pred_dict["distogram"],
                        true_coordinate=label_dict["coordinate"],
                        coordinate_mask=label_dict["coordinate_mask"],
                        rep_atom_mask=feat_dict["distogram_rep_atom_mask"],
                    )
                }
            )

        cum_loss, metrics = self.aggregate_losses(
            loss_fns, reference=pred_dict["coordinate"]
        )
        return cum_loss, metrics

    def forward(
        self,
        feat_dict: dict[str, Any],
        pred_dict: dict[str, torch.Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for calculating the cumulative loss and aggregated metrics.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - losses (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        diffusion_chunk_size = self.configs.loss.diffusion_chunk_size_outer
        assert mode in ["train", "eval"]

        if mode == "train":
            # Calculate losses
            cum_loss, losses = self.calculate_losses(
                feat_dict=feat_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                mode=mode,
            )
        else:
            if "coordinate" in pred_dict:
                N_sample = pred_dict["coordinate"].shape[-3]
            else:
                raise KeyError("Missing key: coordinate (in pred_dict).")
            if diffusion_chunk_size is None:
                diffusion_chunk_size = N_sample
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            cum_loss = None
            losses = {}
            for i in range(no_chunks):
                start = i * diffusion_chunk_size
                end = (i + 1) * diffusion_chunk_size
                cur_sample_num = min(
                    diffusion_chunk_size, N_sample - start
                )
                pred_dict_i = {}
                for key, value in pred_dict.items():
                    if key in [
                        "coordinate",
                    ]:
                        pred_dict_i[key] = value[..., start:end, :, :]
                    elif key == "noise_level":
                        pred_dict_i[key] = value[..., start:end]
                    else:
                        pred_dict_i[key] = value
                cum_loss_i, losses_i = self.calculate_losses(
                    feat_dict=feat_dict,
                    pred_dict=pred_dict_i,
                    label_dict=label_dict,
                    mode=mode,
                )
                weighted_loss_i = cum_loss_i * cur_sample_num
                cum_loss = (
                    weighted_loss_i
                    if cum_loss is None
                    else cum_loss + weighted_loss_i
                )
                # Aggregate metrics
                for key, value in losses_i.items():
                    if key in losses:
                        if "per_sample" in key:
                            losses[key] = torch.cat([losses[key], value], dim=-1)
                        else:
                            losses[key] += value * cur_sample_num
                    else:
                        if "per_sample" in key:
                            losses[key] = value
                        else:
                            losses[key] = value * cur_sample_num
            if cum_loss is None:
                cum_loss = self._zero_loss_like(pred_dict["coordinate"])
            cum_loss /= N_sample
            for key in losses.keys():
                if "per_sample" not in key:
                    losses[key] /= N_sample
        return cum_loss, losses
