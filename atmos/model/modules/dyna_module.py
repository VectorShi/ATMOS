from collections.abc import Sequence
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from atmos.model.utils import append_batch_dim, fp32_no_autocast, fp32_prep_distogram
from protenix.model.modules.embedders import FourierEmbedding
from protenix.model.modules.pairformer import PairformerStack
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.triangular.layers import LayerNorm


class TimeIntervalEmbedding(nn.Module):
    """
    This module encodes time intervals using sinusoidal positional embedding,
    followed by layer normalization and a linear transformation without bias.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.layernorm = LayerNorm(d_model)
        self.linear_no_bias = LinearNoBias(d_model, d_model)

    def forward(self, interval: torch.Tensor) -> torch.Tensor:
        batch_shape = interval.shape
        embedding = self.pe[interval.long(), :] # [..., N_embed]
        return self.linear_no_bias(self.layernorm(embedding)).view(
            *batch_shape, 1, 1, -1
        )


class TemperatureEmbedding(nn.Module):
    """
    This module encodes temperature values using Fourier embedding,
    followed by layer normalization and a linear transformation without bias.
    """

    def __init__(
        self,
        c: int,
        max_temperature: float = 470,
        min_temperature: float = 260,
    ) -> None:
        super().__init__()
        self.fourier_embedding = FourierEmbedding(c)
        self.layernorm = LayerNorm(c)
        self.linear_no_bias = LinearNoBias(c, c)
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature

    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        batch_shape = temperature.shape
        temperature = (temperature - self.min_temperature) / (
            self.max_temperature - self.min_temperature
        )
        embedding = self.fourier_embedding(temperature) # [..., c]
        return self.linear_no_bias(self.layernorm(embedding)).view(
            *batch_shape, 1, 1, -1
        )


class DistogramEmbedding(nn.Module):
    """
    This module encodes pairwise distances between tokens using distogram binning,
    followed by layer normalization and linear transformations without bias.
    """

    def __init__(
        self,
        c_tokenpair: int,
        prefix_distogram_min_bin: float = 2.3125,
        prefix_distogram_max_bin: float = 21.6875,
        prefix_distogram_no_bins: int = 64,
    ) -> None:
        super().__init__()
        extra_dims = 1 + 8 # pair_mask + ori_distogram
        self.linear_no_bias_0 = LinearNoBias(
            prefix_distogram_no_bins + extra_dims,
            c_tokenpair,
        )
        self.layernorm = LayerNorm(c_tokenpair)
        self.linear_no_bias_1 = LinearNoBias(c_tokenpair, c_tokenpair)
        self.prefix_distogram_min_bin = prefix_distogram_min_bin
        self.prefix_distogram_max_bin = prefix_distogram_max_bin
        self.prefix_distogram_no_bins = prefix_distogram_no_bins

    def forward(
        self,
        pair_mask: torch.Tensor, # [..., N_token, N_token]
        token_distogram: torch.Tensor, # [..., N_token, N_token]
        ori_distogram: torch.Tensor, # [..., N_token, N_token, 8]
    ) -> torch.Tensor:
        pair_mask = pair_mask.to(dtype=token_distogram.dtype)
        token_distogram = token_distogram * pair_mask
        boundaries = torch.linspace(
            start=self.prefix_distogram_min_bin,
            end=self.prefix_distogram_max_bin,
            steps=self.prefix_distogram_no_bins - 1,
            device=token_distogram.device,
        )
        dist_bins = torch.sum(token_distogram.unsqueeze(-1) > boundaries, dim=-1)
        dist_bins = F.one_hot(dist_bins, self.prefix_distogram_no_bins).to(
            dtype=token_distogram.dtype
        ) # [..., N_token, N_token, no_bins]
        distogram_feat = torch.cat([dist_bins, pair_mask.unsqueeze(-1)], dim=-1)
        # Orientation features are concatenated without multiplying by pair_mask.
        distogram_feat = torch.cat([distogram_feat, ori_distogram], dim=-1)
        distogram_feat = self.linear_no_bias_0(distogram_feat)
        distogram_feat = self.layernorm(distogram_feat)
        distogram_feat = self.linear_no_bias_1(distogram_feat) # [..., N_token, N_token, C_tokenpair]
        return distogram_feat


class DynaSSMAA(nn.Module):
    
    def __init__(
        self,
        c_token: int,
        c_tokenpair: int,
        pairformer_n_block: int,
        blocks_per_ckpt: Optional[int] = None,
        triangle_multiplicative: str = "cuequivariance",
        triangle_attention: str = "triattention",
        max_temperature: float = 470,
        min_temperature: float = 260,
        prefix_distogram_min_bin: float = 2.3125,
        prefix_distogram_max_bin: float = 21.6875,
        prefix_distogram_no_bins: int = 64,
        prefix_fine_distogram_min_bin: float = 1,
        prefix_fine_distogram_max_bin: float = 5,
        prefix_fine_distogram_no_bins: int = 96,
        destination_enable: bool = False,
    ) -> None:
        super().__init__()
        self.triangle_multiplicative = triangle_multiplicative
        self.triangle_attention = triangle_attention
        self.destination_enable = destination_enable
        self.fp32_no_autocast = fp32_no_autocast

        self.linear_no_bias_z = LinearNoBias(c_tokenpair, c_tokenpair)
        self.layernorm_z = LayerNorm(c_tokenpair)
        self.linear_no_bias_s = LinearNoBias(c_token, c_token)
        self.layernorm_s = LayerNorm(c_token)
        self.linear_no_bias_z_next = LinearNoBias(c_tokenpair, c_tokenpair)
        self.layernorm_z_next = LayerNorm(c_tokenpair)
        self.linear_no_bias_s_next = LinearNoBias(c_token, c_token)
        self.layernorm_s_next = LayerNorm(c_token)

        self.pairformer_stack = PairformerStack(
            n_blocks=pairformer_n_block,
            c_s=c_token,
            c_z=c_tokenpair,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.interval_encoder = TimeIntervalEmbedding(d_model=c_tokenpair)
        self.temperature_encoder = TemperatureEmbedding(
            c=c_tokenpair,
            max_temperature=max_temperature,
            min_temperature=min_temperature,
        )
        self.distogram_encoder = DistogramEmbedding(
            c_tokenpair=c_tokenpair,
            prefix_distogram_min_bin=prefix_distogram_min_bin,
            prefix_distogram_max_bin=prefix_distogram_max_bin,
            prefix_distogram_no_bins=prefix_distogram_no_bins,
        )
        self.fine_distogram_encoder = DistogramEmbedding(
            c_tokenpair=c_tokenpair,
            prefix_distogram_min_bin=prefix_fine_distogram_min_bin,
            prefix_distogram_max_bin=prefix_fine_distogram_max_bin,
            prefix_distogram_no_bins=prefix_fine_distogram_no_bins,
        )

        if self.destination_enable:
            self.linear_no_bias_destination_in = LinearNoBias(
                2 * c_tokenpair, c_tokenpair
            )
            self.linear_no_bias_destination_out = LinearNoBias(c_tokenpair, c_tokenpair)
            self.layernorm_destination = LayerNorm(c_tokenpair)
            self.relu = nn.ReLU()

    def _encode_distogram(
        self,
        pair_mask: torch.Tensor,
        token_distogram: torch.Tensor,
        ori_distogram: Optional[torch.Tensor],
    ) -> torch.Tensor:
        embedding = self.fp32_no_autocast(self.distogram_encoder)(
            pair_mask=pair_mask,
            token_distogram=token_distogram,
            ori_distogram=ori_distogram,
        )
        embedding = embedding + self.fp32_no_autocast(self.fine_distogram_encoder)(
            pair_mask=pair_mask,
            token_distogram=token_distogram,
            ori_distogram=ori_distogram,
        )
        return embedding

    def _run_pairformer(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pairformer_stack(
            s,
            z,
            pair_mask=None,
            triangle_multiplicative=self.triangle_multiplicative,
            triangle_attention=self.triangle_attention,
            chunk_size=chunk_size,
        )

    def _prepare_distogram_features(
        self,
        distogram_rep_atom_mask: torch.Tensor, 
        frame_feat: Sequence[torch.Tensor],
        coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        frame_atom_index, pae_rep_atom_mask, has_frame = frame_feat
        prep_distogram_results = fp32_prep_distogram(
            distogram_rep_atom_mask=distogram_rep_atom_mask, # [N_atom]
            frame_atom_index=frame_atom_index, # [N_token, 3]
            pae_rep_atom_mask=pae_rep_atom_mask, # [N_atom]
            has_frame=has_frame, # [N_token]
            coordinate=coordinate,
            coordinate_mask=coordinate_mask,
            add_orientation=True,
        )
        token_distogram, pair_mask, ori_distogram = prep_distogram_results

        return token_distogram, pair_mask, ori_distogram

    def forward(
        self,
        # static conditioning & feat
        s_init: torch.Tensor, # [N_token, C_single]
        z_init: torch.Tensor, # [N_token, N_token, C_pair]
        s: torch.Tensor, # [N_token, C_single]
        z: torch.Tensor, # [N_token, N_token, C_pair]
        distogram_rep_atom_mask: torch.Tensor, # [N_atom]
        frame_feat: Sequence[torch.Tensor],
        # dynamic prefix feat
        prefix_batch_padding_mask: Optional[torch.Tensor], # [..., N_str_max]
        intervals: torch.Tensor, # [..., N_str_max]
        temperature: torch.Tensor, # [...]
        coordinate: torch.Tensor, # [..., N_str_max, N_atom, 3]
        coordinate_mask: torch.Tensor, # [..., N_str_max, N_atom]
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = intervals.shape[:-1]
        n_str_max = intervals.shape[-1]
        token_distogram, pair_mask, ori_distogram = self._prepare_distogram_features(
            distogram_rep_atom_mask=distogram_rep_atom_mask,
            frame_feat=frame_feat,
            coordinate=coordinate,
            coordinate_mask=coordinate_mask,
        )
        if prefix_batch_padding_mask is None:
            prefix_batch_padding_mask = torch.ones(
                (*batch_shape, n_str_max), device=s_init.device
            )
        prefix_batch_padding_mask = prefix_batch_padding_mask.bool()

        if self.destination_enable:
            n_str_max = n_str_max - 1
            destination_intervals = intervals[..., 0]
            destination_token_distogram = token_distogram[..., 0, :, :]
            destination_pair_mask = pair_mask[..., 0, :, :]
            destination_ori_distogram = ori_distogram[..., 0, :, :, :]
               
            intervals = intervals[..., 1:]
            token_distogram = token_distogram[..., 1:, :, :]
            pair_mask = pair_mask[..., 1:, :, :]
            coordinate = coordinate[..., 1:, :, :]
            coordinate_mask = coordinate_mask[..., 1:, :]
            prefix_batch_padding_mask = prefix_batch_padding_mask[..., 1:]
            ori_distogram = ori_distogram[..., 1:, :, :, :]

        s_init = append_batch_dim(s_init, batch_shape)
        z_init = append_batch_dim(z_init, batch_shape)
        s = append_batch_dim(s, batch_shape)
        z = append_batch_dim(z, batch_shape)
        s_next = torch.zeros_like(s_init)
        z_next = torch.zeros_like(z_init)

        for cycle_no in range(n_str_max):
            with torch.set_grad_enabled(self.training and cycle_no == n_str_max - 1):
                if cycle_no == n_str_max - 1 and torch.is_autocast_enabled():
                    # Sidestep AMP bug (PyTorch issue #65766)
                    torch.clear_autocast_cache()
                z_next_updated = (
                    z_init
                    + self.linear_no_bias_z(self.layernorm_z(z))
                    + self.linear_no_bias_z_next(self.layernorm_z_next(z_next))
                )
                z_next_updated = z_next_updated + self.interval_encoder(
                    intervals[..., cycle_no]
                )
                z_next_updated = z_next_updated + self.temperature_encoder(temperature)
                z_next_updated = z_next_updated + self._encode_distogram(
                    pair_mask=pair_mask[..., cycle_no, :, :],
                    token_distogram=token_distogram[..., cycle_no, :, :],
                    ori_distogram=ori_distogram[..., cycle_no, :, :, :],
                )

                if self.destination_enable:
                    with torch.no_grad():
                        destination_distogram_emb = self._encode_distogram(
                            pair_mask=destination_pair_mask,
                            token_distogram=destination_token_distogram,
                            ori_distogram=destination_ori_distogram,
                        )
                        destination_interval_emb = self.interval_encoder(
                            destination_intervals
                        )
                    destination_interval_emb = destination_interval_emb.expand_as(
                        destination_distogram_emb
                    )
                    destination_emb = torch.cat(
                        [destination_interval_emb, destination_distogram_emb], dim=-1
                    )
                    destination_emb = self.linear_no_bias_destination_in(destination_emb)
                    destination_emb = self.relu(destination_emb)
                    destination_emb = self.layernorm_destination(destination_emb)
                    destination_emb = self.linear_no_bias_destination_out(destination_emb)
                    z_next_updated = z_next_updated + destination_emb

                s_next_updated = (
                    s_init
                    + self.linear_no_bias_s(self.layernorm_s(s))
                    + self.linear_no_bias_s_next(self.layernorm_s_next(s_next))
                )

                s_next_updated, z_next_updated = self._run_pairformer(
                    s_next_updated, z_next_updated, chunk_size=chunk_size
                )
                valid_prefix = prefix_batch_padding_mask[..., cycle_no]
                s_next = torch.where(
                    valid_prefix.unsqueeze(-1).unsqueeze(-1), s_next_updated, s_next
                )
                z_next = torch.where(
                    valid_prefix.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                    z_next_updated,
                    z_next,
                )
                if self.destination_enable:
                    destination_intervals = destination_intervals - valid_prefix.to(
                        dtype=destination_intervals.dtype
                    )

        return s_next, z_next

    def one_cycle_forward(
        self,
        # static conditions & feat
        s_init: torch.Tensor, # [N_token, C_single]
        z_init: torch.Tensor, # [N_token, N_token, C_pair]
        s: torch.Tensor, # [N_token, C_single]
        z: torch.Tensor, # [N_token, N_token, C_pair]
        distogram_rep_atom_mask: torch.Tensor,
        frame_feat: Sequence[torch.Tensor],
        # dynamic prefix feat
        intervals: torch.Tensor, # [..., 1 or 2], 2 for destination_enable
        temperature: torch.Tensor, # [...]
        coordinate: torch.Tensor, # [..., 1 or 2, N_atom, 3]
        coordinate_mask: torch.Tensor, # [..., 1 or 2, N_atom]
        # dynamic conditions
        s_next: Optional[torch.Tensor] = None,
        z_next: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not self.training

        batch_shape = temperature.shape
        n_str_max = intervals.shape[-1]
        assert n_str_max in [1, 2]

        token_distogram, pair_mask, ori_distogram = self._prepare_distogram_features(
            distogram_rep_atom_mask=distogram_rep_atom_mask,
            frame_feat=frame_feat,
            coordinate=coordinate,
            coordinate_mask=coordinate_mask,
        )

        if self.destination_enable:
            assert n_str_max == 2
            destination_intervals = intervals[..., 0]
            destination_token_distogram = token_distogram[..., 0, :, :]
            destination_pair_mask = pair_mask[..., 0, :, :]
            destination_ori_distogram = ori_distogram[..., 0, :, :, :]
            intervals = intervals[..., 1]
            token_distogram = token_distogram[..., 1, :, :]
            pair_mask = pair_mask[..., 1, :, :]
            coordinate = coordinate[..., 1, :, :]
            coordinate_mask = coordinate_mask[..., 1, :]
            ori_distogram = ori_distogram[..., 1, :, :, :]
                
        else:
            intervals = intervals[..., 0]
            token_distogram = token_distogram[..., 0, :, :]
            pair_mask = pair_mask[..., 0, :, :]
            coordinate = coordinate[..., 0, :, :]
            coordinate_mask = coordinate_mask[..., 0, :]
            ori_distogram = ori_distogram[..., 0, :, :, :]

        s_init = append_batch_dim(s_init, batch_shape)
        z_init = append_batch_dim(z_init, batch_shape)
        s = append_batch_dim(s, batch_shape)
        z = append_batch_dim(z, batch_shape)
        if s_next is None:
            s_next = torch.zeros_like(s_init)
            z_next = torch.zeros_like(z_init)

        z_next_updated = (
            z_init
            + self.linear_no_bias_z(self.layernorm_z(z))
            + self.linear_no_bias_z_next(self.layernorm_z_next(z_next))
        )
        z_next_updated = z_next_updated + self.interval_encoder(intervals)
        z_next_updated = z_next_updated + self.temperature_encoder(temperature)
        z_next_updated = z_next_updated + self._encode_distogram(
            pair_mask=pair_mask,
            token_distogram=token_distogram,
            ori_distogram=ori_distogram,
        )

        if self.destination_enable:
            destination_distogram_emb = self._encode_distogram(
                pair_mask=destination_pair_mask,
                token_distogram=destination_token_distogram,
                ori_distogram=destination_ori_distogram,
            )
            destination_interval_emb = self.interval_encoder(destination_intervals)
            destination_interval_emb = destination_interval_emb.expand_as(
                destination_distogram_emb
            )
            destination_emb = torch.cat(
                [destination_interval_emb, destination_distogram_emb], dim=-1
            )
            destination_emb = self.linear_no_bias_destination_in(destination_emb)
            destination_emb = self.relu(destination_emb)
            destination_emb = self.layernorm_destination(destination_emb)
            destination_emb = self.linear_no_bias_destination_out(destination_emb)
            z_next_updated = z_next_updated + destination_emb

        s_next_updated = (
            s_init
            + self.linear_no_bias_s(self.layernorm_s(s))
            + self.linear_no_bias_s_next(self.layernorm_s_next(s_next))
        )

        return self._run_pairformer(
            s_next_updated, z_next_updated, chunk_size=chunk_size
        )
