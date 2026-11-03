from collections.abc import Sequence
from typing import Optional

import torch

from protenix.model.modules.frames import expressCoordinatesInFrame


def append_batch_dim(x: torch.Tensor, batch_shape: Sequence[int]) -> torch.Tensor:
    x = x.reshape(*((1,) * len(batch_shape)), *x.shape)
    return x.expand(*batch_shape, *((-1,) * (x.dim() - len(batch_shape))))


def fp32_no_autocast(func):
    def wrapper(*args, **kwargs):
        def cast_fp32(value):
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                return value.to(dtype=torch.float32)
            return value

        with torch.amp.autocast("cuda", enabled=False):
            return func(
                *(cast_fp32(value) for value in args),
                **{key: cast_fp32(value) for key, value in kwargs.items()},
            )

    return wrapper


def fp32_prep_distogram(
    distogram_rep_atom_mask: torch.Tensor,
    frame_atom_index: torch.Tensor,
    pae_rep_atom_mask: torch.Tensor,
    has_frame: torch.Tensor,
    coordinate: torch.Tensor,
    coordinate_mask: torch.Tensor,
    add_orientation: bool = True,
) -> tuple[torch.Tensor, ...]:
    return fp32_no_autocast(prep_distogram)(
        distogram_rep_atom_mask=distogram_rep_atom_mask,
        frame_atom_index=frame_atom_index,
        pae_rep_atom_mask=pae_rep_atom_mask,
        has_frame=has_frame,
        coordinate=coordinate,
        coordinate_mask=coordinate_mask,
        add_orientation=add_orientation,
    )


def prep_distogram(
    distogram_rep_atom_mask: torch.Tensor,
    frame_atom_index: torch.Tensor,
    pae_rep_atom_mask: torch.Tensor,
    has_frame: torch.Tensor,
    coordinate: torch.Tensor,
    coordinate_mask: torch.Tensor,
    add_orientation: bool = True,
) -> tuple[torch.Tensor, ...]:
    distogram_rep_atom_mask = distogram_rep_atom_mask.bool()
    token_coordinate = coordinate[..., distogram_rep_atom_mask, :]
    token_distogram = torch.cdist(token_coordinate, token_coordinate)

    token_mask = coordinate_mask[..., distogram_rep_atom_mask]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    token_distogram = token_distogram * pair_mask

    if not add_orientation:
        return token_distogram, pair_mask

    ori_distogram = prep_ori_distogram(
        frame_atom_index=frame_atom_index,
        pae_rep_atom_mask=pae_rep_atom_mask,
        has_frame=has_frame,
        coordinate=coordinate,
        coordinate_mask=coordinate_mask,
    )
    return token_distogram, pair_mask, ori_distogram


def prep_ori_distogram(
    frame_atom_index: torch.Tensor,
    pae_rep_atom_mask: torch.Tensor,
    has_frame: torch.Tensor,
    coordinate: torch.Tensor,
    coordinate_mask: torch.Tensor,
) -> torch.Tensor:
    batch_shape = coordinate.shape[:-2]

    def gather_frames(
        values: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        n_frame, n_axis = frame_indices.shape
        value_dim = values.shape[-1]
        frames = values.index_select(dim=-2, index=frame_indices.reshape(-1))
        return frames.view(*values.shape[:-2], n_frame, n_axis, value_dim)

    n_token = frame_atom_index.shape[0]
    pae_rep_atom_mask = pae_rep_atom_mask.bool()
    has_frame = has_frame.bool()
    frame_atom_index = frame_atom_index[has_frame, :]

    valid_frame_mask = (
        gather_frames(coordinate_mask.unsqueeze(-1), frame_atom_index).sum(dim=(-2, -1))
        >= 3
    )
    valid_token_mask = coordinate_mask[..., pae_rep_atom_mask].bool()
    valid_pair_mask = valid_frame_mask[..., None] * valid_token_mask[..., None, :]

    frames = gather_frames(coordinate, frame_atom_index)
    pae_rep_coordinate = coordinate[..., pae_rep_atom_mask, :]
    transformed_coordinate = expressCoordinatesInFrame(pae_rep_coordinate, frames)
    transformed_coordinate = transformed_coordinate / (
        torch.linalg.vector_norm(transformed_coordinate, dim=-1, keepdim=True) + 1e-8
    )

    ori_distogram = torch.zeros(
        *batch_shape,
        n_token,
        n_token,
        4,
        dtype=torch.float32,
        device=transformed_coordinate.device,
    )
    ori_distogram[..., has_frame, :, :-1] = transformed_coordinate
    ori_distogram[..., has_frame, :, -1] = valid_pair_mask.to(dtype=torch.float32)
    return torch.cat([ori_distogram, ori_distogram.transpose(-2, -3)], dim=-1)

