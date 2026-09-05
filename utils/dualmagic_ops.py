"""Device-aware MagicNet cube operations used by DualMagic-v1.

The original helpers contain implicit ``.cuda()`` calls, which always target
logical CUDA device 0.  These equivalents allocate tensors on the device of
their inputs and include the official unlabeled-feature indexing correction.
The original helper files are intentionally left unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def get_loc_mask(volume_batch: torch.Tensor, cube_size: int = 32):
    _, _, w, h, d = volume_batch.shape
    sx = math.ceil((w - cube_size) / cube_size) + 1
    sy = math.ceil((h - cube_size) / cube_size) + 1
    sz = math.ceil((d - cube_size) / cube_size) + 1
    loc_list = []
    for x in range(1, sx + 1):
        for y in range(1, sy + 1):
            for z in range(1, sz + 1):
                value = (x - 1) + sx * (y - 1) + sx * sy * (z - 1)
                loc_list.append(torch.tensor([value], dtype=torch.long))
    return loc_list


def get_part_and_rec_ind(volume_shape, nb_cubes: int, nb_chnls: int, device):
    bs, channels, w, h, d = volume_shape
    rand_loc_ind = torch.argsort(
        torch.rand(bs, nb_cubes, nb_cubes, nb_cubes, device=device), dim=0
    )
    cube_part_ind = rand_loc_ind.view(bs, 1, nb_cubes, nb_cubes, nb_cubes)
    cube_part_ind = cube_part_ind.repeat_interleave(channels, dim=1)
    cube_part_ind = cube_part_ind.repeat_interleave(w // nb_cubes, dim=2)
    cube_part_ind = cube_part_ind.repeat_interleave(h // nb_cubes, dim=3)
    cube_part_ind = cube_part_ind.repeat_interleave(d // nb_cubes, dim=4)

    rec_ind = torch.argsort(rand_loc_ind, dim=0)
    cube_rec_ind = rec_ind.view(bs, 1, nb_cubes, nb_cubes, nb_cubes)
    cube_rec_ind = cube_rec_ind.repeat_interleave(nb_chnls, dim=1)
    cube_rec_ind = cube_rec_ind.repeat_interleave(w // nb_cubes, dim=2)
    cube_rec_ind = cube_rec_ind.repeat_interleave(h // nb_cubes, dim=3)
    cube_rec_ind = cube_rec_ind.repeat_interleave(d // nb_cubes, dim=4)
    return cube_part_ind, cube_rec_ind


def get_patch_list(volume_batch: torch.Tensor, cube_size: int = 32):
    _, _, w, h, d = volume_batch.shape
    w_ = w // cube_size * cube_size
    h_ = h // cube_size * cube_size
    d_ = d // cube_size * cube_size
    sx = math.ceil((w_ - cube_size) / cube_size) + 1
    sy = math.ceil((h_ - cube_size) / cube_size) + 1
    sz = math.ceil((d_ - cube_size) / cube_size) + 1
    patch_list = []
    for x in range(1, sx + 1):
        xs = min(cube_size * (x - 1), w_ - cube_size)
        for y in range(1, sy + 1):
            ys = min(cube_size * (y - 1), h_ - cube_size)
            for z in range(1, sz + 1):
                zs = min(cube_size * (z - 1), d_ - cube_size)
                patch = volume_batch[
                    :, :, xs : xs + cube_size, ys : ys + cube_size, zs : zs + cube_size
                ]
                patch_list.append(patch.unsqueeze(1))
    return patch_list


def unmix_tensor(patch_tensor: torch.Tensor, unmix_shape):
    _, _, w, h, d = unmix_shape
    bs, _, channels, cube_x, cube_y, cube_z = patch_tensor.shape
    sx = math.ceil((w - cube_x) / cube_x) + 1
    sy = math.ceil((h - cube_y) / cube_y) + 1
    sz = math.ceil((d - cube_z) / cube_z) + 1
    result = patch_tensor.new_zeros((bs, channels, w, h, d))
    for x in range(1, sx + 1):
        xs = min(cube_x * (x - 1), w - cube_x)
        for y in range(1, sy + 1):
            ys = min(cube_y * (y - 1), h - cube_y)
            for z in range(1, sz + 1):
                zs = min(cube_z * (z - 1), d - cube_z)
                loc = (x - 1) + sx * (y - 1) + sx * sy * (z - 1)
                result[:, :, xs : xs + cube_x, ys : ys + cube_y, zs : zs + cube_z] = (
                    patch_tensor[:, loc, :]
                )
    return result


def cube_location_loss(model, loc_list, patch_list, idx):
    patches = torch.cat(patch_list, dim=1)
    loc_mask = torch.cat(loc_list, dim=0).to(patches.device)
    loc_loss = patches.new_zeros(())
    feat_list = []
    for sample_index in range(patches.shape[0]):
        feat_patch = model.forward_encoder(patches[sample_index, :])
        feat_list.append(feat_patch)
        feat_flatten = torch.flatten(feat_patch[-1], start_dim=1, end_dim=4)
        feat_tmp = feat_flatten[idx, :].view(feat_flatten.size())
        loc_mask_tmp = loc_mask[idx].view(loc_mask.size())
        loc_pred = F.log_softmax(model.fc_layer(feat_tmp), dim=1)
        loc_loss = loc_loss + F.nll_loss(loc_pred, loc_mask_tmp)
    return loc_loss / patches.shape[0], feat_list


def get_mix_pl(model, feat_list, ori_shape, unlabeled_bs: int):
    """Decode the last ``unlabeled_bs`` feature entries (official fix)."""

    if unlabeled_bs <= 0 or len(feat_list) < unlabeled_bs:
        raise ValueError("invalid unlabeled batch size for feat_list")
    start = len(feat_list) - unlabeled_bs
    with torch.no_grad():
        pred_list = []
        for i in range(unlabeled_bs):
            pred = model.forward_decoder(feat_list[start + i])[0].detach()
            pred_list.append(pred.unsqueeze(0))
        return unmix_tensor(torch.cat(pred_list, dim=0), ori_shape)

