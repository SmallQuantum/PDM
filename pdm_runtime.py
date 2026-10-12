"""Shared runtime utilities for the final PDM training entry."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataloaders.dataset import (
    AMOS_fast,
    Synapse_fast,
)
from GALoss import GACE, GADice
from networks.magicnet import VNet_Magic
from utils import (
    ramps,
    test_amos,
    test_amos_vnet_AB,
    test_util,
    test_util_vnet_AB,
)
from utils.cube_utils import OrganClassLogger


ORGAN_NAMES = {
    "Synapse": [
        "spleen", "r.kidney", "l.kidney", "gallbladder", "esophagus", "liver",
        "stomach", "aorta", "inferior vena cava", "portal and splenic vein",
        "pancreas", "right adrenal gland", "left adrenal gland",
    ],
    "AMOS": [
        "spleen", "r.kidney", "l.kidney", "gallbladder", "esophagus", "liver",
        "stomach", "aorta", "inferior vena cava", "pancreas",
        "right adrenal gland", "left adrenal gland", "duodenum", "bladder",
        "prostate/uterus",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["Synapse", "AMOS"], default="Synapse")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument(
        "--synapse_split_path", type=str,
        default="./data/Synapse_npy/splits",
        help="directory containing the released DHC/SKCDF Synapse split files",
    )
    parser.add_argument("--amos_split_path", type=str, default="./data/amos_splits")
    parser.add_argument("--save_path", type=str, default="./model")
    parser.add_argument("--exp", type=str, default="DualMagicV2FullBase")
    parser.add_argument("--max_iteration", type=int, default=17000)
    parser.add_argument("--stop_iteration", type=int, default=0)
    parser.add_argument("--labelnum", type=int, default=None)
    parser.add_argument("--labeled_bs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cube_size", type=int, default=32)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--consistency", type=float, default=0.1)
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument("--T_dist", type=float, default=1.0)
    parser.add_argument("--device_a", type=int, default=0)
    parser.add_argument("--device_b", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--validation_interval", type=int, default=0)
    parser.add_argument("--aux_max_weight", type=float, default=0.02)
    parser.add_argument("--aux_warmup_end", type=int, default=1000)
    parser.add_argument("--aux_ramp_end", type=int, default=5000)
    parser.add_argument("--aux_temperature", type=float, default=1.0)
    parser.add_argument("--aux_margin_scale", type=float, default=0.1)
    parser.add_argument("--skip_final_test", action="store_true")
    return parser


def read_split_list(split_dir: str, split: str):
    path = os.path.join(split_dir, f"{split}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"split file not found: {path}")
    values = np.loadtxt(path, dtype=str).tolist()
    if isinstance(values, str):
        values = [values]
    return sorted(values)


def validate_partition(groups, expected_total: int, name: str):
    for group_name, values in groups.items():
        if not values:
            raise ValueError(f"{name} {group_name} split is empty")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} {group_name} split contains duplicate IDs")
    group_names = list(groups)
    for index, left_name in enumerate(group_names):
        for right_name in group_names[index + 1:]:
            overlap = sorted(set(groups[left_name]) & set(groups[right_name]))
            if overlap:
                raise ValueError(
                    f"{name} split overlap between {left_name} and "
                    f"{right_name}: {overlap}"
                )
    union = set().union(*(set(values) for values in groups.values()))
    if len(union) != expected_total:
        raise ValueError(
            f"{name} split must cover {expected_total} unique cases, "
            f"found {len(union)}"
        )


def configure_dataset(args):
    if args.dataset_name == "Synapse":
        args.root_path = args.root_path or "./data/Synapse/"
        args.labelnum = 4 if args.labelnum is None else args.labelnum
        split_prefix = {2: "10p", 4: "20p", 8: "40p"}.get(args.labelnum)
        if split_prefix is None:
            raise ValueError(
                "DHC/SKCDF Synapse split supports labelnum 2, 4, or 8 "
                "(10%, 20%, or 40% labeled)"
            )
        groups = {
            "labeled": read_split_list(
                args.synapse_split_path, f"labeled_{split_prefix}"
            ),
            "unlabeled": read_split_list(
                args.synapse_split_path, f"unlabeled_{split_prefix}"
            ),
            "eval": read_split_list(args.synapse_split_path, "eval"),
            "test": read_split_list(args.synapse_split_path, "test"),
        }
        validate_partition(groups, expected_total=30, name="DHC/SKCDF Synapse")
        return {
            **groups,
            "dataset_class": Synapse_fast,
            "num_classes": 14,
            "validation_interval": args.validation_interval or 500,
            "val_strides": (80, 64),
            "single_test": test_util,
            "pair_test": test_util_vnet_AB,
        }

    args.root_path = args.root_path or "./data/AMOS/"
    args.labelnum = 10 if args.labelnum is None else args.labelnum
    split_prefix = {4: "2p", 10: "5p"}.get(args.labelnum)
    if split_prefix is None:
        raise ValueError("AMOS labelnum must be 4 or 10")
    return {
        "labeled": read_split_list(args.amos_split_path, f"labeled_{split_prefix}"),
        "unlabeled": read_split_list(args.amos_split_path, f"unlabeled_{split_prefix}"),
        "eval": read_split_list(args.amos_split_path, "eval"),
        "test": read_split_list(args.amos_split_path, "test"),
        "dataset_class": AMOS_fast,
        "num_classes": 16,
        "validation_interval": args.validation_interval or 1000,
        "val_strides": (90, 80),
        "single_test": test_amos,
        "pair_test": test_amos_vnet_AB,
    }


def aux_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def configure_logging(snapshot_path: str, name: str):
    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(os.path.join(snapshot_path, f"log_{name}.txt"), mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def create_model(device, num_classes: int, cube_size: int, ema: bool = False):
    model = VNet_Magic(
        n_channels=1,
        n_classes=num_classes,
        cube_size=cube_size,
        patch_size=96,
    ).to(device)
    if ema:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha: float, global_step: int):
    alpha = min(1.0 - 1.0 / (global_step + 1.0), alpha)
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(alpha).add_(parameter, alpha=1.0 - alpha)


def current_consistency_weight(args, iteration: int) -> float:
    return args.consistency * ramps.sigmoid_rampup(
        iteration // 350, args.consistency_rampup
    )


class Branch:
    def __init__(self, device, args, num_classes: int):
        self.device = device
        self.model = create_model(device, num_classes, args.cube_size)
        self.ema_model = create_model(device, num_classes, args.cube_size, ema=True)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001
        )
        self.dice_loss = GADice()
        self.ce_loss = GACE(k=10, gama=0.5)
        self.dist_logger = OrganClassLogger(num_classes=num_classes)




class LogitAdapter(nn.Module):
    """Return MagicNet logits on the evaluator input device."""

    def __init__(self, model: nn.Module, model_device):
        super().__init__()
        self.model = model
        self.model_device = model_device

    def forward(self, inputs):
        return self.model(inputs.to(self.model_device, non_blocking=True))[0].to(
            inputs.device, non_blocking=True
        )


class TupleAdapter(LogitAdapter):
    """Match the tuple output expected by the original single-model evaluator."""

    def forward(self, inputs):
        logits = super().forward(inputs)
        return logits, None


def save_pair(snapshot_path, iteration, dice, branch_a, branch_b, best: bool):
    suffix = "_best" if best else ""
    stem = f"iter_{iteration:05d}_dice_{dice:.6f}{suffix}"
    path_a = os.path.join(snapshot_path, stem + "_A.pth")
    path_b = os.path.join(snapshot_path, stem + "_B.pth")
    torch.save(branch_a.model.state_dict(), path_a)
    torch.save(branch_b.model.state_dict(), path_b)
    return path_a, path_b


def load_state(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def summarize_metrics(logger, title: str, metrics, organ_names):
    metric_mean = np.mean(metrics, axis=0)
    metric_std = np.std(metrics, axis=0)
    logger.info(
        "%s Average DSC: %.4f, HD95: %.4f, NSD: %.4f, ASD: %.4f",
        title,
        metric_mean[0].mean(),
        metric_mean[1].mean(),
        metric_mean[2].mean(),
        metric_mean[3].mean(),
    )
    for index, organ in enumerate(organ_names):
        logger.info(
            "%s: DSC %.4f+-%.4f, HD95 %.4f+-%.4f, NSD %.4f+-%.4f, ASD %.4f+-%.4f",
            organ,
            metric_mean[0][index], metric_std[0][index],
            metric_mean[1][index], metric_std[1][index],
            metric_mean[2][index], metric_std[2][index],
            metric_mean[3][index], metric_std[3][index],
        )




def final_test(args, config, snapshot_path, branch_a, branch_b, best_paths):
    device_a = branch_a.device
    device_b = branch_b.device
    branch_a.model.load_state_dict(load_state(best_paths[0], device_a))
    branch_b.model.load_state_dict(load_state(best_paths[1], device_b))
    branch_a.model.eval()
    branch_b.model.eval()

    tuple_a = TupleAdapter(branch_a.model, device_a)
    tuple_b = TupleAdapter(branch_b.model, device_b)
    logits_a = LogitAdapter(branch_a.model, device_a)
    logits_b = LogitAdapter(branch_b.model, device_b)
    common = dict(
        num_classes=config["num_classes"],
        base_dir=args.root_path,
        image_list=config["test"],
        patch_size=(96, 96, 96),
        stride_xy=32,
        stride_z=16,
    )
    _, _, metrics_a = config["single_test"].validation_all_case(tuple_a, **common)
    _, _, metrics_b = config["single_test"].validation_all_case(tuple_b, **common)
    _, _, metrics_ensemble = config["pair_test"].validation_all_case(
        logits_a, logits_b, **common
    )
    np.save(os.path.join(snapshot_path, "metric_final_A.npy"), metrics_a)
    np.save(os.path.join(snapshot_path, "metric_final_B.npy"), metrics_b)
    np.save(os.path.join(snapshot_path, "metric_final_ensemble.npy"), metrics_ensemble)

    logger = configure_logging(snapshot_path, "total_metric")
    summarize_metrics(logger, "Student A", metrics_a, ORGAN_NAMES[args.dataset_name])
    summarize_metrics(logger, "Student B", metrics_b, ORGAN_NAMES[args.dataset_name])
    summarize_metrics(
        logger, "A/B logit ensemble", metrics_ensemble, ORGAN_NAMES[args.dataset_name]
    )




