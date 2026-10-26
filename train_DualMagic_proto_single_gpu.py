"""Single-GPU training for PDM.

The experiment retains the complete official-fix MagicNet+GALoss objective for
both students and a directional disagreement term. It adds
branch-local multi-prototype memories that (1) supervise labeled decoder
features and (2) help choose/correct the teacher on A/B disagreement voxels.

Large 3D computation graphs are back-propagated sequentially, so a single
experiment uses one CUDA device.  Original project and DualMagic files are not
modified.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import stat

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import RandomCrop, ToTensor, TwoStreamBatchSampler
from pdm_schedules import cosine_ramp_weight
from prototype_guidance import (
    MultiPrototypeBank,
    PrototypeDirectionalDisagreement,
    classwise_confident_mask,
)
from pdm_runtime import (
    Branch,
    LogitAdapter,
    aux_tag,
    build_parser,
    configure_dataset,
    configure_logging,
    current_consistency_weight,
    final_test,
    save_pair,
    update_ema_variables,
)
from utils import dualmagic_ops


def build_proto_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Single-GPU PDM training"
    parser.set_defaults(exp="DualMagicProto", aux_max_weight=0.02)
    for action in parser._actions:
        if action.dest in {"device_a", "device_b"}:
            action.help = argparse.SUPPRESS
    parser.add_argument(
        "--device", type=int, default=0,
        help="logical CUDA device inside CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--prototype_count", type=int, default=3)
    parser.add_argument("--prototype_momentum", type=float, default=0.99)
    parser.add_argument("--prototype_temperature", type=float, default=0.1)
    parser.add_argument("--prototype_samples_per_class", type=int, default=256)
    parser.add_argument(
        "--prototype_supervised_weight", type=float, default=0.02,
        help="labeled voxel-to-prototype classification weight",
    )
    parser.add_argument(
        "--prototype_disagreement_weight", type=float, default=0.02,
        help="prototype correction weight on selected disagreement voxels",
    )
    parser.add_argument(
        "--prototype_guidance_strength", type=float, default=0.35,
        help="blend between confidence/entropy reliability and prototype support",
    )
    parser.add_argument(
        "--prototype_confidence_quantile", type=float, default=0.7,
        help="per-class confidence quantile used to update prototypes",
    )
    parser.add_argument(
        "--prototype_min_confidence", type=float, default=0.25,
        help="absolute floor for unlabeled prototype updates",
    )
    parser.add_argument(
        "--prototype_min_teacher_score", type=float, default=0.45,
        help="minimum combined reliability for prototype disagreement supervision",
    )
    return parser


def compute_magicnet_proto_loss(
    branch: Branch,
    prototype_bank: MultiPrototypeBank,
    volume_batch: torch.Tensor,
    label_batch: torch.Tensor,
    loc_list,
    iteration: int,
    args,
    num_classes: int,
):
    """Original full MagicNet+GALoss loss plus reliable labeled prototype CE."""

    model = branch.model
    model.train()
    batch_size, _, _, height, _ = volume_batch.shape
    labeled_bs = args.labeled_bs
    unlabeled_bs = batch_size - labeled_bs
    unlabeled_volume = volume_batch[labeled_bs:]

    outputs, global_embedding = model(volume_batch)

    nb_cubes = height // args.cube_size
    part_ind, rec_ind = dualmagic_ops.get_part_and_rec_ind(
        volume_batch.shape, nb_cubes, 16, volume_batch.device
    )
    cross_mix = torch.gather(volume_batch, dim=0, index=part_ind)
    _, embedding = model(cross_mix)
    recovered_embedding = torch.gather(embedding, dim=0, index=rec_ind)
    outputs_unmix = model.forward_prediction_head(recovered_embedding)

    noise = torch.clamp(torch.randn_like(unlabeled_volume) * 0.1, -0.2, 0.2)
    with torch.no_grad():
        ema_output = branch.ema_model(unlabeled_volume + noise)[0]
        teacher_soft = F.softmax(ema_output, dim=1)
        teacher_class = teacher_soft.argmax(dim=1)

    outputs_soft = F.softmax(outputs, dim=1)
    outputs_unmix_soft = F.softmax(outputs_unmix, dim=1)
    supervised_loss = (
        branch.ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
        + branch.dice_loss(outputs_soft[:labeled_bs], label_batch[:labeled_bs])
        + branch.dice_loss(outputs_unmix_soft[:labeled_bs], label_batch[:labeled_bs])
    )
    supervised_terms = 3

    patch_list = dualmagic_ops.get_patch_list(volume_batch, args.cube_size)
    random_indices = torch.randperm(len(patch_list), device=volume_batch.device)
    location_loss, feat_list = dualmagic_ops.cube_location_loss(
        model, loc_list, patch_list, random_indices
    )

    embed_list = []
    for sample_index in range(batch_size):
        _, embed_tmp = model.forward_decoder(feat_list[sample_index])
        embed_list.append(embed_tmp.unsqueeze(0))
    embed_all = torch.cat(embed_list, dim=0)
    embed_all_unmix = dualmagic_ops.unmix_tensor(embed_all, volume_batch.shape)
    pred_all_unmix = model.forward_prediction_head(embed_all_unmix)
    supervised_loss = supervised_loss + branch.dice_loss(
        F.softmax(pred_all_unmix, dim=1)[:labeled_bs], label_batch[:labeled_bs]
    )
    supervised_terms += 1

    pred_class_mix = None
    pseudo_soft = teacher_soft
    with torch.no_grad():
        if iteration > 100:
            current_dist = branch.dist_logger.get_class_dist().to(volume_batch.device)
            current_dist = current_dist ** (1.0 / args.T_dist)
            current_dist = current_dist / current_dist.sum()
            current_dist = current_dist / current_dist.max()
            weight_map = current_dist[teacher_class].unsqueeze(1).repeat(
                1, num_classes, 1, 1, 1
            )
            within_logits = dualmagic_ops.get_mix_pl(
                model, feat_list, volume_batch.shape, unlabeled_bs
            )
            blended_logits = (1.0 - weight_map) * ema_output + weight_map * within_logits
            pseudo_soft = F.softmax(blended_logits, dim=1)
            pred_class_mix = pseudo_soft.argmax(dim=1)
            logger_class = pred_class_mix
        else:
            logger_class = teacher_class
        branch.dist_logger.append_class_list(logger_class.reshape(-1, 1))

    if iteration % 20 == 0 and branch.dist_logger.class_total_pixel_store:
        branch.dist_logger.update_class_dist()

    pseudo_target = teacher_class if pred_class_mix is None else pred_class_mix
    consistency_loss = branch.dice_loss(
        outputs_unmix_soft[labeled_bs:], pseudo_target
    )
    consistency_weight = current_consistency_weight(args, iteration)
    supervised_loss = supervised_loss / supervised_terms

    labeled_mask = torch.ones_like(label_batch[:labeled_bs], dtype=torch.bool)
    supervised_proto = prototype_bank.classification_loss(
        global_embedding[:labeled_bs], label_batch[:labeled_bs], labeled_mask
    )
    total = (
        supervised_loss
        + 0.1 * location_loss
        + consistency_weight * consistency_loss
        + args.prototype_supervised_weight * supervised_proto
    )
    return {
        "total": total,
        "global_logits_unlabeled": outputs[labeled_bs:],
        "global_embedding_labeled": global_embedding[:labeled_bs],
        "global_embedding_unlabeled": global_embedding[labeled_bs:],
        "pseudo_soft": pseudo_soft,
        "pseudo_target": pseudo_target,
        "supervised": supervised_loss,
        "supervised_proto": supervised_proto,
        "consistency": consistency_loss,
        "consistency_weight": consistency_weight,
        "location": location_loss,
    }


def _cache_result(result):
    return {
        "base": result["total"].detach(),
        "consistency": result["consistency"].detach(),
        "supervised_proto": result["supervised_proto"].detach(),
        "logits": result["global_logits_unlabeled"].detach(),
        "features_l": result["global_embedding_labeled"].detach(),
        "features_u": result["global_embedding_unlabeled"].detach(),
        "pseudo_soft": result["pseudo_soft"].detach(),
        "pseudo_target": result["pseudo_target"].detach(),
    }


@torch.no_grad()
def update_prototype_banks(
    bank_a: MultiPrototypeBank,
    bank_b: MultiPrototypeBank,
    cache_a,
    cache_b,
    labeled_target: torch.Tensor,
    args,
):
    labeled_valid = torch.ones_like(labeled_target, dtype=torch.bool)
    bank_a.update(cache_a["features_l"], labeled_target, labeled_valid)
    bank_b.update(cache_b["features_l"], labeled_target, labeled_valid)

    agreement = cache_a["pseudo_target"] == cache_b["pseudo_target"]
    update_a = classwise_confident_mask(
        cache_a["pseudo_soft"], cache_a["pseudo_target"], agreement,
        args.prototype_confidence_quantile, args.prototype_min_confidence,
    )
    update_b = classwise_confident_mask(
        cache_b["pseudo_soft"], cache_b["pseudo_target"], agreement,
        args.prototype_confidence_quantile, args.prototype_min_confidence,
    )
    bank_a.update(cache_a["features_u"], cache_a["pseudo_target"], update_a)
    bank_b.update(cache_b["features_u"], cache_b["pseudo_target"], update_b)
    return float(update_a.float().mean()), float(update_b.float().mean())


def train_single_gpu(args, config, snapshot_path: str):
    logger = configure_logging(snapshot_path, "train")
    logger.info("%s", args)
    device = torch.device(f"cuda:{args.device}")
    branch_a = Branch(device, args, config["num_classes"])
    branch_b = Branch(device, args, config["num_classes"])
    bank_kwargs = dict(
        num_classes=config["num_classes"],
        feature_dim=16,
        num_prototypes=args.prototype_count,
        momentum=args.prototype_momentum,
        temperature=args.prototype_temperature,
        samples_per_class=args.prototype_samples_per_class,
    )
    bank_a = MultiPrototypeBank(**bank_kwargs).to(device)
    bank_b = MultiPrototypeBank(**bank_kwargs).to(device)
    directional = PrototypeDirectionalDisagreement(
        temperature=args.aux_temperature,
        margin_scale=args.aux_margin_scale,
        prototype_strength=args.prototype_guidance_strength,
        minimum_teacher_score=args.prototype_min_teacher_score,
    )

    dataset = config["dataset_class"](
        config["labeled"], config["unlabeled"], base_dir=args.root_path,
        transform=transforms.Compose([RandomCrop((96, 96, 96)), ToTensor()]),
    )
    labeled_indices = list(range(len(config["unlabeled"]) * 2))
    unlabeled_indices = list(
        range(len(config["unlabeled"]) * 2, len(config["unlabeled"]) * 4)
    )
    sampler = TwoStreamBatchSampler(
        labeled_indices, unlabeled_indices, args.batch_size,
        args.batch_size - args.labeled_bs,
    )
    workers = (0 if os.name == "nt" else 2) if args.num_workers is None else args.num_workers

    def worker_init(worker_id):
        random.seed(args.seed + worker_id)

    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=workers, pin_memory=True,
        worker_init_fn=worker_init,
    )
    logger.info(
        "single GPU %s; labeled=%d, unlabeled=%d, eval=%d, test=%d, iterations_per_epoch=%d",
        device, len(config["labeled"]), len(config["unlabeled"]),
        len(config["eval"]), len(config["test"]), len(loader),
    )

    writer = SummaryWriter(snapshot_path)
    stop_iteration = args.stop_iteration if args.stop_iteration > 0 else args.max_iteration
    max_epoch = stop_iteration // len(loader) + 1
    iteration = 0
    best_dice = -1.0
    best_paths = None
    loc_list = None

    for _ in tqdm(range(max_epoch), ncols=80):
        for sampled_batch in loader:
            volume = sampled_batch["image"].to(device, non_blocking=True)
            labels = sampled_batch["label"].to(device, non_blocking=True)
            if loc_list is None:
                loc_list = dualmagic_ops.get_loc_mask(volume, args.cube_size)

            branch_a.optimizer.zero_grad(set_to_none=True)
            branch_b.optimizer.zero_grad(set_to_none=True)

            result_a = compute_magicnet_proto_loss(
                branch_a, bank_a, volume, labels, loc_list, iteration, args,
                config["num_classes"],
            )
            cache_a = _cache_result(result_a)
            result_a["total"].backward()
            del result_a

            result_b = compute_magicnet_proto_loss(
                branch_b, bank_b, volume, labels, loc_list, iteration, args,
                config["num_classes"],
            )
            cache_b = _cache_result(result_b)
            result_b["total"].backward()
            del result_b

            update_fraction_a, update_fraction_b = update_prototype_banks(
                bank_a, bank_b, cache_a, cache_b,
                labels[: args.labeled_bs], args,
            )
            state = directional.prepare(
                cache_a["logits"], cache_a["features_u"], bank_a,
                cache_b["logits"], cache_b["features_u"], bank_b,
            )
            aux_weight = cosine_ramp_weight(
                iteration, args.aux_max_weight,
                args.aux_warmup_end, args.aux_ramp_end,
            )

            zero = torch.zeros((), device=device)
            kl_a_value = zero
            kl_b_value = zero
            proto_a_value = zero
            proto_b_value = zero
            if state["disagreement_count"].item() > 0:
                logits_a, features_a = branch_a.model(volume)
                kl_a, proto_a = directional.losses_for_a(
                    logits_a[args.labeled_bs :], features_a[args.labeled_bs :],
                    bank_a, state,
                )
                aux_a = aux_weight * kl_a + args.prototype_disagreement_weight * proto_a
                if aux_a.requires_grad:
                    aux_a.backward()
                kl_a_value = kl_a.detach()
                proto_a_value = proto_a.detach()
                del logits_a, features_a, kl_a, proto_a, aux_a

                logits_b, features_b = branch_b.model(volume)
                kl_b, proto_b = directional.losses_for_b(
                    logits_b[args.labeled_bs :], features_b[args.labeled_bs :],
                    bank_b, state,
                )
                aux_b = aux_weight * kl_b + args.prototype_disagreement_weight * proto_b
                if aux_b.requires_grad:
                    aux_b.backward()
                kl_b_value = kl_b.detach()
                proto_b_value = proto_b.detach()
                del logits_b, features_b, kl_b, proto_b, aux_b

            branch_a.optimizer.step()
            branch_b.optimizer.step()
            update_ema_variables(branch_a.model, branch_a.ema_model, args.ema_decay, iteration)
            update_ema_variables(branch_b.model, branch_b.ema_model, args.ema_decay, iteration)
            iteration += 1

            learning_rate = args.base_lr * (1.0 - iteration / 70000.0) ** 0.9
            for optimizer in (branch_a.optimizer, branch_b.optimizer):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

            kl_value = kl_a_value + kl_b_value
            proto_disagreement_value = proto_a_value + proto_b_value
            base_value = cache_a["base"] + cache_b["base"]
            total_value = (
                base_value
                + aux_weight * kl_value
                + args.prototype_disagreement_weight * proto_disagreement_value
            )
            diagnostics = directional.last_diagnostics
            summary_a = bank_a.summary()
            summary_b = bank_b.summary()
            writer.add_scalar("train/loss_total", total_value.item(), iteration)
            writer.add_scalar("train/loss_magic_a", cache_a["base"].item(), iteration)
            writer.add_scalar("train/loss_magic_b", cache_b["base"].item(), iteration)
            writer.add_scalar("train/loss_disagreement", kl_value.item(), iteration)
            writer.add_scalar("train/loss_proto_disagreement", proto_disagreement_value.item(), iteration)
            writer.add_scalar("train/loss_proto_supervised_a", cache_a["supervised_proto"].item(), iteration)
            writer.add_scalar("train/loss_proto_supervised_b", cache_b["supervised_proto"].item(), iteration)
            writer.add_scalar("train/aux_weight", aux_weight, iteration)
            writer.add_scalar("prototype/coverage", diagnostics.prototype_coverage, iteration)
            writer.add_scalar("prototype/changed_direction", diagnostics.prototype_changed_direction, iteration)
            writer.add_scalar("prototype/ready_classes_a", summary_a["ready_classes"], iteration)
            writer.add_scalar("prototype/ready_classes_b", summary_b["ready_classes"], iteration)
            writer.add_scalar("prototype/update_fraction_a", update_fraction_a, iteration)
            writer.add_scalar("prototype/update_fraction_b", update_fraction_b, iteration)

            if iteration % 100 == 0:
                logger.info(
                    "iteration %d: total %.4f, magic_a %.4f, magic_b %.4f, "
                    "v3 %.5f (w %.5f), proto_sup %.5f/%.5f, proto_dis %.5f, "
                    "disagree %.4f, proto_cover %.4f, changed %.4f, ready %.0f/%.0f",
                    iteration, total_value.item(), cache_a["base"].item(),
                    cache_b["base"].item(), kl_value.item(), aux_weight,
                    cache_a["supervised_proto"].item(), cache_b["supervised_proto"].item(),
                    proto_disagreement_value.item(), diagnostics.disagreement_fraction,
                    diagnostics.prototype_coverage, diagnostics.prototype_changed_direction,
                    summary_a["ready_classes"], summary_b["ready_classes"],
                )

            if iteration % config["validation_interval"] == 0:
                branch_a.model.eval()
                branch_b.model.eval()
                adapter_a = LogitAdapter(branch_a.model, device)
                adapter_b = LogitAdapter(branch_b.model, device)
                dice_all, _, _ = config["pair_test"].validation_all_case_fast(
                    adapter_a, adapter_b,
                    num_classes=config["num_classes"], base_dir=args.root_path,
                    image_list=config["eval"], patch_size=(96, 96, 96),
                    stride_xy=config["val_strides"][0], stride_z=config["val_strides"][1],
                )
                dice_average = float(dice_all.mean())
                is_best = dice_average > best_dice
                if is_best:
                    best_dice = dice_average
                saved_paths = save_pair(
                    snapshot_path, iteration, dice_average, branch_a, branch_b, is_best
                )
                if is_best:
                    best_paths = saved_paths
                logger.info(
                    "iteration %d, ensemble validation DSC %.4f, best %.4f, saved %s / %s",
                    iteration, dice_average, best_dice, saved_paths[0], saved_paths[1],
                )
                writer.add_scalar("validation/ensemble_dsc", dice_average, iteration)
                writer.add_scalar("validation/best_ensemble_dsc", best_dice, iteration)
                branch_a.model.train()
                branch_b.model.train()

            del state, cache_a, cache_b
            if iteration >= stop_iteration:
                break
        if iteration >= stop_iteration:
            break

    writer.close()
    if best_paths is None:
        best_paths = save_pair(
            snapshot_path, iteration, -1.0, branch_a, branch_b, best=True
        )
        logger.warning("No validation ran; using final weights as the paired checkpoint.")
    logger.info("training complete; best ensemble DSC %.4f", best_dice)
    return branch_a, branch_b, best_paths


def main():
    args = build_proto_parser().parse_args()
    config = configure_dataset(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Prototype-Guided DualMagic requires CUDA")
    if args.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested logical CUDA device {args.device}, but only "
            f"{torch.cuda.device_count()} devices are visible"
        )
    if not 0 < args.labeled_bs < args.batch_size:
        raise ValueError("require 0 < labeled_bs < batch_size")

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    directory_name = (
        f"{args.dataset_name}_{args.exp}_GA_{args.labelnum}labeled_seed_{args.seed}_"
        f"aux{aux_tag(args.aux_max_weight)}_proto{args.prototype_count}"
    )
    snapshot_path = os.path.join(args.save_path, directory_name)
    if os.path.exists(snapshot_path) and os.listdir(snapshot_path):
        raise FileExistsError(
            f"output directory is not empty: {snapshot_path}; choose a different --exp"
        )
    os.makedirs(snapshot_path, exist_ok=True)
    os.chmod(snapshot_path, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
    # Exclude the selected output root as well as the usual large runtime
    # folders.  This prevents recursive source snapshots when users choose a
    # workspace-local --save_path such as ./runs.
    output_root_name = os.path.basename(os.path.abspath(args.save_path))
    shutil.copytree(
        ".", os.path.join(snapshot_path, "source"),
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "data", "model", "logs", "tmp",
            "reference_*", output_root_name,
        ),
    )
    print("snapshot_path:", snapshot_path)

    branch_a, branch_b, best_paths = train_single_gpu(args, config, snapshot_path)
    if not args.skip_final_test:
        final_test(args, config, snapshot_path, branch_a, branch_b, best_paths)


if __name__ == "__main__":
    main()
