"""Prototype guidance for the single-GPU DualMagic experiment.

This module adapts three ideas from public reference implementations without
copying their training frameworks:

* MPER: multiple micro-prototypes per semantic class and cosine prototype
  classification;
* PCCS: uncertainty-aware selection and EMA-style prototype updates;
* PatchCL-MedSeg: class-conditional feature attraction/repulsion.

MagicNet students are independently initialised, so each branch owns a
prototype bank in its *own* feature space.  Cross-branch decisions compare the
bounded support scores produced by the two banks instead of mixing raw feature
vectors from incompatible spaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def prediction_reliability(probabilities: torch.Tensor) -> torch.Tensor:
    """Confidence times one-minus-normalised entropy, in ``[0, 1]``."""

    confidence = probabilities.max(dim=1).values
    entropy = -torch.sum(
        probabilities * probabilities.clamp_min(1e-8).log(), dim=1
    ) / math.log(max(probabilities.shape[1], 2))
    return confidence * (1.0 - entropy).clamp(0.0, 1.0)


@torch.no_grad()
def classwise_confident_mask(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    quantile: float,
    minimum_confidence: float,
) -> torch.Tensor:
    """Keep the most confident candidate voxels independently per class.

    A percentile is preferable to a single global threshold for imbalanced
    multi-organ data: tail classes compete with themselves, not with liver or
    background.  ``minimum_confidence`` prevents random early predictions from
    entering the prototype bank.
    """

    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must lie in [0, 1)")
    confidence = probabilities.max(dim=1).values
    selected = torch.zeros_like(candidate_mask, dtype=torch.bool)
    for class_index in range(probabilities.shape[1]):
        class_mask = candidate_mask & (targets == class_index)
        values = confidence[class_mask]
        if values.numel() == 0:
            continue
        threshold = torch.quantile(values.float(), quantile).to(values.dtype)
        threshold = torch.maximum(
            threshold, values.new_tensor(float(minimum_confidence))
        )
        selected |= class_mask & (confidence >= threshold)
    return selected


class MultiPrototypeBank(nn.Module):
    """Online class-wise micro-prototypes for a single feature space."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 16,
        num_prototypes: int = 3,
        momentum: float = 0.99,
        temperature: float = 0.1,
        samples_per_class: int = 256,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if feature_dim < 1 or num_prototypes < 1:
            raise ValueError("feature_dim and num_prototypes must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if temperature <= 0.0 or samples_per_class < 1:
            raise ValueError("temperature and samples_per_class must be positive")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.num_prototypes = int(num_prototypes)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.samples_per_class = int(samples_per_class)
        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, num_prototypes, feature_dim),
        )
        self.register_buffer(
            "initialised", torch.zeros(num_classes, num_prototypes, dtype=torch.bool)
        )
        self.register_buffer("updates", torch.zeros(num_classes, dtype=torch.long))

    def _sample_class_features(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        if features.ndim != 5 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape [B, feature_dim, H, W, D]")
        if targets.shape != features.shape[:1] + features.shape[2:]:
            raise ValueError("targets and features have incompatible shapes")
        if valid_mask.shape != targets.shape:
            raise ValueError("valid_mask and targets must have identical shapes")

        sampled: List[List[torch.Tensor]] = [list() for _ in range(self.num_classes)]
        for batch_index in range(features.shape[0]):
            flat_target = targets[batch_index].reshape(-1)
            flat_valid = valid_mask[batch_index].reshape(-1)
            feature_columns = features[batch_index].reshape(self.feature_dim, -1)
            for class_index in range(self.num_classes):
                indices = torch.nonzero(
                    flat_valid & (flat_target == class_index), as_tuple=False
                ).flatten()
                if indices.numel() == 0:
                    continue
                remaining = self.samples_per_class - sum(
                    part.shape[0] for part in sampled[class_index]
                )
                if remaining <= 0:
                    continue
                if indices.numel() > remaining:
                    order = torch.randperm(indices.numel(), device=indices.device)[:remaining]
                    indices = indices[order]
                sampled[class_index].append(
                    feature_columns[:, indices].transpose(0, 1)
                )

        result: List[torch.Tensor] = []
        for parts in sampled:
            if parts:
                result.append(torch.cat(parts, dim=0))
            else:
                result.append(features.new_zeros((0, self.feature_dim)))
        return result

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Update each class with an online spherical k-means/EMA step."""

        class_features = self._sample_class_features(
            features.detach(), targets.detach(), valid_mask.detach()
        )
        for class_index, values in enumerate(class_features):
            if values.numel() == 0:
                continue
            values = F.normalize(values.float(), dim=1)
            if not bool(self.initialised[class_index].all()):
                if values.shape[0] >= self.num_prototypes:
                    chosen = torch.linspace(
                        0, values.shape[0] - 1, self.num_prototypes,
                        device=values.device,
                    ).round().long()
                else:
                    chosen = torch.arange(
                        self.num_prototypes, device=values.device
                    ) % values.shape[0]
                self.prototypes[class_index].copy_(values[chosen])
                self.initialised[class_index].fill_(True)

            prototypes = F.normalize(self.prototypes[class_index].float(), dim=1)
            assignment = torch.matmul(values, prototypes.transpose(0, 1)).argmax(dim=1)
            for proto_index in range(self.num_prototypes):
                assigned = values[assignment == proto_index]
                if assigned.numel() == 0:
                    continue
                centre = F.normalize(assigned.mean(dim=0), dim=0)
                updated = (
                    self.momentum * prototypes[proto_index]
                    + (1.0 - self.momentum) * centre
                )
                self.prototypes[class_index, proto_index].copy_(
                    F.normalize(updated, dim=0)
                )
            self.updates[class_index] += 1

    def _class_logits(self, values: torch.Tensor) -> torch.Tensor:
        values = F.normalize(values.float(), dim=1)
        prototypes = F.normalize(self.prototypes.float(), dim=2)
        similarities = torch.einsum("nf,ckf->nck", values, prototypes)
        logits = similarities.max(dim=2).values / self.temperature
        class_ready = self.initialised.all(dim=1)
        return logits.masked_fill(~class_ready.unsqueeze(0), -1e4)

    def classification_loss(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sampled voxel-to-prototype classification loss."""

        class_features = self._sample_class_features(features, targets, valid_mask)
        values: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for class_index, class_values in enumerate(class_features):
            if class_values.numel() == 0 or not bool(self.initialised[class_index].all()):
                continue
            values.append(class_values)
            labels.append(
                torch.full(
                    (class_values.shape[0],), class_index,
                    device=class_values.device, dtype=torch.long,
                )
            )
        if not values:
            return features.sum() * 0.0
        value_tensor = torch.cat(values, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        return F.cross_entropy(self._class_logits(value_tensor), label_tensor)

    @torch.no_grad()
    def predicted_class_support(
        self,
        features: torch.Tensor,
        predicted_class: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int = 65536,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cosine support for each voxel's predicted class.

        Returns a score in ``[0, 1]`` plus a validity mask.  Computing only the
        selected predicted class avoids allocating dense ``C x K`` logits for
        every voxel in a 96^3 volume.
        """

        support = torch.zeros_like(predicted_class, dtype=torch.float32)
        valid = torch.zeros_like(mask, dtype=torch.bool)
        for batch_index in range(features.shape[0]):
            flat_class = predicted_class[batch_index].reshape(-1)
            flat_mask = mask[batch_index].reshape(-1)
            flat_support = support[batch_index].reshape(-1)
            flat_valid = valid[batch_index].reshape(-1)
            feature_columns = features[batch_index].reshape(self.feature_dim, -1)
            for class_index in range(self.num_classes):
                if not bool(self.initialised[class_index].all()):
                    continue
                indices = torch.nonzero(
                    flat_mask & (flat_class == class_index), as_tuple=False
                ).flatten()
                if indices.numel() == 0:
                    continue
                prototypes = F.normalize(
                    self.prototypes[class_index].float(), dim=1
                )
                for start in range(0, indices.numel(), chunk_size):
                    current = indices[start : start + chunk_size]
                    values = F.normalize(
                        feature_columns[:, current].transpose(0, 1).float(), dim=1
                    )
                    cosine = torch.matmul(values, prototypes.transpose(0, 1)).max(dim=1).values
                    flat_support[current] = (cosine + 1.0) * 0.5
                    flat_valid[current] = True
        return support.clamp(0.0, 1.0), valid

    def summary(self) -> Dict[str, float]:
        ready = self.initialised.all(dim=1)
        return {
            "ready_classes": float(ready.sum()),
            "mean_updates": float(self.updates[ready].float().mean()) if ready.any() else 0.0,
        }


@dataclass
class PrototypeDirectionDiagnostics:
    disagreement_fraction: float = 0.0
    prototype_coverage: float = 0.0
    prototype_changed_direction: float = 0.0
    mean_direction_weight: float = 0.0


class PrototypeDirectionalDisagreement(nn.Module):
    """CPS-v3 direction selection augmented with branch-local prototypes."""

    def __init__(
        self,
        temperature: float = 1.0,
        margin_scale: float = 0.1,
        prototype_strength: float = 0.35,
        minimum_teacher_score: float = 0.45,
    ) -> None:
        super().__init__()
        if temperature <= 0.0 or margin_scale <= 0.0:
            raise ValueError("temperature and margin_scale must be positive")
        if not 0.0 <= prototype_strength <= 1.0:
            raise ValueError("prototype_strength must lie in [0, 1]")
        self.temperature = float(temperature)
        self.margin_scale = float(margin_scale)
        self.prototype_strength = float(prototype_strength)
        self.minimum_teacher_score = float(minimum_teacher_score)
        self.last_diagnostics = PrototypeDirectionDiagnostics()

    @torch.no_grad()
    def prepare(
        self,
        logits_a: torch.Tensor,
        features_a: torch.Tensor,
        bank_a: MultiPrototypeBank,
        logits_b: torch.Tensor,
        features_b: torch.Tensor,
        bank_b: MultiPrototypeBank,
    ):
        probabilities_a = F.softmax(logits_a / self.temperature, dim=1)
        probabilities_b = F.softmax(logits_b / self.temperature, dim=1)
        pseudo_a = probabilities_a.argmax(dim=1)
        pseudo_b = probabilities_b.argmax(dim=1)
        disagreement = pseudo_a != pseudo_b
        count = disagreement.sum()

        reliability_a = prediction_reliability(probabilities_a)
        reliability_b = prediction_reliability(probabilities_b)
        support_a, support_valid_a = bank_a.predicted_class_support(
            features_a, pseudo_a, disagreement
        )
        support_b, support_valid_b = bank_b.predicted_class_support(
            features_b, pseudo_b, disagreement
        )

        score_a = reliability_a.clone()
        score_b = reliability_b.clone()
        score_a[support_valid_a] = (
            (1.0 - self.prototype_strength) * reliability_a[support_valid_a]
            + self.prototype_strength * support_a[support_valid_a]
        )
        score_b[support_valid_b] = (
            (1.0 - self.prototype_strength) * reliability_b[support_valid_b]
            + self.prototype_strength * support_b[support_valid_b]
        )

        confidence_teach_b = disagreement & (reliability_a > reliability_b)
        teach_b = disagreement & (score_a > score_b)
        teach_a = disagreement & ~teach_b
        margin = (score_a - score_b).abs()
        direction_weight = margin / (margin + self.margin_scale)

        teacher_score_a = torch.where(teach_b, score_a, score_b)
        teacher_proto_valid = torch.where(teach_b, support_valid_a, support_valid_b)
        proto_mask = disagreement & teacher_proto_valid & (
            teacher_score_a >= self.minimum_teacher_score
        )
        proto_for_a = proto_mask & teach_a
        proto_for_b = proto_mask & teach_b

        if count.item() == 0:
            self.last_diagnostics = PrototypeDirectionDiagnostics()
        else:
            any_support = disagreement & (support_valid_a | support_valid_b)
            changed = disagreement & (teach_b != confidence_teach_b)
            self.last_diagnostics = PrototypeDirectionDiagnostics(
                disagreement_fraction=float(disagreement.float().mean()),
                prototype_coverage=float(any_support.sum().float() / count.float()),
                prototype_changed_direction=float(changed.sum().float() / count.float()),
                mean_direction_weight=float(direction_weight[disagreement].mean()),
            )

        return {
            "probabilities_a": probabilities_a,
            "probabilities_b": probabilities_b,
            "pseudo_a": pseudo_a,
            "pseudo_b": pseudo_b,
            "teach_a": teach_a,
            "teach_b": teach_b,
            "proto_for_a": proto_for_a,
            "proto_for_b": proto_for_b,
            "direction_weight": direction_weight,
            "disagreement_count": count,
        }

    def _kl(self, logits, teacher_probabilities, mask, state):
        if state["disagreement_count"].item() == 0:
            return logits.sum() * 0.0
        scaled = logits / self.temperature
        per_voxel = F.kl_div(
            F.log_softmax(scaled, dim=1), teacher_probabilities,
            reduction="none",
        ).sum(dim=1) * (self.temperature ** 2)
        weighted = state["direction_weight"] * mask.to(per_voxel.dtype) * per_voxel
        return weighted.sum() / state["disagreement_count"].to(per_voxel.dtype)

    def losses_for_a(self, logits, features, bank: MultiPrototypeBank, state):
        kl = self._kl(logits, state["probabilities_b"], state["teach_a"], state)
        proto = bank.classification_loss(
            features, state["pseudo_b"], state["proto_for_a"]
        )
        return kl, proto

    def losses_for_b(self, logits, features, bank: MultiPrototypeBank, state):
        kl = self._kl(logits, state["probabilities_a"], state["teach_b"], state)
        proto = bank.classification_loss(
            features, state["pseudo_a"], state["proto_for_b"]
        )
        return kl, proto
