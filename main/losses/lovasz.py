"""Lovász-Softmax / Lovász-Hinge loss for binary segmentation.
Directly optimizes IoU (Jaccard index).
Reference: Berman et al., "The Lovász-Softmax loss: A tractable surrogate for
the optimization of the intersection-over-union measure in neural networks", CVPR 2018.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if len(logits) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """Binary Lovász hinge loss operating on logits."""

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid is None:
            valid = torch.ones_like(target)
        if self.per_image:
            losses = []
            for i in range(logits.shape[0]):
                mask = valid[i].view(-1) > 0
                lg = logits[i].view(-1)[mask]
                tg = target[i].view(-1)[mask]
                if lg.numel() > 0:
                    losses.append(_lovasz_hinge_flat(lg, (tg > 0.5).float()))
            if not losses:
                return logits.sum() * 0.0
            return torch.stack(losses).mean()
        else:
            mask = valid.view(-1) > 0
            lg = logits.view(-1)[mask]
            tg = target.view(-1)[mask]
            return _lovasz_hinge_flat(lg, (tg > 0.5).float())


class BCELovaszLoss(nn.Module):
    """Combined BCE + Lovász loss for stable training."""

    def __init__(self, bce_weight: float = 0.5, lovasz_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.lovasz = LovaszHingeLoss(per_image=True)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid is None:
            valid = torch.ones_like(target)
        mask = valid > 0
        bce = F.binary_cross_entropy_with_logits(
            logits[mask], target[mask], reduction="mean"
        )
        lovasz = self.lovasz(logits, target, valid)
        return self.bce_weight * bce + self.lovasz_weight * lovasz
