from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def binary_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    probs = torch.sigmoid(logits)
    pred = probs > float(threshold)
    tgt = target > 0.5
    v = valid > 0

    dims = (1, 2, 3)
    tp = (pred & tgt & v).sum(dims).float()
    fp = (pred & (~tgt) & v).sum(dims).float()
    fn = ((~pred) & tgt & v).sum(dims).float()
    tn = ((~pred) & (~tgt) & v).sum(dims).float()

    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    n_valid = v.sum(dims).float()

    oa = (tp + tn) / (n_valid + eps)

    beta2 = 4.0  # F2-score: beta=2
    f2 = ((1.0 + beta2) * tp) / ((1.0 + beta2) * tp + beta2 * fn + fp + eps)

    mcc_num = tp * tn - fp * fn
    mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc = mcc_num / mcc_den

    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n_valid * n_valid + eps)
    kappa = (oa - pe) / (1.0 - pe + eps)

    iou_bg = tn / (tn + fp + fn + eps)
    miou = (iou + iou_bg) / 2.0

    return {
        "iou": iou,
        "iou_bg": iou_bg,
        "miou": miou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "oa": oa,
        "f2": f2,
        "mcc": mcc,
        "kappa": kappa,
        "n_valid": n_valid,
    }


def _auc_scores_for_sample(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    if y_true.size == 0:
        return 0.0, 0.5
    has_pos = bool((y_true > 0).any())
    has_neg = bool((y_true == 0).any())
    if not (has_pos and has_neg):
        auprc = 1.0 if has_pos else 0.0
        return auprc, 0.5

    from sklearn.metrics import average_precision_score, roc_auc_score

    auprc = float(average_precision_score(y_true, y_score))
    auroc = float(roc_auc_score(y_true, y_score))
    return auprc, auroc


def auc_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    probs = torch.sigmoid(logits.detach())
    tgt = target.detach()
    v = valid.detach()

    probs_cpu = probs.cpu()
    tgt_cpu = tgt.cpu()
    v_cpu = v.cpu()

    auprc_vals = []
    auroc_vals = []
    for i in range(probs_cpu.shape[0]):
        mask = (v_cpu[i].view(-1) > 0).numpy()
        if mask.sum() == 0:
            auprc, auroc = 0.0, 0.5
        else:
            y_true = (tgt_cpu[i].view(-1).numpy()[mask] > 0.5).astype(np.uint8)
            y_score = probs_cpu[i].view(-1).numpy()[mask].astype(np.float32)
            auprc, auroc = _auc_scores_for_sample(y_true, y_score)
        auprc_vals.append(auprc)
        auroc_vals.append(auroc)

    device = logits.device
    return {
        "auprc": torch.tensor(auprc_vals, device=device, dtype=torch.float32),
        "auroc": torch.tensor(auroc_vals, device=device, dtype=torch.float32),
    }
