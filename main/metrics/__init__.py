from __future__ import annotations

from typing import Any, Dict

import torch

from .binary import binary_metrics_from_logits


def compute_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    cfg = dict(cfg or {})
    threshold = float(cfg.get("threshold", 0.5))
    return binary_metrics_from_logits(logits, target, valid, threshold=threshold)


__all__ = ["compute_metrics_from_logits"]
