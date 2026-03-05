from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .lovasz import BCELovaszLoss, LovaszHingeLoss


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    name = str(cfg.get("name", "bce_lovasz"))
    if name in {"lovasz", "lovasz_hinge"}:
        return LovaszHingeLoss(per_image=bool(cfg.get("per_image", True)))
    if name in {"bce_lovasz", "bce+lovasz"}:
        return BCELovaszLoss(
            bce_weight=float(cfg.get("bce_weight", 0.5)),
            lovasz_weight=float(cfg.get("lovasz_weight", 0.5)),
        )
    raise ValueError(f"Unknown loss name: {name}. Supported: 'bce_lovasz', 'lovasz'")


__all__ = [
    "BCELovaszLoss",
    "LovaszHingeLoss",
    "build_loss",
]
