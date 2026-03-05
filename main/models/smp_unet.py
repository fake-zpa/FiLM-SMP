from __future__ import annotations

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None  # type: ignore[assignment]


class SMPUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        out_channels: int = 1,
        decoder_attention_type: str | None = None,
        encoder_depth: int = 5,
    ) -> None:
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation-models-pytorch is required for SMPUNet. "
                "Install it with: pip install segmentation-models-pytorch"
            )
        self.encoder_name = encoder_name
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=int(out_channels),
            activation=None,
            decoder_attention_type=decoder_attention_type,
            encoder_depth=int(encoder_depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
