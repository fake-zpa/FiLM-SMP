"""
FiLM-Gated SMP UNet — Uses SMP's proven encoder+decoder, adds FiLM+Gate AEF conditioning.

Key design:
  - SMP UNet encoder (pretrained ResNet34) for sensor data (S1+S2)
  - SMP UNet decoder with skip connections (proven to work at ~53-57% IoU)
  - FiLM + PriorGate inserted BETWEEN encoder and decoder to inject AEF prior
  - At init, FiLM = identity (gamma=1, beta=0), Gate trusts sensor 88%
  - This preserves SMP baseline performance while adding AEF conditioning

Architecture:
  Sensor (S1+S2, 8ch) → SMP Encoder → multi-scale features
  AEF (64ch)           → AlignAEF    → multi-scale aligned features
  At each encoder scale (5 levels):
    FiLM: (γ, β) = g(F_AEF);  F̃ = γ ⊙ F_sensor + β
    PriorGate: w = σ(q([F̃, F_AEF]));  F_out = w ⊙ F̃ + (1-w) ⊙ F_AEF
  Modified features → SMP Decoder → logits
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# FiLM + Gate blocks (correct initialization)
# ---------------------------------------------------------------------------

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation with identity initialization."""

    def __init__(self, cond_channels: int, feat_channels: int):
        super().__init__()
        self.gamma_gen = nn.Conv2d(cond_channels, feat_channels, 1)
        nn.init.zeros_(self.gamma_gen.weight)
        nn.init.ones_(self.gamma_gen.bias)   # gamma starts at 1

        self.beta_gen = nn.Conv2d(cond_channels, feat_channels, 1)
        nn.init.zeros_(self.beta_gen.weight)
        nn.init.zeros_(self.beta_gen.bias)   # beta starts at 0

    def forward(self, f_sensor: torch.Tensor, f_aef: torch.Tensor) -> torch.Tensor:
        # Align spatial dims if needed
        if f_aef.shape[2:] != f_sensor.shape[2:]:
            f_aef = F.interpolate(f_aef, size=f_sensor.shape[2:], mode="bilinear", align_corners=False)
        gamma = self.gamma_gen(f_aef)
        beta = self.beta_gen(f_aef)
        return gamma * f_sensor + beta


class PriorGate(nn.Module):
    """Interpretable gate: learns where to trust AEF prior vs sensor features."""

    def __init__(self, feat_channels: int, cond_channels: int):
        super().__init__()
        self.gate_conv = nn.Conv2d(feat_channels + cond_channels, feat_channels, 1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, 3.5)  # sigmoid(3.5) ≈ 0.97

    def forward(self, f_film: torch.Tensor, f_aef: torch.Tensor) -> torch.Tensor:
        if f_aef.shape[2:] != f_film.shape[2:]:
            f_aef = F.interpolate(f_aef, size=f_film.shape[2:], mode="bilinear", align_corners=False)
        w = torch.sigmoid(self.gate_conv(torch.cat([f_film, f_aef], dim=1)))
        return w * f_film + (1 - w) * f_aef


class AlignAEF(nn.Module):
    """Lightweight CNN to produce multi-scale AEF features matching encoder stages."""

    def __init__(self, aef_channels: int, target_channels: List[int]):
        super().__init__()
        layers = []
        in_ch = aef_channels
        for out_ch in target_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self.stages = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        h = x
        for stage in self.stages:
            h = stage(h)
            features.append(h)
        return features


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class FiLMSMPModel(nn.Module):
    """
    SMP UNet with FiLM+Gate AEF conditioning.

    Uses SMP's encoder and decoder (proven architecture) and inserts
    FiLM+PriorGate between them to condition on AEF features.
    """

    def __init__(
        self,
        s1_channels: int = 2,
        s2_channels: int = 6,
        aef_channels: int = 64,
        use_s1: bool = True,
        use_s2: bool = True,
        use_aef: bool = True,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        decoder_attention_type: Optional[str] = None,
        modality_dropout_aef: float = 0.0,
        modality_dropout_s2: float = 0.0,
        use_prior_gate: bool = False,
        s1_film: bool = False,
    ):
        super().__init__()
        self.s1_channels = s1_channels if use_s1 else 0
        self.s2_channels = s2_channels if use_s2 else 0
        self.aef_channels = aef_channels if use_aef else 0
        self.use_s1 = use_s1
        self.use_s2 = use_s2
        self.use_aef = use_aef
        self.use_prior_gate = use_prior_gate
        self.s1_film = s1_film and use_s1  # S1 via FiLM only if S1 is used
        self.modality_dropout_aef = modality_dropout_aef
        self.modality_dropout_s2 = modality_dropout_s2

        # When s1_film=True, S1 is NOT concatenated; it goes through FiLM instead
        if self.s1_film:
            sensor_ch = self.s2_channels
        else:
            sensor_ch = self.s2_channels + self.s1_channels
        assert sensor_ch > 0, "Need at least S1 or S2"

        # SMP UNet — use its encoder + decoder
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=sensor_ch,
            classes=1,
            decoder_attention_type=decoder_attention_type,
        )

        # Encoder channel sizes (skip features[0] which is input)
        enc_channels = list(self.unet.encoder.out_channels[1:])  # [64, 64, 128, 256, 512]

        # AEF conditioning branch
        if use_aef:
            self.align_aef = AlignAEF(aef_channels, enc_channels)
            self.film_blocks = nn.ModuleList([
                FiLMBlock(ch, ch) for ch in enc_channels
            ])
            if use_prior_gate:
                self.prior_gates = nn.ModuleList([
                    PriorGate(ch, ch) for ch in enc_channels
                ])

        # S1 FiLM conditioning branch (S1 injected via FiLM, like AEF)
        if self.s1_film:
            self.align_s1 = AlignAEF(self.s1_channels, enc_channels)
            self.film_blocks_s1 = nn.ModuleList([
                FiLMBlock(ch, ch) for ch in enc_channels
            ])

        # For trainer compatibility
        self.encoder_name = encoder_name
        # Expose encoder for differential learning rate
        self.encoder = self.unet.encoder

    def _split_input(self, x: torch.Tensor):
        parts = {}
        offset = 0
        sensor_parts = []
        if self.use_s2:
            parts["s2"] = x[:, offset:offset + self.s2_channels]
            sensor_parts.append(parts["s2"])
            offset += self.s2_channels
        if self.use_s1:
            parts["s1"] = x[:, offset:offset + self.s1_channels]
            if not self.s1_film:
                sensor_parts.append(parts["s1"])
            offset += self.s1_channels
        if self.use_aef:
            parts["aef"] = x[:, offset:offset + self.aef_channels]
            offset += self.aef_channels
        parts["sensor"] = torch.cat(sensor_parts, dim=1) if len(sensor_parts) > 1 else sensor_parts[0]
        return parts

    def _apply_modality_dropout(self, parts: dict) -> dict:
        if not self.training:
            return parts
        B = parts["sensor"].shape[0]
        device = parts["sensor"].device

        if self.use_aef and self.modality_dropout_aef > 0 and "aef" in parts:
            mask = (torch.rand(B, 1, 1, 1, device=device) > self.modality_dropout_aef).float()
            parts["aef"] = parts["aef"] * mask

        if self.use_s2 and self.modality_dropout_s2 > 0 and self.use_s1:
            mask = (torch.rand(B, 1, 1, 1, device=device) > self.modality_dropout_s2).float()
            s2_part = parts["sensor"][:, :self.s2_channels] * mask
            s1_part = parts["sensor"][:, self.s2_channels:]
            parts["sensor"] = torch.cat([s2_part, s1_part], dim=1)

        return parts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = self._split_input(x)
        parts = self._apply_modality_dropout(parts)

        # SMP encoder: returns [input, stage1, stage2, stage3, stage4, stage5]
        encoder_features = self.unet.encoder(parts["sensor"])

        # --- S1 FiLM conditioning (applied first if active) ---
        if self.s1_film and "s1" in parts:
            s1_features = self.align_s1(parts["s1"])
            conditioned_s1 = [encoder_features[0]]
            for i in range(len(s1_features)):
                enc_feat = encoder_features[i + 1]
                f_out = self.film_blocks_s1[i](enc_feat, s1_features[i])
                conditioned_s1.append(f_out)
            encoder_features = conditioned_s1

        # --- AEF FiLM conditioning ---
        if self.use_aef and "aef" in parts:
            # Align AEF to encoder spatial scales
            aef_features = self.align_aef(parts["aef"])

            # Apply FiLM (+ optional PriorGate) to encoder features (skip features[0] = input)
            conditioned = [encoder_features[0]]  # keep input as-is
            for i in range(len(aef_features)):
                enc_feat = encoder_features[i + 1]
                aef_feat = aef_features[i]
                f_film = self.film_blocks[i](enc_feat, aef_feat)
                if self.use_prior_gate:
                    f_out = self.prior_gates[i](f_film, aef_feat)
                else:
                    f_out = f_film
                conditioned.append(f_out)
            decoder_input = conditioned
        else:
            decoder_input = list(encoder_features)

        # SMP decoder expects List[Tensor]
        decoder_output = self.unet.decoder(decoder_input)

        # SMP segmentation head
        logits = self.unet.segmentation_head(decoder_output)
        return logits
