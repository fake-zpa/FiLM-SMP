from __future__ import annotations

from typing import Any, Dict

from ..data.io import S1_NUM_BANDS, S2_NUM_BANDS
from .film_smp_model import FiLMSMPModel
from .smp_unet import SMPUNet


def _infer_in_channels(use_s2: bool, use_aef: bool, use_s1: bool = False, aef_num_channels: int = 64, s1_num_channels: int = S1_NUM_BANDS) -> int:
    in_ch = 0
    if use_s2:
        in_ch += S2_NUM_BANDS
    if use_s1:
        in_ch += s1_num_channels
    if use_aef:
        in_ch += aef_num_channels
    if in_ch == 0:
        raise ValueError("At least one of use_s2/use_s1/use_aef must be true")
    return in_ch


def build_model(model_cfg: Dict[str, Any], use_s2: bool = True, use_aef: bool = True, use_s1: bool = False, aef_num_channels: int = 64, s1_num_channels: int = S1_NUM_BANDS):
    cfg = dict(model_cfg or {})
    name = str(cfg.get("name", "film_smp")).lower()

    if name == "smp_unet":
        in_ch = _infer_in_channels(use_s2=use_s2, use_aef=use_aef, use_s1=use_s1, aef_num_channels=aef_num_channels, s1_num_channels=s1_num_channels)
        encoder_name = str(cfg.get("encoder_name", "resnet34"))
        encoder_weights = cfg.get("encoder_weights", "imagenet")
        if encoder_weights is not None:
            encoder_weights = str(encoder_weights)
        out_channels = int(cfg.get("out_channels", 1))
        decoder_attention_type = cfg.get("decoder_attention_type", None)
        if decoder_attention_type is not None:
            decoder_attention_type = str(decoder_attention_type)
        encoder_depth = int(cfg.get("encoder_depth", 5))
        return SMPUNet(
            in_channels=in_ch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            out_channels=out_channels,
            decoder_attention_type=decoder_attention_type,
            encoder_depth=encoder_depth,
        )

    if name == "film_smp":
        encoder_name = str(cfg.get("encoder_name", "resnet34"))
        encoder_weights = cfg.get("encoder_weights", "imagenet")
        if encoder_weights is not None:
            encoder_weights = str(encoder_weights)
        decoder_attention_type = cfg.get("decoder_attention_type", None)
        if decoder_attention_type is not None:
            decoder_attention_type = str(decoder_attention_type)
        modality_dropout_aef = float(cfg.get("modality_dropout_aef", 0.0))
        modality_dropout_s2 = float(cfg.get("modality_dropout_s2", 0.0))
        use_prior_gate = bool(cfg.get("use_prior_gate", False))
        s1_film = bool(cfg.get("s1_film", False))
        return FiLMSMPModel(
            s1_channels=s1_num_channels,
            s2_channels=S2_NUM_BANDS,
            aef_channels=aef_num_channels,
            use_s1=use_s1,
            use_s2=use_s2,
            use_aef=use_aef,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            modality_dropout_aef=modality_dropout_aef,
            modality_dropout_s2=modality_dropout_s2,
            use_prior_gate=use_prior_gate,
            s1_film=s1_film,
        )

    raise ValueError(f"Unknown model name: {name}. Supported: 'film_smp', 'smp_unet'")
