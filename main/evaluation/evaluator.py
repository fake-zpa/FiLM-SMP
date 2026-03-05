from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lbb2seg.data.collate import pad_collate
from lbb2seg.data.dataset import FloodSegDataset
from lbb2seg.data.manifest import build_manifest, filter_manifest
from lbb2seg.data.normalization import load_stats
from lbb2seg.data.splits import load_splits
from lbb2seg.metrics import compute_metrics_from_logits
from lbb2seg.models import build_model
from lbb2seg.utils.config import load_config


def _resolve_device(device_str: str) -> torch.device:
    if str(device_str).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_input(batch: Dict[str, Any], use_s2: bool, use_aef: bool, use_s1: bool = False) -> torch.Tensor:
    parts = []
    if use_s2:
        parts.append(batch["s2"])
    if use_s1:
        parts.append(batch["s1"])
    if use_aef:
        parts.append(batch["aef"])
    if not parts:
        raise ValueError("No input modalities enabled")
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def _tta_predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Test-Time Augmentation: average logits over 8 geometric transforms.

    Transforms: identity, hflip, vflip, hflip+vflip, rot90, rot180, rot270,
    rot90+hflip. All are lossless and invertible.
    """
    logits_sum = model(x)
    # hflip
    xf = torch.flip(x, [-1])
    logits_sum = logits_sum + torch.flip(model(xf), [-1])
    # vflip
    xf = torch.flip(x, [-2])
    logits_sum = logits_sum + torch.flip(model(xf), [-2])
    # hflip + vflip
    xf = torch.flip(x, [-1, -2])
    logits_sum = logits_sum + torch.flip(model(xf), [-1, -2])
    # rot90
    xr = torch.rot90(x, 1, [-2, -1])
    logits_sum = logits_sum + torch.rot90(model(xr), -1, [-2, -1])
    # rot180
    xr = torch.rot90(x, 2, [-2, -1])
    logits_sum = logits_sum + torch.rot90(model(xr), -2, [-2, -1])
    # rot270
    xr = torch.rot90(x, 3, [-2, -1])
    logits_sum = logits_sum + torch.rot90(model(xr), -3, [-2, -1])
    # rot90 + hflip
    xr = torch.flip(torch.rot90(x, 1, [-2, -1]), [-1])
    logits_sum = logits_sum + torch.rot90(torch.flip(model(xr), [-1]), -1, [-2, -1])
    return logits_sum / 8.0


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def eval_run_dir(
    run_dir: Path,
    split: str = "test",
    threshold: Optional[float] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    metrics_cfg_override: Optional[Dict[str, Any]] = None,
    tta: bool = False,
) -> None:
    cfg = load_config(run_dir / "config.yaml")

    data_cfg = dict(cfg.get("data", {}))
    data_root = Path(data_cfg.get("root", "data"))
    aef_year = data_cfg.get("aef_year", None)

    use_s2 = bool(data_cfg.get("use_s2", True))
    use_s1 = bool(data_cfg.get("use_s1", False))
    use_aef = bool(data_cfg.get("use_aef", True))
    aef_select_channels = data_cfg.get("aef_select_channels", None)
    if aef_select_channels is not None:
        aef_select_channels = [int(c) for c in aef_select_channels]
    aef_num_channels = len(aef_select_channels) if aef_select_channels is not None else 64
    require_s2 = bool(data_cfg.get("require_s2", use_s2))
    require_s1 = bool(data_cfg.get("require_s1", use_s1))
    require_aef = bool(data_cfg.get("require_aef", use_aef))
    s1_add_ratio = bool(data_cfg.get("s1_add_ratio", False))

    manifest = build_manifest(data_root, aef_year=str(aef_year) if aef_year is not None else None)
    manifest = filter_manifest(manifest, require_s2=require_s2, require_aef=require_aef, require_label=True, require_s1=require_s1)

    splits = load_splits(run_dir / "splits.json")
    ids = list(splits[split])

    s2_stats = None
    s1_stats = None
    aef_stats = None
    if use_s2 and (run_dir / "stats_s2.json").exists():
        s2_stats = load_stats(run_dir / "stats_s2.json")
    if use_s1 and (run_dir / "stats_s1.json").exists():
        s1_stats = load_stats(run_dir / "stats_s1.json")
    if use_aef and (run_dir / "stats_aef.json").exists():
        aef_stats = load_stats(run_dir / "stats_aef.json")

    norm_cfg = dict(cfg.get("normalization", {}))
    s2_norm_cfg = dict(norm_cfg.get("s2", norm_cfg.get("s2_norm", {})) or {})
    s1_norm_cfg = dict(norm_cfg.get("s1", norm_cfg.get("s1_norm", {})) or {})
    aef_norm_cfg = dict(norm_cfg.get("aef", norm_cfg.get("aef_norm", {})) or {})

    patch_size = data_cfg.get("patch_size", 512)
    patch_size = None if patch_size is None else int(patch_size)
    s2_scale = float(data_cfg.get("s2_scale", 10000.0))

    ds = FloodSegDataset(
        manifest=manifest,
        split_ids=ids,
        mode=str(split),
        patch_size=patch_size,
        s2_scale=s2_scale,
        s2_stats=s2_stats,
        s1_stats=s1_stats,
        aef_stats=aef_stats,
        s2_norm_cfg=s2_norm_cfg,
        s1_norm_cfg=s1_norm_cfg,
        aef_norm_cfg=aef_norm_cfg,
        use_s2=use_s2,
        use_s1=use_s1,
        use_aef=use_aef,
        s1_add_ratio=s1_add_ratio,
        aef_select_channels=aef_select_channels,
        seed=int(cfg.get("seed", 0)),
    )

    dl_cfg = dict(cfg.get("dataloader", {}))
    cfg_batch_size = int(dl_cfg.get("batch_size", 4))
    if batch_size is None:
        batch_size = cfg_batch_size
    else:
        batch_size = int(batch_size)
    cfg_num_workers = int(dl_cfg.get("num_workers", 4))
    if num_workers is None:
        num_workers = cfg_num_workers
    else:
        num_workers = int(num_workers)
    prefetch_factor = dl_cfg.get("prefetch_factor", None)
    persistent_workers = bool(dl_cfg.get("persistent_workers", num_workers > 0))
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    elif prefetch_factor is None:
        prefetch_factor = 2

    train_cfg = dict(cfg.get("train", {}))
    device = _resolve_device(str(train_cfg.get("device", "cuda")))
    pin_memory = device.type == "cuda"

    loader_kwargs: Dict[str, Any] = {}
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader_kwargs["persistent_workers"] = persistent_workers

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate,
        drop_last=False,
        **loader_kwargs,
    )

    model_cfg = dict(cfg.get("model", {}))
    model_name = str(model_cfg.get("name", "unet"))

    metrics_cfg = cfg.get("metrics", {})
    if not isinstance(metrics_cfg, dict):
        metrics_cfg = {}
    metrics_cfg = dict(metrics_cfg)
    metrics_cfg.setdefault("threshold", float(train_cfg.get("threshold", 0.5)))
    if threshold is not None:
        metrics_cfg["threshold"] = float(threshold)
    if metrics_cfg_override:
        metrics_cfg.update(metrics_cfg_override)

    rows: List[dict] = []
    metrics_sum: Dict[str, float] = {}
    n = 0

    s1_num_channels = 3 if s1_add_ratio else 2

    model = build_model(model_cfg, use_s2=use_s2, use_aef=use_aef, use_s1=use_s1, aef_num_channels=aef_num_channels, s1_num_channels=s1_num_channels)

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    for batch in tqdm(loader, desc=f"eval {split}"):
        batch = _to_device(batch, device)
        x = _make_input(batch, use_s2=use_s2, use_aef=use_aef, use_s1=use_s1)
        if tta:
            logits = _tta_predict(model, x)
        else:
            logits = model(x)
        m = compute_metrics_from_logits(logits, batch["y"], batch["valid"], cfg=metrics_cfg)

        bsz = int(logits.shape[0])
        for k, v in m.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v.sum().item())
        n += bsz

        for i in range(bsz):
            row = {
                "sample_id": batch["sample_id"][i],
                "country": batch["country"][i],
            }
            for k, v in m.items():
                row[k] = float(v[i].item())
            rows.append(row)

    if n == 0:
        metrics = {k: float("nan") for k in metrics_sum}
    else:
        metrics = {k: v / float(n) for k, v in metrics_sum.items()}

    out_dir = run_dir / "eval" / str(split)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(out_dir / "per_sample_metrics.csv", index=False)
    try:
        per_sample.to_parquet(out_dir / "per_sample_metrics.parquet", index=False)
    except Exception:
        pass


@torch.no_grad()
def search_best_threshold(
    run_dir: Path,
    split: str = "val",
    thresholds: Optional[Sequence[float]] = None,
    metric: str = "dice",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Tuple[float, pd.DataFrame]:
    cfg = load_config(run_dir / "config.yaml")

    data_cfg = dict(cfg.get("data", {}))
    data_root = Path(data_cfg.get("root", "data"))
    aef_year = data_cfg.get("aef_year", None)

    use_s2 = bool(data_cfg.get("use_s2", True))
    use_s1 = bool(data_cfg.get("use_s1", False))
    use_aef = bool(data_cfg.get("use_aef", True))
    aef_select_channels = data_cfg.get("aef_select_channels", None)
    if aef_select_channels is not None:
        aef_select_channels = [int(c) for c in aef_select_channels]
    aef_num_channels = len(aef_select_channels) if aef_select_channels is not None else 64
    require_s2 = bool(data_cfg.get("require_s2", use_s2))
    require_s1 = bool(data_cfg.get("require_s1", use_s1))
    require_aef = bool(data_cfg.get("require_aef", use_aef))
    s1_add_ratio = bool(data_cfg.get("s1_add_ratio", False))

    manifest = build_manifest(data_root, aef_year=str(aef_year) if aef_year is not None else None)
    manifest = filter_manifest(manifest, require_s2=require_s2, require_aef=require_aef, require_label=True, require_s1=require_s1)

    splits = load_splits(run_dir / "splits.json")
    ids = list(splits[split])
    norm_cfg = dict(cfg.get("normalization", {}))
    s2_norm_cfg = dict(norm_cfg.get("s2", norm_cfg.get("s2_norm", {})) or {})
    s1_norm_cfg = dict(norm_cfg.get("s1", norm_cfg.get("s1_norm", {})) or {})
    aef_norm_cfg = dict(norm_cfg.get("aef", norm_cfg.get("aef_norm", {})) or {})
    s2_stats = None
    s1_stats = None
    aef_stats = None
    if use_s2 and (run_dir / "stats_s2.json").exists():
        s2_stats = load_stats(run_dir / "stats_s2.json")
    if use_s1 and (run_dir / "stats_s1.json").exists():
        s1_stats = load_stats(run_dir / "stats_s1.json")
    if use_aef and (run_dir / "stats_aef.json").exists():
        aef_stats = load_stats(run_dir / "stats_aef.json")

    patch_size = data_cfg.get("patch_size", 512)
    patch_size = None if patch_size is None else int(patch_size)
    s2_scale = float(data_cfg.get("s2_scale", 10000.0))

    ds = FloodSegDataset(
        manifest=manifest,
        split_ids=ids,
        mode=str(split),
        patch_size=patch_size,
        s2_scale=s2_scale,
        s2_stats=s2_stats,
        s1_stats=s1_stats,
        aef_stats=aef_stats,
        s2_norm_cfg=s2_norm_cfg,
        s1_norm_cfg=s1_norm_cfg,
        aef_norm_cfg=aef_norm_cfg,
        use_s2=use_s2,
        use_s1=use_s1,
        use_aef=use_aef,
        s1_add_ratio=s1_add_ratio,
        aef_select_channels=aef_select_channels,
        seed=int(cfg.get("seed", 0)),
    )

    dl_cfg = dict(cfg.get("dataloader", {}))
    cfg_batch_size = int(dl_cfg.get("batch_size", 4))
    if batch_size is None:
        batch_size = cfg_batch_size
    else:
        batch_size = int(batch_size)
    cfg_num_workers = int(dl_cfg.get("num_workers", 4))
    if num_workers is None:
        num_workers = cfg_num_workers
    else:
        num_workers = int(num_workers)
    prefetch_factor = dl_cfg.get("prefetch_factor", None)
    persistent_workers = bool(dl_cfg.get("persistent_workers", num_workers > 0))
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    elif prefetch_factor is None:
        prefetch_factor = 2

    train_cfg = dict(cfg.get("train", {}))
    device = _resolve_device(str(train_cfg.get("device", "cuda")))
    pin_memory = device.type == "cuda"

    loader_kwargs: Dict[str, Any] = {}
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader_kwargs["persistent_workers"] = persistent_workers

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate,
        drop_last=False,
        **loader_kwargs,
    )

    model_cfg = dict(cfg.get("model", {}))
    model_name = str(model_cfg.get("name", "unet"))

    metrics_cfg = cfg.get("metrics", {})
    if not isinstance(metrics_cfg, dict):
        metrics_cfg = {}
    metrics_cfg = dict(metrics_cfg)

    if thresholds is None:
        thresholds = [float(x) for x in np.arange(0.05, 1.0, 0.05)]
    else:
        thresholds = [float(x) for x in thresholds]
    if not thresholds:
        raise ValueError("No thresholds provided for search")

    thresholds = sorted(set(thresholds))
    metrics_cfgs = {th: dict(metrics_cfg, threshold=float(th)) for th in thresholds}
    metrics_sum: Dict[float, Dict[str, float]] = {th: {} for th in thresholds}
    n = 0

    s1_num_channels = 3 if s1_add_ratio else 2
    model = build_model(model_cfg, use_s2=use_s2, use_aef=use_aef, use_s1=use_s1, aef_num_channels=aef_num_channels, s1_num_channels=s1_num_channels)

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    for batch in tqdm(loader, desc=f"threshold search {split}"):
        batch = _to_device(batch, device)
        x = _make_input(batch, use_s2=use_s2, use_aef=use_aef, use_s1=use_s1)
        logits = model(x)
        for th in thresholds:
            m = compute_metrics_from_logits(logits, batch["y"], batch["valid"], cfg=metrics_cfgs[th])
            for k, v in m.items():
                metrics_sum[th][k] = metrics_sum[th].get(k, 0.0) + float(v.sum().item())
        n += int(logits.shape[0])

    if n == 0:
        raise ValueError("No samples available for threshold search")

    rows: List[dict] = []
    for th in thresholds:
        row = {"threshold": float(th)}
        for k, v in metrics_sum[th].items():
            row[k] = v / float(n)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("threshold")
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in threshold search results")

    best_idx = int(df[metric].idxmax())
    best_threshold = float(df.loc[best_idx, "threshold"])
    best_score = float(df.loc[best_idx, metric])

    out_dir = run_dir / "eval" / str(split)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "threshold_search.csv", index=False)
    with (out_dir / "best_threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"metric": metric, "threshold": best_threshold, "score": best_score}, f, indent=2)

    return best_threshold, df
