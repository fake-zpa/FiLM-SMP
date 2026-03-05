from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from lbb2seg.data.collate import pad_collate
from lbb2seg.data.dataset import AugmentCfg, FloodSegDataset, ModalDropoutCfg
from lbb2seg.data.io import read_aef, read_label, read_s2
from lbb2seg.data.manifest import build_manifest, filter_manifest
from lbb2seg.data.normalization import ChannelStats, compute_channel_stats, load_stats, save_stats
from lbb2seg.data.splits import iid_split, leave_one_country_split, load_splits, save_splits
from lbb2seg.losses import build_loss
from lbb2seg.metrics import compute_metrics_from_logits
from lbb2seg.models import build_model
from lbb2seg.utils.config import save_config
from lbb2seg.utils.seed import seed_everything


def _resolve_device(device_str: str) -> torch.device:
    if str(device_str).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_run_dir(results_root: Path, run_name: str) -> Path:
    run_dir = results_root / run_name
    if run_dir.exists():
        raise FileExistsError(str(run_dir))
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    return run_dir


def _infer_in_channels(use_s2: bool, use_aef: bool, use_s1: bool = False, aef_num_channels: int = 64) -> int:
    from ..data.io import S1_NUM_BANDS, S2_NUM_BANDS
    in_ch = 0
    if use_s2:
        in_ch += S2_NUM_BANDS
    if use_s1:
        in_ch += S1_NUM_BANDS
    if use_aef:
        in_ch += aef_num_channels
    if in_ch == 0:
        raise ValueError("At least one of use_s2/use_s1/use_aef must be true")
    return in_ch


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


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def _eval_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_s2: bool,
    use_aef: bool,
    metrics_cfg: Dict[str, Any],
    use_s1: bool = False,
) -> Dict[str, float]:
    model.eval()
    metrics_sum: Dict[str, float] = {}
    n = 0
    threshold = float(metrics_cfg.get("threshold", 0.5))
    g_tp = g_fp = g_fn = g_tn = 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        x = _make_input(batch, use_s2=use_s2, use_aef=use_aef, use_s1=use_s1)
        logits = model(x)
        m = compute_metrics_from_logits(logits, batch["y"], batch["valid"], cfg=metrics_cfg)
        bsz = int(next(iter(m.values())).shape[0]) if m else int(logits.shape[0])
        for k, v in m.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v.sum().item())
        n += bsz
        pred = torch.sigmoid(logits) > threshold
        tgt = batch["y"] > 0.5
        v = batch["valid"] > 0
        g_tp += float((pred & tgt & v).sum().item())
        g_fp += float((pred & (~tgt) & v).sum().item())
        g_fn += float((~pred & tgt & v).sum().item())
        g_tn += float((~pred & (~tgt) & v).sum().item())

    if n == 0:
        return {k: float("nan") for k in metrics_sum}
    result = {k: v / float(n) for k, v in metrics_sum.items()}
    eps = 1e-6
    result["global_iou"] = g_tp / (g_tp + g_fp + g_fn + eps)
    result["global_iou_bg"] = g_tn / (g_tn + g_fp + g_fn + eps)
    result["global_miou"] = (result["global_iou"] + result["global_iou_bg"]) / 2.0
    result["global_dice"] = (2.0 * g_tp) / (2.0 * g_tp + g_fp + g_fn + eps)
    result["global_precision"] = g_tp / (g_tp + g_fp + eps)
    result["global_recall"] = g_tp / (g_tp + g_fn + eps)
    return result


def _compute_stats(
    manifest: pd.DataFrame,
    train_ids: list[str],
    use_s2: bool,
    use_aef: bool,
    s2_scale: float,
    stats_max_chips: int,
    max_pixels: int,
    seed: int,
    use_s1: bool = False,
    **kwargs: Any,
) -> Tuple[Optional[ChannelStats], Optional[ChannelStats], Optional[ChannelStats]]:
    rng = np.random.RandomState(int(seed))
    ids = list(train_ids)
    rng.shuffle(ids)
    ids = ids[: int(stats_max_chips)]

    m = manifest.set_index("sample_id")

    s2_stats = None
    if use_s2:
        arrays = []
        for sid in ids:
            row = m.loc[sid]
            arrays.append(read_s2(Path(row["s2_path"]), scale=float(s2_scale)))
        s2_stats = compute_channel_stats(tuple(arrays), max_pixels=int(max_pixels), seed=int(seed))

    s1_stats = None
    if use_s1:
        from ..data.io import read_s1
        s1_add_ratio = bool(kwargs.get("s1_add_ratio", False))
        arrays = []
        for sid in ids:
            row = m.loc[sid]
            arrays.append(read_s1(Path(row["s1_path"]), add_ratio=s1_add_ratio))
        s1_stats = compute_channel_stats(tuple(arrays), max_pixels=int(max_pixels), seed=int(seed))

    aef_stats = None
    if use_aef:
        aef_select = kwargs.get("aef_select_channels", None)
        arrays = []
        for sid in ids:
            row = m.loc[sid]
            a = read_aef(Path(row["aef_path"]))
            if aef_select is not None:
                a = a[list(aef_select)]
            arrays.append(a)
        aef_stats = compute_channel_stats(tuple(arrays), max_pixels=int(max_pixels), seed=int(seed))

    return s2_stats, s1_stats, aef_stats


def train_from_config(cfg: Dict[str, Any]) -> Path:
    run_name = str(cfg.get("run_name") or cfg.get("name") or time.strftime("run_%Y%m%d_%H%M%S"))
    results_root = Path(cfg.get("results_root", "results"))
    run_dir = _ensure_run_dir(results_root, run_name)

    seed = int(cfg.get("seed", 0))
    deterministic = bool(cfg.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

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

    manifest = build_manifest(data_root, aef_year=str(aef_year) if aef_year is not None else None)
    manifest = filter_manifest(manifest, require_s2=require_s2, require_aef=require_aef, require_label=True, require_s1=require_s1)

    split_cfg = dict(cfg.get("split", {}))
    split_path = split_cfg.get("path", None)
    if split_path is not None:
        splits = load_splits(Path(split_path))
    else:
        split_type = str(split_cfg.get("type", "iid"))
        split_seed = int(split_cfg.get("seed", seed))
        if split_type == "iid":
            ratios = split_cfg.get("ratios", (0.7, 0.1, 0.2))
            ratios_t = (float(ratios[0]), float(ratios[1]), float(ratios[2]))
            splits = iid_split(manifest, seed=split_seed, ratios=ratios_t)
        elif split_type in {"leave_one_country", "ood_country"}:
            holdout_country = str(split_cfg["holdout_country"])
            val_ratio = float(split_cfg.get("val_ratio", 0.1))
            splits = leave_one_country_split(manifest, holdout_country=holdout_country, seed=split_seed, val_ratio=val_ratio)
        else:
            raise ValueError(f"Unknown split type: {split_type}")

    save_splits(splits, run_dir / "splits.json")

    norm_cfg = dict(cfg.get("normalization", {}))
    compute_stats_flag = bool(norm_cfg.get("compute_stats", True))
    stats_max_chips = int(norm_cfg.get("stats_max_chips", 20))
    max_pixels = int(norm_cfg.get("max_pixels", 200_000))
    s2_norm_cfg = dict(norm_cfg.get("s2", norm_cfg.get("s2_norm", {})) or {})
    s1_norm_cfg = dict(norm_cfg.get("s1", norm_cfg.get("s1_norm", {})) or {})
    aef_norm_cfg = dict(norm_cfg.get("aef", norm_cfg.get("aef_norm", {})) or {})

    s2_scale = float(data_cfg.get("s2_scale", 10000.0))
    s1_add_ratio = bool(data_cfg.get("s1_add_ratio", False))

    s2_stats = None
    s1_stats = None
    aef_stats = None
    if compute_stats_flag:
        s2_stats, s1_stats, aef_stats = _compute_stats(
            manifest,
            splits["train"],
            use_s2=use_s2,
            use_aef=use_aef,
            s2_scale=s2_scale,
            stats_max_chips=stats_max_chips,
            max_pixels=max_pixels,
            seed=seed,
            use_s1=use_s1,
            aef_select_channels=aef_select_channels,
            s1_add_ratio=s1_add_ratio,
        )
        if s2_stats is not None:
            save_stats(s2_stats, run_dir / "stats_s2.json")
        if s1_stats is not None:
            save_stats(s1_stats, run_dir / "stats_s1.json")
        if aef_stats is not None:
            save_stats(aef_stats, run_dir / "stats_aef.json")
    else:
        if use_s2 and (run_dir / "stats_s2.json").exists():
            s2_stats = load_stats(run_dir / "stats_s2.json")
        if use_s1 and (run_dir / "stats_s1.json").exists():
            s1_stats = load_stats(run_dir / "stats_s1.json")
        if use_aef and (run_dir / "stats_aef.json").exists():
            aef_stats = load_stats(run_dir / "stats_aef.json")

    patch_size = data_cfg.get("patch_size", 512)
    patch_size = None if patch_size is None else int(patch_size)

    modal_dropout_cfg = dict(cfg.get("modal_dropout", {}))
    modal_dropout = ModalDropoutCfg(
        enabled=bool(modal_dropout_cfg.get("enabled", False)),
        p_drop_s2=float(modal_dropout_cfg.get("p_drop_s2", 0.0)),
        p_drop_aef=float(modal_dropout_cfg.get("p_drop_aef", 0.0)),
    )

    augment_cfg = dict(cfg.get("augment", {}))
    augment = AugmentCfg(
        enabled=bool(augment_cfg.get("enabled", False)),
        hflip=bool(augment_cfg.get("hflip", True)),
        vflip=bool(augment_cfg.get("vflip", True)),
        rot90=bool(augment_cfg.get("rot90", True)),
        gaussian_noise=float(augment_cfg.get("gaussian_noise", 0.0)),
        brightness=float(augment_cfg.get("brightness", 0.0)),
        contrast=float(augment_cfg.get("contrast", 0.0)),
    )

    pseudo_aef = cfg.get("pseudo_aef", None)
    if pseudo_aef is not None:
        pseudo_aef = str(pseudo_aef)

    cache_in_memory = bool(data_cfg.get("cache_in_memory", False))

    tile_size = data_cfg.get("tile_size", None)
    if tile_size is not None:
        tile_size = int(tile_size)
    tile_overlap = int(data_cfg.get("tile_overlap", 0))
    oversample_water = float(data_cfg.get("oversample_water", 0.0))

    train_ds = FloodSegDataset(
        manifest=manifest,
        split_ids=splits["train"],
        mode="train",
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
        modal_dropout=modal_dropout,
        augment=augment,
        pseudo_aef=pseudo_aef,
        seed=seed,
        cache_in_memory=cache_in_memory,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        oversample_water=oversample_water,
    )
    val_ds = FloodSegDataset(
        manifest=manifest,
        split_ids=splits["val"],
        mode="val",
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
        seed=seed,
        cache_in_memory=cache_in_memory,
    )

    dl_cfg = dict(cfg.get("dataloader", {}))
    batch_size = int(dl_cfg.get("batch_size", 4))
    num_workers = int(dl_cfg.get("num_workers", 4))
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
    allow_tf32 = bool(train_cfg.get("allow_tf32", False))
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            if allow_tf32:
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    loader_kwargs: Dict[str, Any] = {}
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate,
        drop_last=False,
        **loader_kwargs,
    )

    model_cfg = dict(cfg.get("model", {}))

    metrics_cfg = cfg.get("metrics", {})
    if not isinstance(metrics_cfg, dict):
        metrics_cfg = {}
    metrics_cfg = dict(metrics_cfg)
    metrics_cfg.setdefault("threshold", float(train_cfg.get("threshold", 0.5)))

    cfg_to_save = dict(cfg)
    cfg_to_save["run_name"] = run_name
    cfg_to_save.setdefault("data", {})
    cfg_to_save["data"]["use_s2"] = use_s2
    cfg_to_save["data"]["use_s1"] = use_s1
    cfg_to_save["data"]["use_aef"] = use_aef
    save_config(cfg_to_save, run_dir / "config.yaml")

    s1_num_channels = 3 if s1_add_ratio else 2
    model = build_model(model_cfg, use_s2=use_s2, use_aef=use_aef, use_s1=use_s1, aef_num_channels=aef_num_channels, s1_num_channels=s1_num_channels)
    model.to(device)

    loss_cfg = dict(cfg.get("loss", {}))
    loss_fn = build_loss(loss_cfg).to(device)

    epochs = int(train_cfg.get("epochs", 10))

    optim_cfg = dict(cfg.get("optim", {}))
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    encoder_lr_scale = float(optim_cfg.get("encoder_lr_scale", 1.0))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer: Optional[torch.optim.Optimizer]
    if len(params) == 0:
        optimizer = None
        if epochs > 0:
            raise ValueError("Model has no trainable parameters; set train.epochs=0")
    else:
        if encoder_lr_scale < 1.0 and hasattr(model, "encoder_name"):
            encoder_params = []
            decoder_params = []
            pretrained_prefixes = ("model.encoder", "encoder_s1.", "encoder_s2.", "encoder.")
            for name_p, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(name_p.startswith(pfx) for pfx in pretrained_prefixes):
                    encoder_params.append(p)
                else:
                    decoder_params.append(p)
            param_groups = [
                {"params": encoder_params, "lr": lr * encoder_lr_scale},
                {"params": decoder_params, "lr": lr},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    sched_cfg = dict(cfg.get("scheduler", {}))
    sched_name = str(sched_cfg.get("name", "none")).lower()
    scheduler = None
    if sched_name == "cosine" and optimizer is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(sched_cfg.get("eta_min", 1e-6))
        )
    elif sched_name not in {"none", ""}:
        raise ValueError(f"Unknown scheduler name: {sched_name}")

    save_every = int(train_cfg.get("save_every", 1))
    early_cfg = dict(train_cfg.get("early_stop", {}))
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 10))
    early_metric = str(early_cfg.get("metric", "dice"))
    early_mode = str(early_cfg.get("mode", "max")).lower()
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    if early_mode not in {"max", "min"}:
        raise ValueError(f"early_stop.mode must be 'max' or 'min' (got {early_mode})")
    best_monitor = -float("inf") if early_mode == "max" else float("inf")
    no_improve = 0

    if epochs <= 0:
        ckpt = {
            "epoch": 0,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict(),
            "cfg": cfg_to_save,
        }
        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")
        torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
        return run_dir

    best_dice = -1.0
    log_rows = []

    for epoch in range(1, epochs + 1):
        set_epoch = getattr(loss_fn, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train e{epoch}/{epochs}"):
            batch = _to_device(batch, device)
            x = _make_input(batch, use_s2=use_s2, use_aef=use_aef, use_s1=use_s1)
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, batch["y"], batch["valid"])

            scaler.scale(loss).backward()
            assert optimizer is not None
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics = _eval_loader(
            model,
            val_loader,
            device=device,
            use_s2=use_s2,
            use_aef=use_aef,
            metrics_cfg=metrics_cfg,
            use_s1=use_s1,
        )

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(run_dir / "logs" / "train_log.csv", index=False)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict(),
            "cfg": cfg_to_save,
        }

        if save_every > 0 and (epoch % save_every == 0 or epoch == epochs):
            torch.save(ckpt, run_dir / "checkpoints" / "last.pt")

        val_dice = float(val_metrics.get("global_dice", val_metrics.get("dice", float("nan"))))
        if np.isfinite(val_dice) and val_dice > best_dice:
            best_dice = val_dice
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")

        monitor_value = float(val_metrics.get(early_metric, val_dice))
        improved = False
        if np.isfinite(monitor_value):
            if early_mode == "max":
                improved = monitor_value > best_monitor + early_min_delta
            else:
                improved = monitor_value < best_monitor - early_min_delta
        if improved:
            best_monitor = monitor_value
            no_improve = 0
        else:
            no_improve += 1
        if scheduler is not None:
            scheduler.step()

        if early_enabled and no_improve >= early_patience:
            print(
                f"Early stopping at epoch {epoch} (metric={early_metric}, best={best_monitor:.6f})."
            )
            break

    return run_dir
