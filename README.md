# FiLM-SMP: Flood Segmentation with GFM Auto-Encoder Features via FiLM Conditioning

Official code for the paper:

> **Flood Water Body Segmentation Using Geospatial Foundation Model Auto-Encoder Features as an Auxiliary Modality: A Multi-Scale Fusion Approach Based on FiLM Conditioning**
>
> Pingan Zhang, Binbin Li
>
> International Journal of Applied Earth Observation and Geoinformation (JAG), 2026

## Overview

We propose the **GFM-as-Auxiliary** paradigm: instead of fine-tuning a Geospatial Foundation Model (GFM), we treat its 64-channel pixel-level Auto-Encoder Features (AEF) as a static auxiliary modality and fuse them into a U-Net segmentation backbone through a **FiLM-SMP** network that applies Feature-wise Linear Modulation (FiLM) with identity initialization at multiple encoder scales.

![Architecture](figures/architecture.png)

### Key Results (Sen1Floods11, 3 random splits, paired Wilcoxon tests)

| Finding | Detail |
|---------|--------|
| AEF + SAR | +9.11% IoU, *p* < 0.001 across all splits |
| FiLM vs Concat | 81.58% vs 78.21% IoU, *p* < 0.002 |
| S2+AEF ≈ S1+S2 | 81.58% vs 81.74% IoU |
| FiLM on S1+S2 | IoU → 82.54% |

## Installation

```bash
# Clone and install dependencies
git clone https://github.com/<your-username>/FiLM-SMP.git
cd FiLM-SMP
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, CUDA-capable GPU (24 GB recommended)

## Data Preparation

### 1. Sen1Floods11

Download the Sen1Floods11 dataset from [GitHub](https://github.com/cloudtostreet/Sen1Floods11) and organize as:

```
data/
├── S2Hand/          # Sentinel-2 chips: {country}_{chipid}_S2Hand.tif
├── S1Hand/          # Sentinel-1 chips: {country}_{chipid}_S1Hand.tif
├── LabelHand/       # Labels: {country}_{chipid}_LabelHand.tif
└── AEF64/           # AEF features (see below)
```

### 2. Auto-Encoder Features (AEF)

AEF are extracted from Google DeepMind's [GAESE product](https://developers.google.com/earth-engine/datasets/catalog/projects_deepmind_assets_global_autoencoder_satellite_embeddings_v1) via Google Earth Engine.

Export 64-channel AEF tiles matching each Sen1Floods11 chip:
```
data/AEF64/{country}_{chipid}_AEF64_{year}.tif
```

## Training

```bash
# Train FiLM-SMP with S2+AEF (main experiment)
python train.py --config configs/s2_aef_film.yaml

# Train with S1+S2+AEF
python train.py --config configs/s1s2_aef_film.yaml

# Baselines
python train.py --config configs/s2_only.yaml
python train.py --config configs/s1_only.yaml
python train.py --config configs/s1s2_concat.yaml
python train.py --config configs/s2_aef_concat.yaml
```

Results are saved to `results/<run_name>/`.

## Evaluation

```bash
python eval.py --run_dir results/<run_name> --split test
```

Per-sample metrics are saved to `results/<run_name>/eval/test/per_sample_metrics.csv`.

## Configuration

All experiments are configured via YAML files. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.name` | `film_smp` (FiLM) or `smp_unet` (Concat) | `film_smp` |
| `data.use_s2` | Use Sentinel-2 bands | `true` |
| `data.use_s1` | Use Sentinel-1 bands | `false` |
| `data.use_aef` | Use AEF features | `true` |
| `loss.name` | Loss function | `bce_lovasz` |
| `loss.bce_weight` / `loss.lovasz_weight` | Loss weights | 0.3 / 0.7 |
| `optim.lr` | Learning rate | 1e-3 |
| `optim.encoder_lr_scale` | Encoder LR multiplier | 0.1 |
| `train.epochs` | Max epochs | 300 |
| `train.early_stop.patience` | Early stopping patience | 50 |

## Project Structure

```
FiLM-SMP/
├── train.py                    # Training entry point
├── eval.py                     # Evaluation entry point
├── configs/                    # Experiment configurations
├── main/
│   ├── models/
│   │   ├── film_smp_model.py   # FiLM-SMP model (core contribution)
│   │   ├── smp_unet.py         # SMP UNet baseline (concat)
│   │   └── factory.py          # Model builder
│   ├── data/
│   │   ├── dataset.py          # FloodSegDataset
│   │   ├── io.py               # Sentinel-1/2 and AEF I/O
│   │   ├── manifest.py         # Data manifest builder
│   │   ├── normalization.py    # Channel statistics
│   │   ├── collate.py          # Batch collation
│   │   └── splits.py           # Train/val/test splitting
│   ├── losses/
│   │   ├── __init__.py         # Loss factory
│   │   └── lovasz.py           # BCE + Lovász-Hinge loss
│   ├── metrics/
│   │   ├── __init__.py         # Metrics factory
│   │   └── binary.py           # IoU, Dice, Precision, Recall
│   ├── training/
│   │   └── trainer.py          # Training loop
│   ├── evaluation/
│   │   └── evaluator.py        # Evaluation loop
│   └── utils/
│       ├── config.py           # YAML/JSON config loader
│       └── seed.py             # Reproducibility utilities
└── requirements.txt
```

## Citation

```bibtex
@article{Zhang2026FiLMSMP,
  author  = {Zhang, Pingan and Li, Binbin},
  title   = {Flood Water Body Segmentation Using Geospatial Foundation Model
             Auto-Encoder Features as an Auxiliary Modality: A Multi-Scale
             Fusion Approach Based on {FiLM} Conditioning},
  journal = {Int. J. Appl. Earth Obs. Geoinf.},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset by Bonafilia et al.
- [GAESE](https://developers.google.com/earth-engine/datasets/catalog/projects_deepmind_assets_global_autoencoder_satellite_embeddings_v1) by Google DeepMind
- [Segmentation Models PyTorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch)
