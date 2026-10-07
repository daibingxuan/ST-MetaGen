# ST-MetaGen

**Structure-Property Guided 3D Microstructure Generation via Discrete Diffusion Models**

ST-MetaGen is a deep learning framework for generating diverse 3D microstructures using discrete diffusion models with multi-condition guidance, including geometric descriptors and elastic material properties.

## Overview

This repository implements a conditional discrete diffusion model for 3D voxel-based microstructure generation. The model supports:

- **Geometric condition guidance** (one-hot encoded structure family, 50-dim)
- **Elastic property condition guidance** (Young's modulus, anisotropy, Poisson's ratio)
- **Classifier-Free Guidance (CFG)** with dual independent weights
- **Sliding Tracking Control CFG (ST-CFG)** for enhanced condition controllability
- **Condition dropout** during training for robust missing-condition inference
- **Voxel post-processing** with periodicity enforcement, connectivity check, and dangling node removal

## Project Structure

```
├── network/
│   ├── model.py              # OccupancyDiffusion: core diffusion model
│   ├── model_trainer.py      # DiffusionModel: PyTorch Lightning training module
│   ├── model_utils.py        # Network building blocks (ResNet, Attention, EMA, etc.)
│   ├── unet.py               # 3D UNet denoiser with cross-attention
│   └── data_loader_text.py   # Dataset loaders for voxel data
├── shape_generation_code/
│   ├── mainDPPIsoGenerator_share.m   # MATLAB: DPP-based isotropic structure generation
│   ├── mainIsoGenerator_share.m      # MATLAB: isotropic structure generation
│   ├── gen_voxel_and_cond.py         # Voxelization and condition extraction
│   ├── plotIsosurface.m              # MATLAB: isosurface visualization
│   ├── getLSField2.m                 # MATLAB: level-set field computation
│   ├── structureFactors.m            # MATLAB: structure factor analysis
│   ├── metaset_64_npy/               # Preprocessed 64³ voxel dataset
│   └── prop_norm_params.npz          # Property normalization parameters
├── utils/
│   ├── utils.py              # Utility functions
│   └── mesh_utils.py         # Mesh processing utilities
├── train.py                  # Training entry point
├── gen_val_set.py            # Validation set generation & sampling
└── eval_val.py               # Comprehensive evaluation metrics
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- PyTorch Lightning
- NumPy, SciPy, Pandas
- tqdm, fire, bitstring
- tensorboard

## Usage

### Training

```bash
python train.py \
    --dataset_folder /path/to/metaset_64_npy \
    --voxel_folder /path/to/voxel_data \
    --csv_path /path/to/voxel_descriptions.csv \
    --name experiment_name \
    --image_size 64 \
    --base_channels 64 \
    --training_epoch 200 \
    --batch_size 4 \
    --lr 2e-4 \
    --use_geo_condition True \
    --use_tensor_condition False \
    --use_validation True \
    --val_ratio 0.2
```

### Validation Set Generation

```bash
# Standard CFG sampling
python gen_val_set.py \
    --ckpt /path/to/checkpoint.ckpt \
    --output ./val_results \
    --steps 100 \
    --w_geo 2.0 \
    --w_prop 1.6 \
    --num_per 3 \
    --batch 8

# With ST-CFG enabled
python gen_val_set.py \
    --ckpt /path/to/checkpoint.ckpt \
    --smc_enable \
    --smc_lambda 5.0 \
    --smc_K 0.2 \
    --smc_sat \
    --smc_both

# Random 10-condition visualization
python gen_val_set.py \
    --ckpt /path/to/checkpoint.ckpt \
    --rand10vis
```

### Evaluation

```bash
python eval_val.py \
    --result_dir ./val_results/experiment_tag \
    --dataset_folder /path/to/metaset_64_npy
```

**Evaluation metrics include:**
- Volume fraction control accuracy (MAE)
- Connectivity validity (single connected component rate)
- Dangling node constraint satisfaction
- Symmetry score (90° rotation & inversion IoU)
- Periodicity consistency
- Structural similarity (IoU vs. ground truth)
- Intra-condition diversity
- Real-mode coverage

## Key Features

### Dual-Condition CFG

The model employs dual classifier-free guidance with independent weights for geometric (`w_geo`) and property (`w_prop`) conditions:

```
x_pred = x_uncond + w_geo * (x_geo - x_uncond) + w_prop * (x_prop - x_uncond)
```

### ST-CFG Sampling

Sliding Tracking Control based CFG provides enhanced condition tracking through adaptive control signals:

```
s_t = (e_t - e_{t-1}) + λ * e_{t-1}
u_t = -K * sign(s_t)    # or saturated version
```

### Condition Dropout

During training, conditions are randomly dropped to enable flexible inference:
- Geometric condition dropout (global + per-block)
- Property condition dropout (global + per-dimension)

## Dataset

The dataset consists of 294 structure families × 100 samples = 29,400 voxelized microstructures at 64³ resolution. Each sample includes:
- Binary voxel representation (64×64×64)
- Geometric condition (one-hot encoding, 50-dim)
- Optional elastic properties (E, ANI, ν)

## License

This project is provided for academic and research purposes.

## Citation

If you use this code in your research, please cite our work.
