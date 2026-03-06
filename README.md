# Multi-Modal RGB-SAR Fusion for Building Detection & Height Estimation

This repository contains the source code for a Master's thesis project focused on multi-task deep learning for satellite imagery analysis. The system performs simultaneous instance segmentation (building footprints) and Digital Surface Model (DSM) height regression by fusing optical RGB images with Synthetic Aperture Radar (SAR) data.

Built on top of the **OpenMMLab (MMDetection)** framework and **PyTorch**, this project extends the base **SOLOv2** architecture to support 4-channel inputs and multi-modal feature fusion.

## 🚀 Key Features

* **Custom Multi-Task Head:** Extended SOLOv2 mask head with an additional regression branch for building height estimation (DSM).
* **Multi-Modal Fusion Strategies:**
  * **Early Fusion:** 4-channel input (RGB + SAR) processed by a single ResNet backbone.
  * **Feature-Level Fusion:** Dual-stream ResNet architecture with independent feature extraction and 1x1 convolution bottleneck reduction prior to the FPN.
  * **Late Fusion (Decision Level):** Dedicated preprocessors for isolated RGB and SAR training pipelines.
* **Custom Data Pipeline:** Custom datasets and preprocessors allowing native 4-channel normalization, bypassing standard framework-level hardcoded RGB constraints.
* **Automated Benchmarking:** Sequential batch scripts for uninterrupted training of all fusion models on remote GPU clusters.

## 📁 Repository Structure

```text
├── datasets/
│   └── mmd_custom_dataset.py      # Custom dataset loader and RGB-SAR synchronizer
├── models/
│   └── custom_solov2.py           # DualResNet backbone, Multi-task Head, Late Fusion preprocessor
├── models_config/
│   ├── config_early.py            # Configuration for Early Fusion
│   ├── config_feature.py          # Configuration for Feature-Level Fusion
│   ├── config_late_rgb.py         # Configuration for Late Fusion (RGB Expert)
│   └── config_late_sar.py         # Configuration for Late Fusion (SAR Expert)
├── pack_project.py                # Utility script for packing the project for remote execution
└── run_all_experiments.bat        # Windows batch script for sequential training
```
# Installation
The environment is containerized using Conda. It requires PyTorch with CUDA 12.1 (optimized for RTX 3000/4000/5000 series).

```bash
# 1. Create and activate the environment
conda create -n mgr python=3.10 -y
conda activate mgr

# 2. Install PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install OpenMMLab dependencies
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install mmdet
```

# Training 
To run the complete suite of experiments sequentially (Early, Feature, and Late Fusion experts), use the provided batch script. Ensure you are in the root directory of the project.

## On Windows:

```
run_all_experiments.bat
```
On Linux (Manual Execution):

```bash
PYTHONPATH="." mim train mmdet models_config/config_early.py --work-dir runs/early_fusion
```

# Transfer Learning
To utilize ImageNet pretrained weights for custom 4-channel inputs or dual-stream backbones, dedicated weight 'surgery' scripts (weights_adjust*.py) are used to clone and average RGB filters into the SAR channel, preventing the network from training the first convolutional layers from scratch.

*Note: The models/baselines/ directory contains legacy PyTorch implementations of U-Net and DeepLabV3+ used as baseline comparisons for the thesis research.*