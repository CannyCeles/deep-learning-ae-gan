<div align="center">

# Comparative Generative Models Study

### Empirical Benchmark of AE, Sparse AE, VAE, GAN, DCGAN, and DDPM on Fashion-MNIST with Classifier Inversion Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-Fashion--MNIST-4B8BBE.svg?style=flat)](https://github.com/zalandoresearch/fashion-mnist)

</div>

---

## Overview

This repository implements a standardized empirical comparison of six generative model families alongside a companion MLP classifier limitation experiment for Deep Learning (D6421) Session 07. All models share a fixed stratified 80/10/10 Fashion-MNIST split (`seed = 42`) and unified evaluation protocols covering reconstruction fidelity (MSE, PSNR, SSIM), generative quality (relative InceptionV3 FID), training stability, and batch-64 inference latency.

---

## Models & Experimental Scope

| Model | Latent Dim | Output Range | Objective / Notes |
|---|---|---|---|
| **MLP Classifier** | 64 | Logits (10 classes) | Cross-Entropy companion model for weight-transpose, frozen-decoder, and prior-sampling proofs |
| **Vanilla Autoencoder** | 64 | `[0, 1]` (Sigmoid) | MSE reconstruction baseline |
| **Sparse Autoencoder** | 256 (overcomplete) | `[0, 1]` (Sigmoid) | MSE + L1 sparsity penalty on ReLU latent activations (higher capacity than 64-dim models) |
| **Variational Autoencoder (VAE)** | 64 | `[0, 1]` (Sigmoid) | BCE/MSE reconstruction + KL divergence |
| **Vanilla GAN** | 64 (noise prior) | `[-1, 1]` (Tanh) | Fully-connected adversarial minimax with `BCEWithLogitsLoss` |
| **DCGAN** | 100 (noise prior) | `[-1, 1]` (Tanh) | Convolutional generator & discriminator with BatchNorm and LeakyReLU |
| **DDPM** | 28×28 image space | `[-1, 1]` | 2D U-Net predicting Gaussian noise over `T = 300` diffusion steps (scaled linear beta schedule) |

> [!NOTE]
> Benchmark metrics are stored in `results/runs.csv` and `results/metrics.json`. FID scores use `torchvision` InceptionV3 on upscaled 28×28 grayscale images and are intended for relative comparison across models within this setup.

---

## Repository Structure

```text
.
├── notebooks/
│   └── tugas-ae-gan.ipynb     # End-to-end experiment notebook (run top to bottom)
├── results/
│   ├── splits.json            # Fixed stratified 80/10/10 Fashion-MNIST indices
│   ├── runs.csv               # Summary comparison table across all models
│   └── metrics.json           # Detailed per-model metrics and metadata
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch & Torchvision |
| **Scientific Computing** | NumPy, SciPy (`linalg.sqrtm` for FID), Scikit-learn |
| **Visualization** | Matplotlib |
| **Tabular Logging** | Pandas, CSV & JSON |
| **Notebook Runtime** | IPyKernel |

---

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable NVIDIA GPU recommended (tested on RTX 4060 8GB; falls back to CPU automatically)

### Installation & Execution

1. **Clone the repository**:
   ```bash
   git clone <this-repo-url>
   cd tugas-ae-gan
   ```

2. **Create and activate an environment, then install dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   Open `notebooks/tugas-ae-gan.ipynb` and run **Restart Kernel and Run All Cells**.

### Notes
- **Dataset**: Fashion-MNIST (~30 MB) downloads automatically into `data/` on first run if missing.
- **Checkpoints & Caching**: Trained weights are saved under `results/weights/`. If checkpoints and logs exist, training cells reload them unless `FORCE_RERUN = True` is set in the setup cell.
- **Reproducibility**: Global random seed is fixed at `42`, and exact split indices are stored in `results/splits.json`.
