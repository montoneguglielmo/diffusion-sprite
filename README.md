# Diffusion Sprite Generator

A PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) for generating 16x16 sprite images. This project uses a U-Net based architecture with context embeddings to learn and generate sprite images through the diffusion process.

## Overview

This project implements a diffusion model that can generate sprite images by learning to reverse a gradual noising process. The model uses a ContextUnet architecture, which is a U-Net with residual connections and context embeddings for conditional generation.

## Features

- **DDPM Implementation**: Full implementation of the Denoising Diffusion Probabilistic Model
- **U-Net Architecture**: Context-aware U-Net with residual blocks for noise prediction
- **Sprite Generation**: Optimized for 16x16 RGB sprite images
- **Context Embeddings**: Support for conditional generation with context labels
- **Training & Inference**: Complete training pipeline and generation script

## Project Structure

```
diffusion-sprite/
├── config.py          # Hyperparameters and configuration
├── dataset.py         # Custom PyTorch dataset for sprite data
├── model.py           # ContextUnet architecture and components
├── train.py           # Training script
├── generate.py        # Generation/sampling script
├── utils.py           # Utility functions for visualization
└── requirements.txt   # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusion-sprite
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The `config.py` file contains all hyperparameters:

- **Network**: `N_FEAT` (64), `N_CFEAT` (5), `HEIGHT` (16)
- **Diffusion**: `TIMESTEPS` (500), `BETA1` (1e-4), `BETA2` (0.02)
- **Training**: `BATCH_SIZE` (100), `N_EPOCH` (32), `LRATE` (1e-3)

You can modify these values to adjust model capacity, training duration, and diffusion schedule.

## Data Format

The dataset expects two NumPy files:
- **Sprite images**: Shape `(N, 16, 16, 3)` - RGB sprite images
- **Labels**: Shape `(N,)` - Context labels for each sprite

Example dataset paths (as used in `train.py`):
- `../datasets/sprites/sprites_1788_16x16.npy`
- `../datasets/sprites/sprite_labels_nc_1788_16x16.npy`

## Usage

### Training

Train the diffusion model on your sprite dataset:

```bash
python train.py
```

The training script will:
- Load the dataset from the specified paths
- Train the model for the configured number of epochs
- Save model checkpoints every 4 epochs to `./weights/`
- Use linear learning rate decay

**Note**: Update the dataset paths in `train.py` (line 18) to point to your data files.

### Generation

Generate new sprites from a trained model:

```bash
python generate.py
```

The generation script will:
- Load the trained model from `./weights/model_trained.pth`
- Generate 20 samples using the DDPM sampling process
- Save intermediate generation steps to `intermediate_images/` directory
- Display progress for each timestep

**Note**: Make sure you have a trained model saved as `model_trained.pth` in the `./weights/` directory, or update the path in `generate.py`.

## Model Architecture

The `ContextUnet` model consists of:

1. **Encoder (Downsampling)**:
   - Initial residual convolution block
   - Two downsampling blocks with residual connections
   - Average pooling to bottleneck

2. **Context & Time Embeddings**:
   - Separate embeddings for timestep and context labels
   - Multi-layer feedforward networks

3. **Decoder (Upsampling)**:
   - Transposed convolutions for upsampling
   - Skip connections from encoder
   - Residual blocks for feature refinement
   - Final convolution to output channels

## How It Works

1. **Forward Process (Training)**: Images are gradually corrupted with Gaussian noise over T timesteps
2. **Reverse Process (Generation)**: The model learns to predict and remove noise at each timestep
3. **Sampling**: Starting from pure noise, the model iteratively denoises to generate new images

## Output

- **Training**: Model checkpoints saved to `./weights/model_{epoch}.pth`
- **Generation**: 
  - Final generated sprites
  - Intermediate generation grids saved to `intermediate_images/` showing the denoising process

## Requirements

- Python 3.7+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- tqdm >= 4.65.0

## Acknowledgments

This implementation is based on the Denoising Diffusion Probabilistic Models (DDPM) paper by Ho et al. (2020).

