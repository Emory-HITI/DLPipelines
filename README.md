# DLPipelines

A collection of deep learning pipeline templates focused on medical imaging, powered by PyTorch Lightning, Weights & Biases, and MONAI.

## Overview

DLPipelines provides ready-to-use templates for building efficient and reproducible deep learning workflows in medical imaging. The repository combines the power of PyTorch Lightning for organized training loops, Weights & Biases for comprehensive experiment tracking, and MONAI for medical-imaging-specific components.

## Features

- Standardized project structure for medical imaging deep learning projects
- Integration with PyTorch Lightning for clean and organized training code
- Automated experiment tracking and visualization with Weights & Biases
- MONAI-based data transformations and neural network architectures
- Configurable training pipelines with reproducible experiments
- Medical imaging specific data loading and preprocessing templates

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

```bash
# Install uv if you haven't already
pip install uv

# Clone the repository
git clone https://github.com/f10409/DLPipelines.git
cd DLPipelines

# Create a virtual environment and install dependencies using uv
uv sync
```

Note: Using uv significantly speeds up dependency resolution and package installation compared to traditional pip.

## Support

For questions and support, please open an issue in the GitHub repository.
