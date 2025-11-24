# UniGame: Turning a Unified Multimodal Model Into Its Own Adversary

Official implementation of **UniGame**, a self-adversarial post-training framework for Unified Multimodal Models (UMMs).
| ![Image1](./fig/consistency.jpg) | ![Image2](./fig/fig-manifold.jpg) |
|:---:|:---:|
## Overview
UniGame is the first self-adversarial post-training framework that improves the consistency between understanding and generation pathways in Unified Multimodal Models. By treating the generation pathway as an active adversary, UniGame enables the model to discover and correct its own inconsistencies.
 ![Image1](./fig/fig-framework.jpg)
## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/AIFrontierLab/UniGame.git
cd UniGame

# Create conda environment
conda create -n unigame python=3.11 -y
conda activate unigame

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Dataset

Download the VQAv2 dataset and update the path in `main.py`:
```python
LOCAL_VQAV2 = "/path/to/your/vqav2"
```

### 2. Training
**Single GPU:**
```bash
python main.py
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 main.py
```

**SLURM Cluster:**
```bash
srun --gres=gpu:4 --cpus-per-task=16 torchrun --nproc_per_node=4 main.py
```

## Citation

If you find this work useful, please cite:


## Acknowledgements

We thank the authors of [Janus-Pro](https://github.com/deepseek-ai/Janus), [VQAv2](https://visualqa.org/), and other open-source projects that made this work possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue or contact:
- Zhaolong Su: zsu05@wm.edu
