<div align="center">
  <img src="./fig/logo.png" width="40%" alt="AIFrontier Lab" />
</div>

<hr>

<div align="center">
<h1>UniGame: Turning a Unified Multimodal Model Into Its Own Adversary</h1>
</div>

<div align="center">

  <a href="https://jd92.wang/" target="_blank">
    <img alt="Lab Homepage" src="https://img.shields.io/badge/Lab-AIFrontier-blue?color=3273dc" />
  </a>
  <a href="https://rollingsu.github.io/" target="_blank">
    <img alt="Zhaolong Su" src="https://img.shields.io/badge/Zhaolong-Su-orange?color=f5a623" />
  </a>
  <a href="https://arxiv.org/abs/2511.19413" target="_blank">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&color=b31b1b" />
  </a>
  <a href="https://github.com/AIFrontierLab/UniGame" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-UniGame-black?logo=github" />
  </a>
  <a href="LICENSE">
    <img alt="License" src="https://imgm.shields.io/badge/License-MIT-f5de53?&color=f5de53">
  </a>

</div>

<p align="center">
  <a href="#overview"><b>Overview</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#quick-start"><b>Quick Start</b></a> |
  <a href="#citation"><b>Citation</b></a>
</p>

<hr>

Official implementation of **UniGame**, a self-adversarial post-training framework for Unified Multimodal Models (UMMs).
<p align="center">
  <img src="./fig/consistency.jpg" width="32%">
  <img src="./fig/fig-manifold.jpg" width="24%">
</p>

## Overview
UniGame is the first self-adversarial post-training framework that improves the consistency between understanding and generation pathways in Unified Multimodal Models. By treating the generation pathway as an active adversary, UniGame enables the model to discover and correct its own inconsistencies.

<p align="center">
  <img src="./fig/fig-framework.jpg" width="80%">
</p>

Quantitative Results:

<p align="center">
  <img src="./fig/fig-case.jpg" width="80%">
</p>
<p align="center">
  <img src="./fig/fig-case_study_gen.jpg" width="80%">
</p>

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

```bibtex
@inproceedings{Su2025UniGameTA,
  title={UniGame: Turning a Unified Multimodal Model Into Its Own Adversary},
  author={Zhaolong Su and Wang Lu and Hao Chen and Sharon Li and Jindong Wang},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:283244819}
}
```
## Acknowledgements

We thank the authors of [Janus-Pro](https://github.com/deepseek-ai/Janus), [VQAv2](https://visualqa.org/), and other open-source projects that made this work possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue or contact:
- Zhaolong Su: zsu05@wm.edu
