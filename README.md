# 🚀 `Ai-sampler`

*Learning to sample with Adversarial Involutive Markov kernals.*

Creators: [Riccardo Valperga](https://twitter.com/RValperga), [Evgenii Egorov](https://github.com/evgenii-egorov)

[![License: MIT](https://img.shields.io/badge/License-MIT-purple)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Style](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Read the docs](https://img.shields.io/badge/docs-latest-blue)](https://fit-a-nef.readthedocs.io/en/latest/)
![Schema](assets/fig-1.png)

📚 This code is for reproducing the experiments in:  [Ai-sampler: Adversarial Learning of Markov kernels with involutive maps](https://arxiv.org/abs/2406.02490) <img src="assets/arxiv.png" width=20px>.

## Getting started with 🚀 `Ai-sampler`

### Installation

To use this library, simply clone the repository and run:

```bash
pip install .
```

This will install the `Ai-sampler` library and all its dependencies. To ensure that JAX and PyTorch are installed with the right CUDA/cuDNN version of your platform, we recommend installing them first (see instructions on the official [Jax](https://jax.readthedocs.io/en/latest/installation.html) and [Pytorch](https://pytorch.org/get-started/locally/)), and then run the command above. You can install `wandb` for better logging. 

### Repository structure

The repository is structured as follows:

- `./aisampler`. **Library** source code that implements the `Ai-sampler`. 
- `./data`. Contains the data for Bayesian logistic regression.
- `./experiments`. **Collection** of the experiments.
- - `/train`. Scripts for training the `Ai-sampler`.
- - `/test`. Scripts for sampling with the trainied `Ai-sampler` and with HMC.

### Usage

To train the `Ai-sampler` on the **2D densities**, from the root folder run:

```bash
python experiments/training/train_toy_density.py --task.target_density.name=hamiltonian_mog2 --task.checkpoint.checkpoint_dir=./checkpoints
```

## Citing

If you want to cite us use the following BibTeX entry:

```bibtex
@article{egorov2024ai,
  title={Ai-Sampler: Adversarial Learning of Markov kernels with involutive maps},
  author={Egorov, Evgenii and Valperga, Ricardo and Gavves, Efstratios},
  journal={arXiv preprint arXiv:2406.02490},
  year={2024}
}
```