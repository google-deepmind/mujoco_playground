# MuJoCo Playground

[![Build](https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_playground/ci.yml?branch=main)](https://github.com/google-deepmind/mujoco_playground/actions)
[![PyPI version](https://img.shields.io/pypi/v/playground)](https://pypi.org/project/playground/)
![Banner for playground](https://github.com/google-deepmind/mujoco_playground/blob/main/assets/banner.png?raw=true)

A comprehensive suite of GPU-accelerated environments for robot learning research and sim-to-real, built with [MuJoCo MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx).

Features include:

- Classic control environments from `dm_control` reimplemented in MJX.
- Quadruped and bipedal locomotion environments.
- Non-prehensile and dexterous manipulation environments.
- Vision-based support available via [Madrona-MJX](https://github.com/shacklettbp/madrona_mjx).

For more details, check out the project [website](https://playground.mujoco.org/).

## Installation

You can install MuJoCo Playground directly from PyPI:

```sh
pip install playground
```

### From Source

> [!IMPORTANT]
> Requires Python 3.10 or later.

1. `git clone git@github.com:google-deepmind/mujoco_playground.git && cd mujoco_playground`
2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), a faster alternative to `pip`
3. Create a virtual environment: `uv venv --python 3.11`
4. Activate it: `source .venv/bin/activate`
5. Install CUDA 12 jax: `uv pip install -U "jax[cuda12]"`
    * Verify GPU backend: `python -c "import jax; print(jax.default_backend())"` should print gpu
6. Install playground: `uv pip install -e ".[all]"`
7. Verify installation (and download Menagerie): `python -c "import mujoco_playground"`

#### Madrona-MJX (optional)

For vision-based environments, please refer to the installation instructions in the [Madrona-MJX](https://github.com/shacklettbp/madrona_mjx?tab=readme-ov-file#installation) repository.

## Getting started

### Basic Tutorials
| Colab | Description |
|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb) | Introduction to the Playground with DM Control Suite |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb) | Locomotion Environments |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb) | Manipulation Environments |

### Vision-Based Tutorials (GPU Colab)
| Colab | Description |
|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1_t4.ipynb) | Training CartPole from Vision (T4 Instance) |

### Local Runtime Tutorials
*Requires local Madrona-MJX installation*

| Colab | Description |
|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_1.ipynb) | Training CartPole from Vision |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/training_vision_2.ipynb) | Robotic Manipulation from Vision |

## How can I contribute?

Get started by installing the library and exploring its features! Found a bug? Report it in the issue tracker. Interested in contributing? If you are a developer with robotics experience, we would love your help—check out the [contribution guidelines](CONTRIBUTING.md) for more details.

## Citation

If you use Playground in your scientific works, please cite it as follows:

```bibtex
@misc{mujoco_playground_2025,
  title = {MuJoCo Playground: An open-source framework for GPU-accelerated robot learning and sim-to-real transfer.},
  author = {Zakka, Kevin and Tabanpour, Baruch and Liao, Qiayuan and Haiderbhai, Mustafa and Holt, Samuel and Luo, Jing Yuan and Allshire, Arthur and Frey, Erik and Sreenath, Koushil and Kahrs, Lueder A. and Sferrazza, Carlo and Tassa, Yuval and Abbeel, Pieter},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/google-deepmind/mujoco_playground}
}
```

## License and Disclaimer

The texture used in the rough terrain for the locomotion environments is from [Polyhaven](https://polyhaven.com/a/rock_face) and licensed under [CC0](https://creativecommons.org/public-domain/cc0/).

All other content in this repository is licensed under the Apache License, Version 2.0. A copy of this license is provided in the top-level [LICENSE](LICENSE) file in this repository. You can also obtain it from https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
