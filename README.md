# MuJoCo Playground

[![Build](https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_playground/ci.yml?branch=main)](https://github.com/google-deepmind/mujoco_playground/actions)
[![PyPI version](https://img.shields.io/pypi/v/playground)](https://pypi.org/project/playground/)
![Banner for playground](https://github.com/google-deepmind/mujoco_playground/blob/main/assets/banner.png?raw=true)

A comprehensive suite of GPU-accelerated environments for robot learning research and sim-to-real, built with [MuJoCo MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx).

Features include:

- Classic control environments from `dm_control`.
- Quadruped and bipedal locomotion environments.
- Non-prehensile and dexterous manipulation environments.
- Vision-based support available via [Madrona-MJX](https://github.com/shacklettbp/madrona_mjx).

For more details, check out the project [website](https://playground.mujoco.org/).

> [!NOTE]
> We now support training with both the MuJoCo MJX JAX implementation, as well as the [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) implementation at HEAD. See this [discussion post](https://github.com/google-deepmind/mujoco_playground/discussions/197) for more details.

## Installation

You can install MuJoCo Playground directly from PyPI:

```sh
pip install playground
```

> [!IMPORTANT]
> We recommend users to install [from source](#from-source) to get the latest features and bug fixes from MuJoCo.

### <a id="from-source">From Source</a>

> [!IMPORTANT]
> Requires Python 3.10 or later.

1. `git clone git@github.com:google-deepmind/mujoco_playground.git && cd mujoco_playground`
2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), a faster alternative to `pip`
3. Create a virtual environment: `uv venv --python 3.12`
4. Activate it: `source .venv/bin/activate`
5. Install CUDA 12 jax: `uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple`
    * Verify GPU backend: `python -c "import jax; print(jax.default_backend())"` should print gpu. `unset LD_LIBRARY_PATH` may need to be run before running this command.
6. Install playground from source: `uv --no-config sync --all-extras`
7. Verify installation: `uv --no-config run python -c "import mujoco_playground; print('Success')"`
    * **Note**: Menagerie assets will be downloaded automatically the first time you load a locomotion or manipulation environment. You can trigger this with: `uv --no-config run python -c "from mujoco_playground import locomotion; locomotion.load('G1JoystickFlatTerrain')"`

#### Madrona-MJX (optional)

For vision-based environments, please refer to the installation instructions in the [Madrona-MJX](https://github.com/shacklettbp/madrona_mjx?tab=readme-ov-file#installation) repository.

## Getting started

### Running from CLI
For basic usage, navigate to the repo's directory, install [from source](#from-source) with `jax[cuda12]`, and run:

```bash
train-jax-ppo --env_name CartpoleBalance
```

To train with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp):

```bash
train-jax-ppo --env_name CartpoleBalance --impl warp
```

Or with `uv`:

```bash
uv --no-config run train-jax-ppo --env_name CartpoleBalance --impl warp
uv --no-config run train-rsl-ppo --env_name CartpoleBalance --impl warp
```

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

### Training Visualization

To interactively view trajectories throughout training with [rscope](https://github.com/Andrew-Luo1/rscope/tree/main), install it (`pip install rscope`) and run:

```
python learning/train_jax_ppo.py --env_name PandaPickCube --rscope_envs 16 --run_evals=False --deterministic_rscope=True
# In a separate terminal
python -m rscope
```

## FAQ

### How can I contribute?

Get started by installing the library and exploring its features! Found a bug? Report it in the issue tracker. Interested in contributing? If you are a developer with robotics experience, we would love your help—check out the [contribution guidelines](CONTRIBUTING.md) for more details.

### Reproducibility / GPU Precision Issues

Users with NVIDIA Ampere architecture GPUs (e.g., RTX 30 and 40 series) may experience reproducibility [issues](https://github.com/google-deepmind/mujoco_playground/issues/86) in mujoco_playground due to JAX’s default use of TF32 for matrix multiplications. This lower precision can adversely affect RL training stability. To ensure consistent behavior with systems using full float32 precision (as on Turing GPUs), please run `export JAX_DEFAULT_MATMUL_PRECISION=highest` in your terminal before starting your experiments (or add it to the end of `~/.bashrc`).

To reproduce results using the same exact learning script as used in the paper, run the brax training script which is available [here](https://github.com/google/brax/blob/1ed3be220c9fdc9ef17c5cf80b1fa6ddc4fb34fa/brax/training/learner.py#L1). There are slight differences in results when using the `learning/train_jax_ppo.py` script, see the issue [here](https://github.com/google-deepmind/mujoco_playground/issues/171) for more context.

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
