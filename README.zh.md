# MuJoCo Playground

[![Build](https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_playground/ci.yml?branch=main)](https://github.com/google-deepmind/mujoco_playground/actions)
[![PyPI version](https://img.shields.io/pypi/v/playground)](https://pypi.org/project/playground/)
![Banner for playground](https://github.com/google-deepmind/mujoco_playground/blob/main/assets/banner.png?raw=true)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

MuJoCo Playground 是一套专为机器人学习研究与仿真到真机迁移（Sim-to-Real）打造的全面 GPU 加速环境套件，基于 [MuJoCo MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx) 构建。

核心特性包括：

- 源自 `dm_control` 的经典控制环境。
- 四足与双足足式移动（Locomotion）环境。
- 非抓握式与灵巧手操作（Manipulation）环境。
- 通过 [MJWarp 批量渲染器 (MJWarp Batch Renderer)](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#batch-rendering) 提供基于视觉的观测支持。

更多详细信息，请访问项目[官方网站](https://playground.mujoco.org/)。

> [!NOTE]
> 我们现已同时支持使用 MuJoCo MJX JAX 实现以及最新的 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 实现进行训练。详情请参阅该[讨论帖](https://github.com/google-deepmind/mujoco_playground/discussions/197)。

## 安装指南

你可以直接从 PyPI 安装 MuJoCo Playground：

```sh
pip install playground
```

> [!IMPORTANT]
> 我们推荐用户[从源码安装](#from-source)，以获取 MuJoCo 最新的特性与问题修复。

### <a id="from-source">从源码安装 (From Source)</a>

> [!IMPORTANT]
> 需要 Python 3.10 或更高版本。

1. `git clone git@github.com:google-deepmind/mujoco_playground.git && cd mujoco_playground`
2. [安装 uv](https://docs.astral.sh/uv/getting-started/installation/)（比 `pip` 更快的高性能包管理器）
3. 创建虚拟环境：`uv venv --python 3.12`
4. 激活虚拟环境：`source .venv/bin/activate`
5. 安装适配 CUDA 12 的 JAX：`uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple`
    * 验证 GPU 后端：执行 `python -c "import jax; print(jax.default_backend())"` 应输出 `gpu`。运行此命令前可能需要执行 `unset LD_LIBRARY_PATH`。
6. 从源码安装 Playground：`uv --no-config sync --all-extras`
7. 验证安装：`uv --no-config run python -c "import mujoco_playground; print('Success')"`
    * **提示**：首次加载移动（Locomotion）或操作（Manipulation）环境时，系统将自动下载 Menagerie 机器人资产。你可以执行以下命令预先触发下载：`uv --no-config run python -c "from mujoco_playground import locomotion; locomotion.load('G1JoystickFlatTerrain')"`

## 快速入门

### 通过命令行 (CLI) 运行
基础使用场景下，请切换至本仓库目录，按[从源码安装](#from-source)指引配置好 `jax[cuda12]` 后运行：

```bash
train-jax-ppo --env_name CartpoleBalance
```

如需使用 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 进行训练：

```bash
train-jax-ppo --env_name CartpoleBalance --impl warp
```

或使用 `uv` 运行：

```bash
uv --no-config run train-jax-ppo --env_name CartpoleBalance --impl warp
uv --no-config run train-rsl-ppo --env_name CartpoleBalance --impl warp
```

### 基础入门教程
| Colab 交互式运行 | 教程描述 |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb) | 搭配 DM Control Suite 快速入门 Playground |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb) | 足式移动 (Locomotion) 环境实战 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb) | 机械臂与灵巧手操作 (Manipulation) 环境实战 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/vision.ipynb) | 视觉观测 (Vision) 环境实战 |

### 训练过程可视化 (Training Visualization)

若要在训练过程中使用 [rscope](https://github.com/Andrew-Luo1/rscope/tree/main) 交互式查看运动轨迹，请先安装该工具（`pip install rscope`），然后运行：

```bash
python learning/train_jax_ppo.py --env_name PandaPickCube --rscope_envs 16 --run_evals=False --deterministic_rscope=True
# 在另一个终端窗口中启动
python -m rscope
```

## 常见问题 (FAQ)

### 如何参与贡献？

欢迎首先安装该库并探索各项功能！如果发现任何 Bug，欢迎在 Issue 跟踪器中提交反馈。有意参与开源贡献？如果你是一名具备机器人学研发经验的开发者，我们十分期待你的加入 —— 详情请查阅[贡献指南 (CONTRIBUTING.md)](CONTRIBUTING.md)。

### 实验可复现性与 GPU 精度问题 (Reproducibility / GPU Precision Issues)

使用 NVIDIA Ampere 架构 GPU（例如 RTX 30 系列与 40 系列）的用户，可能会在 mujoco_playground 中遇到因 JAX 默认采用 TF32 进行矩阵乘法而导致的[复现性问题](https://github.com/google-deepmind/mujoco_playground/issues/86)。该低精度模式可能会对强化学习训练的数值稳定性造成不良影响。为确保与使用完整 Float32 精度系统（如 Turing 架构 GPU）保持行为一致，请在开始实验前在终端中运行 `export JAX_DEFAULT_MATMUL_PRECISION=highest`（或将其添加到 `~/.bashrc` 末尾）。

如需使用与原论文完全一致的学习脚本复现结果，请运行 [Brax 官方训练脚本](https://github.com/google/brax/blob/1ed3be220c9fdc9ef17c5cf80b1fa6ddc4fb34fa/brax/training/learner.py#L1)。使用 `learning/train_jax_ppo.py` 脚本时会有微小的数值差异，更多背景信息请参阅[该 Issue 讨论](https://github.com/google-deepmind/mujoco_playground/issues/171)。

## 论文引用 (Citation)

如果您在科学研究中使用了 Playground，请按照以下格式进行引用：

```bibtex
@misc{mujoco_playground_2025,
  title = {MuJoCo Playground: An open-source framework for GPU-accelerated robot learning and sim-to-real transfer.},
  author = {Zakka, Kevin and Tabanpour, Baruch and Liao, Qiayuan and Haiderbhai, Mustafa and Holt, Samuel and Luo, Jing Yuan and Allshire, Arthur and Frey, Erik and Sreenath, Koushil and Kahrs, Lueder A. and Sferrazza, Carlo and Tassa, Yuval and Abbeel, Pieter},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/google-deepmind/mujoco_playground}
}
```

## 开源协议与免责声明 (License and Disclaimer)

足式移动环境中用于粗糙地形的地形纹理来自 [Polyhaven](https://polyhaven.com/a/rock_face)，并遵循 [CC0](https://creativecommons.org/public-domain/cc0/) 协议。

本代码仓库中的所有其他内容均遵循 Apache License 2.0 开源协议。协议副本位于仓库根目录的 [LICENSE](LICENSE) 文件中。您也可以从 https://www.apache.org/licenses/LICENSE-2.0 获取。

这不是一款得到 Google 官方支持的产品 (This is not an officially supported Google product)。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月21日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
