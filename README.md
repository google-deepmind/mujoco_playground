# MuJoCo Playground

[![Build](https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_playground/ci.yml?branch=main)](https://github.com/google-deepmind/mujoco_playground/actions)
[![Documentation Status](https://readthedocs.org/projects/reaf-sim/badge/?version=latest)](https://reaf-sim.readthedocs.io/en/latest/playground/index.html)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Banner for playground](https://www.gstatic.com/mujoco/playground/banner/1/banner.png)

MuJoCo Playground is a suite of robot environments, reinforcement learning tasks, and hardware workflows written in [PyTorch](https://pytorch.org/) and built on [`reaf-sim`](https://github.com/google-deepmind/reaf-sim), [MuJoCo](https://github.com/google-deepmind/mujoco), and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).

Robots exist in `mujoco_playground/robots/`, either in simulation or paired with a hardware counterpart, enabling end-to-end workflows across simulation and hardware:

- **Large-batch RL training on GPU**: with state and vision observations across control, manipulation, and humanoid whole-body tracking.
- **Sim & real teleoperation**: with leader-follower control and episodic MCAP logging.
- **System identification**: fitting simulated joint dynamics to logged hardware trajectories via `mujoco.sysid`.
- **Zero-shot sim-to-real policy deployment**: running the same task layer and policy in simulation (`--mode=sim`) and on hardware (`--mode=real`).
- **Agentic manipulation**: for vision-language model (VLM) examples across simulation and hardware (`SO-101`, `i2rt YAM`).

<table>
  <tr>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/whole_body/tracking/README.md"><img width="420" src="https://www.gstatic.com/mujoco/playground/demos/g1_lafan_dance/2/g1_lafan_dance.webp" alt="Whole body motion tracking"></a>
      <br><b><a href="mujoco_playground/tasks/whole_body/tracking/README.md">Whole body motion tracking</a></b>
    </td>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/teleop/so101/README.md"><img width="420" src="https://www.gstatic.com/mujoco/playground/demos/so101_sim_teleop/2/so101_sim_teleop.webp" alt="SO-101 Simulation Teleoperation Demo"></a>
      <br><b><a href="mujoco_playground/tasks/teleop/so101/README.md">Sim teleoperation</a></b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/teleop/so101/README.md"><img width="420" src="https://www.gstatic.com/mujoco/playground/demos/so101_real_teleop/2/so101_real_teleop.webp" alt="SO-101 Real Teleoperation Demo"></a>
      <br><b><a href="mujoco_playground/tasks/teleop/so101/README.md">Real teleoperation</a></b>
    </td>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/sysid/so101/README.md"><img width="420" src="https://www.gstatic.com/mujoco/playground/demos/so101_sysid/2/so101_sysid.webp" alt="SO-101 System Identification Rollout"></a>
      <br><b><a href="mujoco_playground/tasks/sysid/so101/README.md">System identification</a></b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/manipulation/pick_cube/so101/deploy/README.md"><img width="420" src="https://www.gstatic.com/mujoco/playground/demos/so101_pick_cube_sim2real/3/so101_pick_cube_sim2real.webp" alt="Sim2Real and interactive viewer"></a>
      <br><b><a href="mujoco_playground/tasks/manipulation/pick_cube/so101/deploy/README.md">Sim2Real and interactive viewer</a></b>
    </td>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/agentic/tabletop_v0/README.md"><img width="420" src="https://www.gstatic.com/mujoco/playground/demos/yam_mug_real/2/yam_mug_real.webp" alt="YAM Agentic Tabletop Manipulation"></a>
      <br><b><a href="mujoco_playground/tasks/agentic/tabletop_v0/README.md">Agentic manipulation</a></b>
    </td>
  </tr>
</table>

> [!NOTE]
> Looking for the original RSS 2025 JAX/MJX environments? They remain available under [`mujoco_playground/rss2025`](mujoco_playground/rss2025).

## Installation

Clone the repository and sync dependencies with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/google-deepmind/mujoco_playground.git && cd mujoco_playground

# For GPU (CUDA 12.4 + RSL-RL — recommended for RL training)
make sync

# Or for CPU-only
make sync-cpu
```

Verify your installation by running the test suite:

```bash
make test
```

> [!NOTE]
> Real-robot hardware SDKs are optional extras so simulation-only checkouts stay lightweight. To install hardware drivers for a specific robot, add `--extra so101` or `--extra yam` (e.g., `uv sync --extra cpu --extra so101`).

## Getting Started

### 1. Train & Visualize an RL Policy

Train a PPO policy across parallel GPU environments with [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (available environments: `CartpoleBalance`, `CartpoleSwingUp`, `CartpoleBalanceVision`, `SO101PickCube`, `SO101PickCubeVision`, `TrossenPickCube`, `TrossenPickCubeVision`, `G1WholeBodyTracking`):

```bash
# Train SO-101 Pick-Cube on GPU
uv run --no-sync python -m mujoco_playground.rl.rsl_rl.train \
  --env=SO101PickCube \
  --logdir=/tmp/rslrl_logs/so101_pick_cube

# Play back the trained policy in the interactive MuJoCo viewer
uv run --no-sync python -m mujoco_playground.rl.rsl_rl.play \
  --env=SO101PickCube \
  --resume_path=/tmp/rslrl_logs/so101_pick_cube
```

Monitor training curves in TensorBoard:

```bash
uv run --no-sync tensorboard --logdir=/tmp/rslrl_logs/so101_pick_cube
```

### 2. Try Agentic Manipulation in Simulation

Run the tabletop manipulation harness in simulation with the built-in scripted oracle (or connect a VLM via `--agent=bridge`):

```bash
uv run --no-sync python -m mujoco_playground.tasks.agentic.tabletop_v0.run \
  --mode=sim --robot=so101 --task=cube_in_bowl --agent=oracle --viewer
```

### 3. Workflow Guides

| Guide | Description |
|-------|-------------|
| [SO-101 Hardware Setup](mujoco_playground/robots/so101/real/README.md) | USB port & wrist camera identification and arm calibration |
| [i2rt YAM Hardware Setup](mujoco_playground/robots/yam/README.md) | SocketCAN bring-up and RealSense D405 camera setup |
| [Sim & Real Teleoperation](mujoco_playground/tasks/teleop/so101/README.md) | Leader-follower teleoperation, foot-pedal gating, and MCAP episode logging |
| [System Identification](mujoco_playground/tasks/sysid/so101/README.md) | Fit simulated joint dynamics to logged hardware trajectories with `mujoco.sysid` |
| [Sim-to-Real Pick-Cube Deployment](mujoco_playground/tasks/manipulation/pick_cube/so101/deploy/README.md) | Train vision policies and deploy zero-shot to `--mode=sim` and `--mode=real` |
| [G1 Whole-Body Motion Tracking](mujoco_playground/tasks/whole_body/tracking/README.md) | Retarget motion capture clips and train humanoid tracking policies |
| [Agentic Tabletop Manipulation](mujoco_playground/tasks/agentic/tabletop_v0/README.md) | Zero-shot sim-to-real VLM code-as-policies harness (`SO-101`, `i2rt YAM`) |

## Profiling with Tracy

MuJoCo Playground includes [Tracy](https://github.com/wolfpld/tracy) instrumentation for profiling physics substeps, environment MDP loops, and RL training (see [`profiling/README.md`](profiling/README.md)):

```bash
# Capture 10s of headless telemetry from a running training script and export stats
tracy-capture -o /tmp/training_profile.tracy -f -s 10
tracy-csvexport /tmp/training_profile.tracy

# Or connect the interactive GUI profiler (127.0.0.1:8086) / open a saved trace
tracy-profiler /tmp/training_profile.tracy
```

## Documentation & Contributing

- **Contributing:** See [mujoco_playground/docs/contributing.rst](mujoco_playground/docs/contributing.rst) for contribution guidelines (`uv run pre-commit run --all-files`, `make test`).
- **Changelog:** See [mujoco_playground/docs/changelog.rst](mujoco_playground/docs/changelog.rst).

## Citation

If you use MuJoCo Playground in your work, please cite it as follows (see the [GitHub contributors page](https://github.com/google-deepmind/mujoco_playground/graphs/contributors?from=10%2F12%2F2026) for the full list of contributors):

```bibtex
@software{mujoco_playground_2026,
  author = {Tabanpour, Baruch and Howell, Taylor and Haves, Sam and Abeyruwan, Saminda and Ma, Tony and Suh, Dan and {MuJoCo Playground Contributors}},
  title = {{MuJoCo Playground}},
  url = {https://github.com/google-deepmind/mujoco_playground},
  year = {2026},
}
```

For the RSS 2025 benchmark suite that was built on JAX/MJX (see [`mujoco_playground/rss2025`](mujoco_playground/rss2025/README.md)), please cite:

```bibtex
@misc{mujoco_playground_2025,
  title = {MuJoCo Playground: An open-source framework for GPU-accelerated robot learning and sim-to-real transfer.},
  author = {Zakka, Kevin and Tabanpour, Baruch and Liao, Qiayuan and Haiderbhai, Mustafa and Holt, Samuel and Luo, Jing Yuan and Allshire, Arthur and Frey, Erik and Sreenath, Koushil and Kahrs, Lueder A. and Sferrazza, Carlo and Tassa, Yuval and Abbeel, Pieter},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/google-deepmind/mujoco_playground}
}
```

## License & Disclaimer

All content in this repository is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

This is not an officially supported Google product.
