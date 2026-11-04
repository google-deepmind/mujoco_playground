# MuJoCo Playground

[![Build](https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_playground/ci.yml?branch=main)](https://github.com/google-deepmind/mujoco_playground/actions)
[![Documentation Status](https://readthedocs.org/projects/reaf-sim/badge/?version=latest)](https://reaf-sim.readthedocs.io/en/latest/playground/index.html)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Banner for playground](https://www.gstatic.com/reaf-sim/banner/0/banner.png)

MuJoCo Playground is a suite of robot environments, reinforcement learning tasks, and hardware workflows written in [PyTorch](https://pytorch.org/) and built on [`reaf-sim`](https://github.com/google-deepmind/reaf-sim), [MuJoCo](https://github.com/google-deepmind/mujoco), and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).

Robots exist in [`mujoco_playground/robots/`](mujoco_playground/robots), either in simulation or paired with a hardware counterpart, enabling end-to-end workflows across simulation and hardware:

- **[Large-batch RL training on GPU](mujoco_playground/rl/rsl_rl)**: with state and vision observations across control, manipulation, and humanoid whole-body tracking.
- **[Sim & real teleoperation](mujoco_playground/tasks/teleop/so101)**: with leader-follower control and episodic MCAP logging.
- **[System identification](mujoco_playground/tasks/sysid)**: fitting simulated joint dynamics to logged hardware trajectories via `mujoco.sysid`.
- **[Zero-shot sim-to-real policy deployment](mujoco_playground/tasks/manipulation/pick_cube/so101/deploy)**: running the same task layer and policy in simulation (`--mode=sim`) and on hardware (`--mode=real`).
- **[Agentic manipulation](mujoco_playground/tasks/agentic/tabletop_v0)**: for vision-language model (VLM) examples across simulation and hardware (`SO-101`, `i2rt YAM`).

<table>
  <tr>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/whole_body/tracking"><img width="420" src="https://www.gstatic.com/reaf-sim/demos/g1_lafan_dance/0/g1_lafan_dance.webp" alt="Whole body motion tracking"></a>
      <br><b><a href="mujoco_playground/tasks/whole_body/tracking">Whole body motion tracking</a></b>
    </td>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/teleop/so101"><img width="420" src="https://www.gstatic.com/reaf-sim/demos/so101_sim_teleop/0/so101_sim_teleop.webp" alt="SO-101 Simulation Teleoperation Demo"></a>
      <br><b><a href="mujoco_playground/tasks/teleop/so101">Sim teleoperation</a></b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/teleop/so101"><img width="420" src="https://www.gstatic.com/reaf-sim/demos/so101_real_teleop/0/so101_real_teleop.webp" alt="SO-101 Real Teleoperation Demo"></a>
      <br><b><a href="mujoco_playground/tasks/teleop/so101">Real teleoperation</a></b>
    </td>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/sysid/so101"><img width="420" src="https://www.gstatic.com/reaf-sim/demos/so101_sysid/0/so101_sysid.webp" alt="SO-101 System Identification Rollout"></a>
      <br><b><a href="mujoco_playground/tasks/sysid/so101">System identification</a></b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/manipulation/pick_cube/so101/deploy"><img width="420" src="https://www.gstatic.com/reaf-sim/demos/so101_pick_cube_sim2real/1/so101_pick_cube_sim2real.webp" alt="Sim2Real and interactive viewer"></a>
      <br><b><a href="mujoco_playground/tasks/manipulation/pick_cube/so101/deploy">Sim2Real and interactive viewer</a></b>
    </td>
    <td align="center" width="50%">
      <a href="mujoco_playground/tasks/agentic/tabletop_v0"><img width="420" src="https://www.gstatic.com/reaf-sim/demos/yam_mug_real/0/yam_mug_real.webp" alt="YAM Agentic Tabletop Manipulation"></a>
      <br><b><a href="mujoco_playground/tasks/agentic/tabletop_v0">Agentic manipulation</a></b>
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
> Real-robot hardware SDKs are optional extras so simulation-only stays lightweight. To install hardware drivers for a specific robot, add `--extra so101` or `--extra yam` (e.g., `uv sync --extra cpu --extra so101`).

## Getting Started

To get started, either try training an RL policy, or try agentic manipulation in simulation or on the real hardware.

### Train & Visualize an RL Policy

Train a PPO policy across parallel GPU environments with [RSL-RL](https://github.com/leggedrobotics/rsl_rl):

```bash
# Train SO-101 Pick-Cube on GPU (on CPU, add --overrides='{"nworld": 64}')
uv run --no-sync python -m mujoco_playground.rl.rsl_rl.train \
  --env=SO101PickCubeVision \
  --logdir=/tmp/rslrl_logs/so101_pick_cube_vision

# Play back the trained policy in the interactive viewer
uv run --no-sync python -m mujoco_playground.rl.rsl_rl.play \
  --env=SO101PickCubeVision \
  --resume_path=/tmp/rslrl_logs/so101_pick_cube_vision
```

Monitor training curves in TensorBoard:

```bash
uv run --no-sync tensorboard --logdir=/tmp/rslrl_logs/so101_pick_cube_vision
```

Optionally [set up the SO-101 hardware](mujoco_playground/robots/so101/real) and [deploy to the real arm](mujoco_playground/tasks/manipulation/pick_cube/so101/deploy):

```bash
# Deploy the trained policy on the physical SO-101 follower arm
uv run --extra so101 --extra rl python -m \
  mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
  --mode=real \
  --checkpoint=/tmp/rslrl_logs/so101_pick_cube_vision
```

### Try Agentic Manipulation

Spin up the [agentic tabletop manipulation task](mujoco_playground/tasks/agentic/tabletop_v0) directly from your coding agent with a prompt in `sim` or `real`:

> "Start the tabletop_v0 bridge for the yam in sim and get it to \<your task\>, following [`mujoco_playground/tasks/agentic/tabletop_v0/AGENT_PROMPT.md`](mujoco_playground/tasks/agentic/tabletop_v0/AGENT_PROMPT.md)"
>
> *(Swap `sim` for `real` to run on hardware, or `so101` for `yam`.)*

> [!NOTE]
> On macOS, `--viewer` requires `mjpython` in place of `python`. Before the
> first run, link `libpython` into the virtualenv so `mjpython` can load it:
>
> ```bash
> ln -sf "$(uv run python -c 'import sys; print(sys.base_prefix)')"/lib/libpython3.13.dylib .venv/lib/
> ```

## Documentation & Contributing

- **Documentation:** See the [MuJoCo Playground documentation](https://reaf-sim.readthedocs.io/en/latest/playground/index.html).
- **Contributing:** See the [contributing guide](https://reaf-sim.readthedocs.io/en/latest/playground/contributing.html) for contribution guidelines.
- **Changelog:** See the [changelog](https://reaf-sim.readthedocs.io/en/latest/playground/changelog.html).

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

For the RSS 2025 benchmark suite that was built on JAX/MJX (see [`mujoco_playground/rss2025`](mujoco_playground/rss2025)), please cite:

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
