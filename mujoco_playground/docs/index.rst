Overview
========

MuJoCo Playground is the robot and RL task suite built on `reaf-sim`, providing
ready-to-run robot models, environments, RSL-RL training scripts, and
hardware workflows.

Package layout
--------------

`mujoco_playground` is organized into `robots/` (per-robot `sim/` and `real/`
modules), `tasks/` (environment configs, teleop, sysid, and deployment), `rl/`
(RSL-RL PPO training and evaluation), `real/` (shared hardware utilities), and
`profiling/` (benchmarks and Tracy profiling).

Installation
------------

Playground uses `uv <https://docs.astral.sh/uv/>`_ with optional dependency
extras (`make sync` for GPU training, `make sync-cpu` for CPU evaluation, and
`--extra so101` or `--extra yam` for robot hardware drivers).
