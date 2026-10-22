# SO101 Pick-Cube Sim2Real Training & Deployment

This directory implements sim2real policy deployment for the SO101 pick-cube task (`SO101PickCubeVision`), supporting both single-world simulation (`--mode=sim`) and the physical follower arm (`--mode=real`).

<p align="center">
  <img width="640" src="https://www.gstatic.com/mujoco/playground/demos/so101_pick_cube_sim2real/2/so101_pick_cube_sim2real.webp" alt="SO-101 Pick-Cube Sim-to-Real Deployment">
</p>

## Setup

Sync both extras before training or deploying:

```bash
uv sync --extra so101 --extra rl
```

For real hardware deployment, ensure the SO101 follower arm and wrist camera are identified and calibrated first—see [the SO101 hardware setup](../../../../../robots/so101/real/README.md).

---

## 1. Train the Policy

Train the vision-based pick-cube policy (`SO101PickCubeVision`) in simulation using RSL-RL (best done on a CUDA GPU):

```bash
uv run --extra rl python -m mujoco_playground.rl.rsl_rl.train \
    --env=SO101PickCubeVision \
    --logdir=/tmp/rslrl_logs/so101_pick_cube_vision
```

Checkpoints (`model_*.pt`) are written to `--logdir`. You can preview the trained policy in the batched training environment with:

```bash
uv run --extra rl python -m mujoco_playground.rl.rsl_rl.play \
    --env=SO101PickCubeVision \
    --resume_path=/tmp/rslrl_logs/so101_pick_cube_vision
```

---

## 2. Deploy in Simulation (`--mode=sim`)

Test the deployment pipeline, observation adapter, and interactive viewer in single-world simulation before running on hardware. Pass either a specific `model_*.pt` file or a run directory (which automatically loads the latest checkpoint) to `--checkpoint`:

```bash
uv run --no-sync --extra so101 --extra rl python -m \
    mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
    --mode=sim \
    --impl=cpp \
    --checkpoint=/tmp/rslrl_logs/so101_pick_cube_vision
```

---

## 3. Deploy on Real Hardware (`--mode=real`)

Deploy the trained checkpoint onto the physical SO101 follower arm and wrist camera:

```bash
uv run --no-sync --extra so101 --extra rl python -m \
    mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
    --mode=real \
    --checkpoint=/tmp/rslrl_logs/so101_pick_cube_vision
```

### Hardware Behavior & Useful Flags

* **Homing & Shutdown**: At the start of every episode, the real arm smoothly cosine-interpolates to the training hover pose (`--homing_duration_secs=2.0`). When the session ends (`Ctrl-C` or `--num_episodes`), the arm safely parks at its rest pose before torque is released.
* **Calibration Offsets & Domain Randomization**: Manual middle-pose calibration on the physical SO101 arm can easily be off by 1–2+ cm in end-effector height (e.g., hovering around 8 cm vs. 10 cm at the initial home pose). `SO101PickCubeVision` trains with per-episode calibration bias randomization (a few millimeters in position and small joint angle offsets) on `joint_pos`, `ee_pos`, and `robot_base_pos` height to tolerate moderate calibration error, but larger offsets will eventually degrade grasp success or cause the robot to slam the table—recalibrate the follower arm if it consistently reaches too low or hovers too high above the table.
* **`--target_pos=0.25,0.0,0.15`**: Goal 3D lift position in the arm's base frame (can also be adjusted live by dragging the target mocap in the viewer).
* **`--gfx=web`**: Graphics mode for the Studio viewer (`"web"` opens in the
  browser by default; `""` uses the native window if available).
* **`--action_scale=1.0`**: Multiplier applied to policy action outputs before sending to the robot.
* **`--episode_length_secs=20.0`**: Duration of each episode before automatic reset.
* **`--num_episodes=5`**: Number of episodes to run when `--noviewer` (defaults
  to running until `Ctrl-C`).

---

## 4. Inspect Episode Logs

By default, every deployment episode (sim or real) is recorded as an MCAP file under `.local/so101_sim2real_deploy_logs/YYYY/MM/DD/` (override with `--mcap_log_dir=<path>`, or pass `--mcap_log_dir=""` to disable logging).

Replay the latest logged episode in [Rerun](https://rerun.io):

```bash
# View most recent episode across .local/
uv run --extra real python -m mujoco_playground.real.view_log

# View a specific MCAP file
uv run --extra real python -m mujoco_playground.real.view_log \
    --file=.local/so101_sim2real_deploy_logs/YYYY/MM/DD/episode_*.mcap
```
