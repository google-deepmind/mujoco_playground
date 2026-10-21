# SO101 system identification

This directory contains system identification tools to tune the SO101 simulation
model against real robot hardware using `mujoco.sysid`.

The optimizer fits a 24-parameter model (`armature`, `frictionloss`, `stiffness`,
`damping` for all 6 joints) from teleoperation MCAP logs by minimizing rollout
tracking error in MuJoCo physics.

## 1. Collect excitation trajectory

Collect a teleoperation episode on the real robot that articulates all joints:

```bash
uv run --extra so101 python -m mujoco_playground.tasks.teleop.so101.teleop \
  --instruction="sysid_excitation" \
  --num_episodes=1
```

1. Press **`a`** (or left foot pedal) to start recording.
2. Move the leader arm to excite all joints:
   - Sweep all 6 joints through varying speeds and directional reversals.
   - Open and close the gripper multiple times.
3. Press **`d`** (or right foot pedal) to finish the episode as `success`.

## 2. Run system identification

Run the sys-id optimization on the collected teleoperation logs (defaults to reading `.local/so101_teleop_logs`):

```bash
uv run --extra so101 python -m mujoco_playground.tasks.sysid.so101.run
```

The script will:
1. Estimate and log the serial command transport delay ($t_1 - t_0$).
2. Align control times with measurement physics rollouts.
3. Run Gauss-Newton least squares optimization on the 24 dynamic parameters.
4. Generate an interactive HTML report.

## 3. View the report

Open the generated report in your browser:

```bash
open .local/so101_sysid_results/report.html
```

The report provides:
- Time-series tracking overlays (measured vs. simulated rollouts).
- Parameter comparison table (nominal vs. identified).
- Residual error distributions and covariance matrices.
