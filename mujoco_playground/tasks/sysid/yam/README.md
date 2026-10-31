# YAM system identification

This directory contains system identification tools to tune the simulated I2RT
YAM Pro joint dynamics (`armature`, `frictionloss`, `kp`, `kv`) against
hardware trajectories using `mujoco.sysid`.

Unlike the SO-101 workflow (which fits from teleoperation logs), the YAM
workflow plays three scripted 50 Hz excitation trajectories through the I2RT
SDK. Every trajectory is first verified in simulation before moving the arm on
hardware.

<p align="center">
  <img width="640" src="https://www.gstatic.com/reaf-sim/demos/yam_sysid/0/yam_sysid.webp" alt="YAM System Identification Rollout Comparison">
</p>

## 1. Collect excitation trajectories

Set up the arm and bring up the SocketCAN interface as described in
[`mujoco_playground/robots/yam/README.md`](../../../robots/yam/README.md).
Clear the desk around the arm and start near `REST_POSE` (all joints at zero):

```bash
uv run --no-sync --extra yam python -m mujoco_playground.tasks.sysid.yam.collect
```

`collect` runs three stages:

1. **Generate 50 Hz excitation episodes (`excitation.py`)**: Each episode
   ramps from `REST_POSE` to a raised center pose along a minimum-jerk profile,
   executes a $C^2$-tapered excitation window, and ramps back to rest:

   | Episode | Excitation profile | Role |
   | --- | --- | --- |
   | `ep01_multisine_all_joints` | Simultaneous 3-sine sum per joint (`0.15–1.05 Hz`) | Fit (coupled inertia, armature, servo lag) |
   | `ep02_joint_chirp_and_reversals` | Single-joint slow reversal + `0.2–1.1 Hz` chirp | Fit (per-joint dry friction & damping) |
   | `ep03_heldout_validation` | Independent multisine + step doublets around a shifted center | Held-out validation |

2. **Simulation pre-flight check (`preflight.py`)**: Replays each episode
   kinematically and dynamically (`mj_step`) in the YAM desk scene and aborts
   before opening the CAN bus if the arm contacts the scene, comes within
   `15 cm` of the desk during active excitation, or violates joint-limit
   margins (`0.15 rad`).

   > [!IMPORTANT]
   > The default scene only models the desk. Add any workcell obstacles (camera
   > mounts, fixtures, objects) to `preflight.py` before running on hardware.

3. **Hardware execution**: Sends 50 Hz position commands to the I2RT SDK's
   250 Hz PD + gravity-compensation controller, aborting if any arm joint lags
   its command by more than `0.65 rad`. Logs measured positions, velocities,
   commands, and timestamps to `.local/yam_sysid_logs/ep0*.npz`.

## 2. Run system identification

Fit the joint parameters on `ep01` and `ep02` and evaluate on `ep03`:

```bash
uv run --no-sync --extra yam python -m mujoco_playground.tasks.sysid.yam.run
```

`run` performs the following steps:

1. **Optimize dynamics**: Fits `armature`, `frictionloss`, `kp`, and `kv` per
   actuator starting from `scene.yam_menagerie()`, aligning command and
   measurement timestamps.
   - `joint1_kp` is frozen at the SDK gain (`--freeze_kp=joint1`) because
     Joint 1 carries no gravity torque, so position tracking only constrains
     the ratio of `kp`, `kv`, `frictionloss`, and `armature`, not their absolute
     scale.
   - The gripper is frozen at nominal (`freeze_gripper=True`) because unloaded
     finger tracking is insensitive to its dynamics.
2. **Evaluate tracking RMSE**: Prints per-joint tracking RMSE (`mrad` for arm
   joints, `mm` for the finger) for the nominal and identified models on both
   the training and held-out validation splits.
3. **Flag unidentified parameters**: Estimates 95% confidence intervals from
   the Jacobian at the optimum and prints a `--freeze_params` flag listing any
   poorly identified parameters.

### Freezing unidentified parameters

A fit that reduces training error can still drift on parameters that the
excitation trajectories do not constrain well. A fitted parameter is flagged as
unidentified when:

- **At bound**: it converges within `0.1%` of its `[min, max]` box bounds.
- **Interval contains nominal**: its 95% confidence interval covers the nominal
  MJCF value (`|identified - nominal| <= ci95_half_width`).
- **Wide interval**: its 95% half-width exceeds 50% of its magnitude
  (`ci95_half_width > 0.5 * |identified|`).

Iteratively refit with the flagged parameters held at their nominal values by
passing the printed `--freeze_params` flag (or `--freeze_joints` for an entire
actuator):

```bash
uv run --no-sync --extra yam python -m mujoco_playground.tasks.sysid.yam.run \
  --freeze_params=joint2_armature,joint3_armature,joint4_armature,joint4_frictionloss,joint4_kp
```

Stop when no parameters are flagged and verify that held-out RMSE on `ep03`
does not degrade. Because time-series tracking residuals are autocorrelated,
Jacobian confidence intervals are optimistic—treat borderline parameters as
unidentified, and check `correlations_abs_ge_0p8` in `identified_params.json`
for strongly coupled pairs (such as a joint's `kp` and `kv`).

On a reference YAM Pro, this procedure identified friction and damping on
joints 1–5, `kp` on joints 2, 3, and 5, and `armature` on joints 1 and 5
(keeping the remaining parameters at Menagerie nominals), reducing held-out
tracking RMSE from `11.5 mrad` to `4.8 mrad`.

## 3. View the report

Open the diagnostic report in your browser:

```bash
open .local/yam_sysid_results/report.html
```

- `report.html`: Time-series tracking overlays (measured vs. simulated
  rollouts), parameter comparison tables, residual distributions, and
  covariance matrices.
- `identified_params.json`: Nominal and identified parameter values, 95%
  confidence half-widths, unidentified-parameter flags, strongly correlated
  parameter pairs (`|r| >= 0.8`), and per-episode tracking RMSE.

## 4. Update simulation constants

Copy the identified values into `ARMATURE`, `FRICTIONLOSS`, `STIFFNESS`, and
`DAMPING` in
[`mujoco_playground/robots/yam/sim/constant.py`](../../../robots/yam/sim/constant.py),
retaining the nominal value for any frozen parameter.

- `scene.yam_linear_4310()` (used by YAM simulation scenes and tasks) applies the
  identified constants in `sim/constant.py` on top of Menagerie's model.
- `scene.yam_menagerie()` preserves the unmodified Menagerie parameters so
  subsequent `sysid` runs always start from the same baseline.

> [!NOTE]
> The identified values represent the effective closed-loop dynamics of the
> 50 Hz command → 250 Hz I2RT SDK PD + gravity-compensation loop rather than
> bare motor settings: `kp` and `kv` are the MuJoCo position-actuator gains
> (`N·m/rad` and `N·m·s/rad`), `frictionloss` captures dry Coulomb friction
> (`N·m`), and velocity-dependent damping is absorbed into `kv`.
