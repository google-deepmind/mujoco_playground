# Tabletop V0: Sim-to-Real VLM Robot Manipulation Demo

> [!CAUTION]
> **Use this code at your own risk.** The harness runs VLM-generated Python
> (`exec`) and lets the VLM drive the robot directly.

## Try It on Your Robot

Give your agent this prompt; it starts the bridge and drives the robot:

> "Start the tabletop_v0 bridge for the yam in sim and get it to &lt;your task&gt;,
> following `mujoco_playground/tasks/agentic/tabletop_v0/AGENT_PROMPT.md`"

Swap `sim` for `real` to run on hardware. The SO-101 can also be used (say
`so101` instead of `yam`); on hardware it requires port finding and robot
calibration first.

**Alternatively, start the bridge manually** and then point your agent at it
with the prompt above:

```bash
# Simulation
uv run --default-index https://pypi.org/simple --extra yam python -m mujoco_playground.tasks.agentic.tabletop_v0.run --mode=sim --robot=yam --agent=bridge --viewer

# Real hardware
uv run --default-index https://pypi.org/simple --extra yam python -m mujoco_playground.tasks.agentic.tabletop_v0.run --mode=real --robot=yam --agent=bridge
```

For the SO-101, use `--extra so101 --robot=so101`.

## About

This robot-agnostic sim/real harness was **written entirely by an agent** and
is provided as-is. The agent built and iterated on the IK controller, primitive
executor, camera calibration, and task suite in simulation with `reaf-sim`, then
the same stack was deployed zero-shot to real hardware (`--mode=sim` →
`--mode=real`). See the Google Developers blogpost for more details:
[Gemini 4 Argon for Robotics](https://developers.googleblog.com/gemini-4-argon-for-robotics).

<table>
  <tr>
    <td align="center" width="50%">
      <img width="420" src="https://www.gstatic.com/mujoco/playground/demos/yam_mug_real/2/yam_mug_real.webp" alt="YAM Agentic Real Hardware Manipulation">
      <br><b>YAM agentic manipulation (Real)</b>
    </td>
    <td align="center" width="50%">
      <img width="420" src="https://www.gstatic.com/mujoco/playground/demos/yam_pantry_spice/2/yam_pantry_spice.webp" alt="YAM Agentic Pantry Spices">
      <br><b>YAM agentic pantry spices (Sim)</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img width="420" src="https://www.gstatic.com/mujoco/playground/demos/yam_chess/2/yam_chess.webp" alt="YAM Agentic Chess Manipulation">
      <br><b>YAM agentic chess (Sim)</b>
    </td>
    <td align="center" width="50%">
      <img width="420" src="https://www.gstatic.com/mujoco/playground/demos/yam_rock_cairn/2/yam_rock_cairn.webp" alt="YAM Agentic Rock Balancing">
      <br><b>YAM agentic rock balancing (Sim)</b>
    </td>
  </tr>
</table>

## How It Works

Each turn, the bridge serves `obs.json` and an annotated wrist image
(`wrist_cam.png`); the VLM replies with a Python plan (`POST /plan`).

- **Metric overlay**: a 2 cm world-XY grid and `[ + ]` jaw reticle are drawn on
  the wrist image so the VLM can target world `(x, y)` or pixel `(u, v)`.
- **Code-as-policies primitives**: `look_at`, `grasp`, `place`, `goto`,
  `move_by`, `move_until_contact`, `sweep` compile to 50 Hz minimum-jerk
  trajectories solved by damped least-squares IK.
- **Sensorless contact and slip detection**: tracking lag and jaw aperture stop
  motion on contact and flag empty or slipped grasps.

## Supported Hardware

- **i2rt YAM** (`--robot=yam`): 6-DoF arm, wrist camera.
- **SO-101** (`--robot=so101`): 5-DoF arm, wrist camera.

## Adding New Hardware

Follow [`robots/yam`](../../../robots/yam/) or
[`robots/so101`](../../../robots/so101/) and provide:

1. An `MjSpec` scene with a `wrist_cam` and a `pinch` site between the
   fingertips (z along the fingers, x along the jaw-closing direction).
2. Follower devices in `sim/device.py` and `real/device.py`.
3. An `embodiment.Embodiment` entry in `tabletop_v0/embodiment.py`.
