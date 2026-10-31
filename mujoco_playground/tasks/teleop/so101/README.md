# SO101 teleoperation

This README shows how to collect demonstrations by driving an SO101 follower arm
from an SO101 leader arm, in simulation or on real hardware. Both modes run the same
runloop, leader agent, foot-pedal gating and episodic logger, so the episodes
they produce are interchangeable.

<table>
  <tr>
    <td align="center" width="50%">
      <img width="420" src="https://www.gstatic.com/reaf-sim/demos/so101_sim_teleop/0/so101_sim_teleop.webp" alt="SO-101 Simulation Teleoperation Demo">
      <br><b>Sim Teleop (<code>--sim</code>)</b>
    </td>
    <td align="center" width="50%">
      <img width="420" src="https://www.gstatic.com/reaf-sim/demos/so101_real_teleop/0/so101_real_teleop.webp" alt="SO-101 Real Teleoperation Demo">
      <br><b>Real Teleop</b>
    </td>
  </tr>
</table>

## Before you start

We assume the arms are already identified and calibrated. Work through
[the SO101 hardware setup](../../../robots/so101/real/README.md) first to identify
and calibrate the hardware.

## Running

### Sim

Start in simulation. The real follower is not involved, so it is the safe way
to check the leader, the pedal and the logger before anything can collide:

```bash
uv run --extra so101 python -m mujoco_playground.tasks.teleop.so101.teleop --sim
```

On **macOS** the viewer must run under `mjpython` (Apple requires the GUI to
own the main thread).

```bash
# one-time fix for mjpython on mac
ln -sf "$(uv run python -c 'import sys; print(sys.base_prefix)')"/lib/libpython3.13.dylib .venv/lib/
# run sim teleop
uv run --extra so101 mjpython -m mujoco_playground.tasks.teleop.so101.teleop --sim
```

### Real

For real teleop:

```bash
uv run --extra so101 python -m mujoco_playground.tasks.teleop.so101.teleop
```

Episodes are gated by a three-button foot pedal — or its keyboard stand-in
(default keys `a`/`s`/`d`):

- **`a` (left)** — start the next episode; *during* one, end it as **failure**
- **`s` (middle)** — end the episode as **invalid** (discard)
- **`d` (right)** — end the episode as **success**

Keys are read from the terminal. The session runs until Ctrl-C, or `--num_episodes`.
On exit the follower is sent home and given a moment to arrive before torque
disengages.

**Passive (Torqueless) Leader Arm**

If using a passive (torqueless) leader arm, pass `--noreset_leader`:

```bash
uv run --extra so101 python -m mujoco_playground.tasks.teleop.so101.teleop --noreset_leader
```

Without `--noreset_leader`, the script attempts to torque-drive the leader arm
to the home pose before each episode. On a torqueless leader arm, the motors
overdrive without being able to move the arm itself, which can lead to hardware
malfunctioning. Since a torqueless leader arm will naturally be at a position
similar to the home position after each operation anyway, `--noreset_leader`
should be okay for most use cases.


## Episode logs

Every episode — sim and real, same schema — is one MCAP file under the
machine-local `.local/so101_teleop_logs/YYYY/MM/DD/` (`--output_dir`
overrides). Each step carries the per-joint follower positions, the
wrist-camera frame (jpeg), the commanded action, the raw (unclipped) leader
pose, and a wall-clock timestamp. Inspect one in [rerun](https://rerun.io):

```bash
uv run --extra real python -m mujoco_playground.real.view_log            # latest episode
uv run --extra real python -m mujoco_playground.real.view_log --file=... # specific file
```


The viewer reads the shapes it needs out of the log itself, so it works on any
SO101 episode — including the ones a policy deployment writes under
`--mcap_log_dir`, whose observations look nothing like these.
