# SO101 Pick-Cube Sim2Real Training & Deployment

Train a visual policy to pick up a cube in sim and deploy zero-shot onto the real robot.
<table>
  <tr>
    <td align="center" width="50%">
      <img width="420" src="https://www.gstatic.com/reaf-sim/demos/so101_pick_cube_sim2real/1/so101_pick_cube_sim2real.webp" alt="SO-101 Pick-Cube Sim-to-Real Deployment">
    </td>
    <td align="center" width="50%">
      <img width="420" src="https://www.gstatic.com/reaf-sim/demos/so101_pick_cube_sim2real2/1/so101_pick_cube_sim2real2.webp" alt="SO-101 Pick-Cube Sim-to-Real Deployment on a separately verified robot">
    </td>
  </tr>
</table>

## Setup

Sync both extras before training or deploying:

```bash
uv sync --extra so101 --extra rl
```

For real hardware deployment, ensure the SO101 follower arm and wrist camera are identified and calibrated first—see [the SO101 hardware setup](../../../../../robots/so101/real).

---

## 1. Train the Policy

Train the vision-based pick-cube policy in simulation using RSL-RL:

```bash
uv run --extra rl python -m mujoco_playground.rl.rsl_rl.train \
    --env=SO101PickCubeVision \
    --logdir=/tmp/rslrl_logs/so101_pick_cube_vision
    # --overrides='{"nworld": 64}'  # For CPU
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
uv run --extra so101 --extra rl python -m \
    mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
    --mode=sim \
    --impl=cpp \
    --checkpoint=/tmp/rslrl_logs/so101_pick_cube_vision
```

### Run the Policy with ExecuTorch

[ExecuTorch](https://docs.pytorch.org/executorch/stable/index.html) runs a policy exported to a `.pte` file, without RSL-RL. Export the actor with `export_policy_to_executorch()`, which is on RSL-RL's `extras` branch. That branch is installed from GitHub, not from PyPI:

```bash
uv run --extra rl --with executorch==1.6.0 --with "rsl-rl-lib @ git+https://github.com/leggedrobotics/rsl_rl.git@extras" python -c '
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import env, run
runner = run.load_runner(env.create_sim_env(), "/tmp/rslrl_logs/so101_pick_cube_vision")
from rsl_rl.utils import export_policy_to_executorch
export_policy_to_executorch(runner.alg.get_policy(), "/tmp/rslrl_logs/so101_pick_cube_vision")'
```

Import `rsl_rl.utils` after `load_runner()`. With the robot extras installed, `rsl_rl.utils` loads TensorFlow through TensorBoard, and TensorFlow aborts the process with an LLVM error if Triton loads after it. `load_runner()` imports tensordict, which loads Triton first.

Notes on both commands:

- With `backend="cuda"`, the export also writes a `.ptd` weights file. Keep it next to the `.pte`, with the same file name and a `.ptd` ending. A CUDA program only runs on a machine like the one that exported it, so export it there, or use the default XNNPACK backend.
- executorch 1.6.0 needs torch 2.15. `--with` adds both for that run only, without changing `.venv`.
- On a machine synced with `make sync-cu12` or `make sync-cpu`, add `--group cu12` or `--group cpu`, as to every `uv run`, so `.venv` keeps its own torch. The torch that `--with` brings there is a CUDA 13 build: on CPU-only Linux it downloads its NVIDIA libraries, and on a `cu12` machine (NVIDIA driver older than 580) it cannot use the GPU, so only the XNNPACK export runs.

Then pass the `.pte` file to `--checkpoint`:

```bash
uv run --extra so101 --with executorch==1.6.0 python -m \
    mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
    --mode=sim \
    --impl=cpp \
    --checkpoint=/tmp/rslrl_logs/so101_pick_cube_vision/policy.pte
```

With `backend="tensorrt"`, the program runs in [TensorRT](https://developer.nvidia.com/tensorrt) on an NVIDIA GPU. That backend comes to RSL-RL's `extras` branch with [leggedrobotics/rsl_rl#275](https://github.com/leggedrobotics/rsl_rl/pull/275). TensorRT needs the GPU, and the torch that `--with` installs is a CUDA 13 build, so this path does not work on a `cu12` machine either. TensorRT engines load only on the same kind of GPU with the same TensorRT and Torch-TensorRT versions, so export on the machine that runs the policy. For both steps, use `--with "torch_tensorrt[executorch]==2.15.0"` instead of `--with executorch==1.6.0`. It installs ExecuTorch together with the TensorRT backend, which `run.py` loads on its own:

```bash
uv run --extra rl --with "torch_tensorrt[executorch]==2.15.0" --with "rsl-rl-lib @ git+https://github.com/leggedrobotics/rsl_rl.git@extras" python -c '
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import env, run
runner = run.load_runner(env.create_sim_env(), "/tmp/rslrl_logs/so101_pick_cube_vision")
from rsl_rl.utils import export_policy_to_executorch
export_policy_to_executorch(runner.alg.get_policy(), "/tmp/rslrl_logs/so101_pick_cube_vision", backend="tensorrt")'
```

```bash
uv run --extra so101 --with "torch_tensorrt[executorch]==2.15.0" python -m \
    mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
    --mode=sim \
    --impl=cpp \
    --checkpoint=/tmp/rslrl_logs/so101_pick_cube_vision/policy.pte
```

---

## 3. Deploy on Real Hardware (`--mode=real`)

Deploy the trained checkpoint onto the physical SO101 follower arm and wrist camera:

```bash
uv run --extra so101 --extra rl python -m \
    mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
    --mode=real \
    --checkpoint=/tmp/rslrl_logs/so101_pick_cube_vision
```

### Hardware Behavior & Useful Flags

* **Homing & Shutdown**: At the start of every episode, the real arm smoothly cosine-interpolates to the training hover pose (`--homing_duration_secs=2.0`). When the session ends (`Ctrl-C`, `--num_episodes`, or an error after setup, such as a checkpoint that fails to load), the arm safely parks at its rest pose before torque is released.
* **Calibration Offsets & Domain Randomization**: Manual middle-pose calibration on the physical SO101 arm can easily be off by 1–2+ cm in end-effector height. `SO101PickCubeVision` trains with per-episode calibration bias randomization to tolerate moderate calibration error, but larger offsets will eventually degrade grasp success or cause the robot to slam the table— **recalibrate the follower arm if it consistently reaches too low or hovers too high above the table**.
* We tested on 3 different robot setups under various lighting conditions, but there may still be visual domain gap with darker shadows. If you see this, **make sure the lighting conditions prevent darker shadows**, or help submit a more robust domain randomization recipe for the pixels inputs!
* `--control_freq_hz` modifies the control frequency for smoother motions

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
