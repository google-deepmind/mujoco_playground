# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs a trained SO101 pick-cube policy, in simulation or on the real arm.

  On hardware:
    uv run --extra so101 --extra rl python -m \
        mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
        --mode=real --checkpoint=/path/to/model_*.pt

  In simulation:
    uv run --extra so101 --extra rl python -m \
        mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
        --mode=sim --impl=cpp --checkpoint=/path/to/model_*.pt

  With a policy exported to ExecuTorch, which needs `executorch` instead of
  RSL-RL:
    uv run --extra so101 --with executorch==1.6.0 python -m \
        mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run \
        --mode=sim --impl=cpp --checkpoint=/path/to/policy.pte

  With a policy exported to ExecuTorch with RSL-RL's TensorRT backend, use
  `--with "torch_tensorrt[executorch]==2.15.0"` instead of
  `--with executorch==1.6.0`.
"""

from collections.abc import Callable
from collections.abc import Mapping
import contextlib
import copy
import dataclasses
import importlib.util
import os
import shutil
import tempfile
import time
from typing import Any

from absl import app
from absl import flags
import dm_env
from dm_env import specs
from etils import epath
from gdm_robotics.interfaces import policy as policy_lib
from gdm_robotics.runtime import runloop as runloop_lib
from mujoco_playground import paths
from mujoco_playground import real
from mujoco_playground import rl
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.real import device as real_device
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import env as pick_cube_env
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import viewer as deploy_viewer
from reaf.simulation.torch import environment as torch_environment
import torch

_MODE = flags.DEFINE_enum(
    "mode", "sim", ["sim", "real"], "Where to run the policy."
)
_IMPL = flags.DEFINE_enum(
    "impl", "cpp", ["cpp", "warp"], "Simulation backend, when --mode=sim."
)
_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    None,
    "A model_*.pt checkpoint, a run directory to take the latest from, or a"
    " policy.pte exported with RSL-RL's export_policy_to_executorch(). Without"
    " one the policy commands zero actions, holding the hover pose.",
)
_EPISODE_LENGTH_SECS = flags.DEFINE_float(
    "episode_length_secs", 20.0, "Episode length, in seconds."
)
_NUM_EPISODES = flags.DEFINE_integer(
    "num_episodes",
    None,
    "Episodes to run when --noviewer; unset runs until Ctrl-C.",
)
_TARGET_POS = flags.DEFINE_list(
    "target_pos",
    ["0.25", "0.0", "0.15"],
    "Where to lift the cube to, in the arm's base frame, when --mode=real.",
)
_HOMING_DURATION_SECS = flags.DEFINE_float(
    "homing_duration_secs",
    2.0,
    "How long the real arm takes to reach its start pose, in seconds.",
)


def default_log_dir() -> epath.Path:
  """Returns the default directory for sim2real deploy logs."""
  return paths.local_dir() / f"{constant.ROBOT}_sim2real_deploy_logs"


_LOG_DIR = flags.DEFINE_string(
    "mcap_log_dir",
    None,
    "Where to write episode MCAP logs; pass empty string to disable.",
)
_CONTROL_FREQ_HZ = flags.DEFINE_float(
    "control_freq_hz",
    None,
    "Control frequency in Hz sent to the robot or sim (e.g. 50.0). "
    "Unset uses the default environment rate.",
)
_ACTION_SCALE = flags.DEFINE_float(
    "action_scale",
    1.0,
    "Multiplier applied to policy action outputs before sending to the"
    " environment.",
)
_CAMERA_WIDTH = flags.DEFINE_integer(
    "camera_width",
    constant.CAMERA_WIDTH,
    "Capture width in pixels for the real wrist camera. Defaults to"
    f" {constant.CAMERA_WIDTH}.",
)
_CAMERA_HEIGHT = flags.DEFINE_integer(
    "camera_height",
    constant.CAMERA_HEIGHT,
    "Capture height in pixels for the real wrist camera. Defaults to"
    f" {constant.CAMERA_HEIGHT}.",
)
_VIEWER = flags.DEFINE_bool(
    "viewer", True, "Whether to show the deployment in an interactive viewer."
)
_GFX = flags.DEFINE_string(
    "gfx", "web", 'Graphics mode ("" for native, "web" for browser).'
)


class TorchPolicy(policy_lib.Policy[Any]):
  """Runs a torch network against the environment's single-world batch."""

  def __init__(
      self,
      network: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
      action_spec: Mapping[str, specs.Array],
      action_scale: float = 1.0,
  ):
    self._network = network
    self._action_spec = action_spec
    self._action_scale = action_scale
    (self._action_key,) = action_spec

  def initial_state(self) -> Any:
    return ()

  def step_spec(self, timestep_spec: Any) -> Any:
    del timestep_spec
    return (self._action_spec, {}), ()

  def action_spec(self) -> Mapping[str, specs.Array]:
    return self._action_spec

  def policy_extra_spec(self) -> Mapping[str, specs.Array]:
    return {}

  @torch.no_grad()
  def step(
      self, timestep: dm_env.TimeStep, prev_state: Any = ()
  ) -> tuple[tuple[Any, Mapping[str, Any]], Any]:
    action = self._network(timestep.observation)
    if self._action_scale != 1.0:
      action = action * self._action_scale
    return ({self._action_key: action}, {}), prev_state


def create_env(
    *,
    control_freq_hz: float | None = None,
) -> torch_environment.Environment:
  """Builds the deployment environment for the requested mode."""
  if _MODE.value == "sim":
    return pick_cube_env.create_sim_env(
        impl=_IMPL.value,  # pyrefly: ignore[bad-argument-type]
        episode_length_secs=_EPISODE_LENGTH_SECS.value,
        control_freq_hz=control_freq_hz,
    )
  return pick_cube_env.create_real_env(
      target_pos=[float(x) for x in _TARGET_POS.value],
      episode_length_secs=_EPISODE_LENGTH_SECS.value,
      homing_duration_secs=_HOMING_DURATION_SECS.value,
      control_freq_hz=control_freq_hz,
      camera_config=real_device.CameraConfig(
          width=_CAMERA_WIDTH.value,
          height=_CAMERA_HEIGHT.value,
      ),
  )


_MISSING_EXECUTORCH = (
    "executorch is required to run a .pte policy. Add `--with"
    " executorch==1.6.0` to `uv run`, or `--with"
    ' "torch_tensorrt[executorch]==2.15.0"` for a TensorRT program.'
)


def resolve_checkpoint(checkpoint: str | None) -> epath.Path | None:
  """Expands `--checkpoint` and checks it exists, touching no hardware."""
  if checkpoint is None:
    return None
  path = epath.Path(checkpoint).expanduser()
  if not path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {path}")
  # executorch is in no extra, so report it missing before the arm moves.
  if (
      path.suffix == ".pte"
      and path.is_file()
      and importlib.util.find_spec("executorch") is None
  ):
    raise ImportError(_MISSING_EXECUTORCH)
  return path


def load_runner(
    env: torch_environment.Environment,
    checkpoint: epath.Path | str,
    env_id: str = "SO101PickCubeVision",
) -> Any:
  """Builds the RSL-RL runner with the trained actor loaded from a checkpoint."""
  from mujoco_playground._src.rl.rsl_rl import checkpoint as checkpoint_lib  # pylint: disable=g-import-not-at-top
  from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper  # pylint: disable=g-import-not-at-top
  from rsl_rl import runners  # pylint: disable=g-import-not-at-top

  device = env.xpu_device
  entry = rl.env_registry.get(env_id)
  runner_cfg = copy.deepcopy(entry.runner_cfg)
  runner_cfg.nworld = 1
  # Deployment observes no privileged state, so the critic — which is not
  # loaded, and not run — is pointed at the policy's own observations, but unused.
  runner_cfg.obs_groups = {
      "actor": list(pick_cube_env.OBSERVATION_GROUPS),
      "critic": ["policy"],
  }

  runner = runners.OnPolicyRunner(
      vecenv_wrapper.RslRlVecEnvWrapper(env, cfg=runner_cfg.vecenv_wrapper_cfg),
      dataclasses.asdict(runner_cfg),
      device=device,
  )
  checkpoint_path = os.path.expanduser(str(checkpoint))
  if os.path.isdir(checkpoint_path):
    resolved = checkpoint_lib.get_latest_checkpoint(checkpoint_path)
    if resolved is None:
      raise FileNotFoundError(f"No checkpoint found in: {checkpoint_path}")
    checkpoint_path = resolved
  runner.load(checkpoint_path, load_cfg={"actor": True}, map_location=device)
  return runner


def load_actor(
    env: torch_environment.Environment,
    checkpoint: epath.Path | str,
    env_id: str = "SO101PickCubeVision",
) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
  """Loads the trained actor from a checkpoint as an observations -> action map."""
  import tensordict  # pylint: disable=g-import-not-at-top

  runner = load_runner(env, checkpoint, env_id)
  actor = runner.get_inference_policy(device=env.xpu_device)
  return lambda obs: actor(
      tensordict.TensorDict(dict(obs), batch_size=[1])  # pyrefly: ignore[bad-argument-type, bad-specialization]
  )


def load_executorch_program(program_path: epath.Path | str) -> Any:
  """Loads a program exported to ExecuTorch and its `forward` method."""
  try:
    from executorch.runtime import Runtime  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top
  except ImportError as exc:
    raise ImportError(_MISSING_EXECUTORCH) from exc
  # Importing it adds the TensorRT backend to the runtime, which programs
  # exported with RSL-RL's backend="tensorrt" need. Other programs load without
  # it, so a failed import only shows up if this program fails to load.
  try:
    import torch_tensorrt_executorch_runtime  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top,unused-import

    tensorrt_hint = (
        "If it is a TensorRT export, it loads only on the same kind of GPU"
        " with the same TensorRT and Torch-TensorRT versions that exported it."
    )
  except ImportError as exc:
    if (
        isinstance(exc, ModuleNotFoundError)
        and exc.name == "torch_tensorrt_executorch_runtime"
    ):
      tensorrt_hint = (
          "If it is a TensorRT export, use `--with"
          ' "torch_tensorrt[executorch]==2.15.0"` instead of `--with'
          " executorch==1.6.0`."
      )
    else:
      tensorrt_hint = (
          f"If it is a TensorRT export, its backend failed to load: {exc}"
      )

  program_path = epath.Path(program_path).expanduser()
  # RSL-RL's CUDA export keeps the weights in a .ptd file next to the program.
  data_path = program_path.with_suffix(".ptd")
  has_weights = data_path.exists()
  weights = f" with {data_path.name}" if has_weights else ""
  # The runtime gets the program in memory and a private copy of the weights,
  # which load_program maps, so a new export replacing the files cannot crash a
  # run. The mapping keeps the copy readable after the directory is deleted. On
  # NFS the delete can leave a placeholder file behind, so cleanup errors are
  # ignored.
  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as copy_dir:
    try:
      program = Runtime.get().load_program(
          program_path.read_bytes(),
          data_path=shutil.copy(data_path, copy_dir) if has_weights else None,
      )
    except RuntimeError as exc:
      raise ValueError(
          f"Could not load {program_path}{weights}: {exc}"
      ) from exc
    for method in ("forward", "input_names"):
      if method not in program.method_names:
        raise ValueError(
            f"{program_path} has no {method} method. Export it with RSL-RL's"
            " export_policy_to_executorch()."
        )
    try:
      program.load_method("forward")
    except RuntimeError as exc:
      hint = (
          ""
          if has_weights
          else (
              f" If it is a CUDA export, its weights file, {data_path.name},"
              " has to be next to it."
          )
      )
      raise ValueError(
          f"Could not load {program_path}{weights}: {exc}.{hint}"
          f" {tensorrt_hint}"
      ) from exc
  return program


def load_executorch_actor(
    env: torch_environment.Environment,
    program_path: epath.Path | str,
    program: Any = None,
) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
  """Loads an actor exported to ExecuTorch as an observations -> action map."""
  if program is None:
    program = load_executorch_program(program_path)
  forward = program.load_method("forward")
  # RSL-RL's export takes the actor's 1D observation groups concatenated in the
  # order of OBSERVATION_GROUPS, which load_runner gives it, then one input per
  # image group. Its `input_names` method names the inputs, the 1D one first.
  _, *image_groups = program.load_method("input_names").execute([])
  vector_groups = [
      group
      for group in pick_cube_env.OBSERVATION_GROUPS
      if group not in image_groups
  ]
  device = env.xpu_device

  def network(obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    inputs = [torch.cat([obs[group] for group in vector_groups], dim=-1)]
    inputs += [obs[group] for group in image_groups]
    (action,) = forward.execute([x.cpu().contiguous() for x in inputs])
    return action.to(device)

  # Run once on zero observations and check the action shape, so a program
  # whose shapes do not fit this env fails here instead of after the arm moves.
  observation_spec = env.timestep_spec().observation
  assert isinstance(observation_spec, Mapping)
  action_spec = env.action_spec()
  assert isinstance(action_spec, Mapping)
  (spec,) = action_spec.values()
  try:
    action = network({
        group: torch.from_numpy(observation_spec[group].generate_value()).to(device)  # pyrefly: ignore[missing-attribute]
        for group in pick_cube_env.OBSERVATION_GROUPS
    })
  except (KeyError, RuntimeError, ValueError) as exc:
    raise ValueError(
        f"{program_path} failed on zero observations of this env: {exc}. Was"
        " it exported for this env?"
    ) from exc
  if action.shape != spec.shape:
    raise ValueError(
        f"{program_path} returns actions of shape {tuple(action.shape)}, but"
        f" this env takes {tuple(spec.shape)}."
    )
  return network


def create_policy(
    env: torch_environment.Environment,
    checkpoint: epath.Path | None,
    action_scale: float = 1.0,
    program: Any = None,
) -> TorchPolicy:
  """The trained actor, or a zero-action stand-in when no checkpoint is given."""
  action_spec = env.action_spec()
  assert isinstance(action_spec, Mapping)
  if checkpoint is None:
    print("No checkpoint: commanding zero actions.")
    (spec,) = action_spec.values()
    zeros = torch.zeros(spec.shape, device=env.xpu_device)
    network = lambda obs: zeros
  elif program is not None or (
      checkpoint.suffix == ".pte" and checkpoint.is_file()
  ):
    network = load_executorch_actor(env, checkpoint, program)
  else:
    network = load_actor(env, checkpoint)
  return TorchPolicy(network, action_spec, action_scale=action_scale)


def _resolve_log_dir() -> epath.Path | None:
  if _LOG_DIR.value == "":
    return None
  if _LOG_DIR.value is not None:
    return epath.Path(_LOG_DIR.value).expanduser()
  return default_log_dir()


def create_logger(
    env: torch_environment.Environment, policy: TorchPolicy
) -> Any:
  """The episodic MCAP logger, writing under `--mcap_log_dir`."""
  log_dir = _resolve_log_dir()
  assert log_dir is not None
  return real.create_logger(
      timestep_spec=env.timestep_spec(),
      action_spec=policy.action_spec(),
      policy_extra_spec=policy.policy_extra_spec(),
      image_observation_keys=[],
      timestamp_key=constant.TIMESTAMP,
      task_id="so101_pick_cube",
      agent_id=constant.ROBOT,
      output_directory=log_dir,
      control_timestep_seconds=env.device_layer.control_timestep,
  )


class RateLimiterOperation(runloop_lib.RunloopRuntimeOperations):
  """Paces the simulation loop to match real-time control frequency."""

  def __init__(self, period_seconds: float):
    self._period_seconds = period_seconds
    self._last_time = time.perf_counter()

  def before_episode_reset(self) -> bool:
    self._last_time = time.perf_counter()
    return True

  def inspect_timestep(self, timestep: dm_env.TimeStep) -> None:
    del timestep
    target_time = self._last_time + self._period_seconds
    time.sleep(max(0.0, target_time - time.perf_counter()))
    self._last_time = max(target_time, time.perf_counter())


def main(argv: Any) -> None:
  del argv

  # Resolve the checkpoint before any hardware is touched.
  checkpoint = resolve_checkpoint(_CHECKPOINT.value)
  # A program that cannot load, such as a TensorRT one without its runtime,
  # fails here, before the real arm is connected.
  program = (
      load_executorch_program(checkpoint)
      if checkpoint is not None
      and checkpoint.suffix == ".pte"
      and checkpoint.is_file()
      else None
  )

  env = create_env(control_freq_hz=_CONTROL_FREQ_HZ.value)
  # From here on, every exit closes the env, which parks and releases the real
  # arm. The stack closes the loggers first, and still closes the env if one of
  # them fails.
  with contextlib.ExitStack() as stack:
    stack.callback(env.close)
    policy = create_policy(
        env, checkpoint, action_scale=_ACTION_SCALE.value, program=program
    )
    loggers = []
    if _resolve_log_dir() is not None:
      loggers.append(
          stack.enter_context(contextlib.closing(create_logger(env, policy)))
      )

    if _VIEWER.value:
      deploy_viewer.launch(env, policy, loggers=loggers, gfx=_GFX.value)
    else:
      operations: list[runloop_lib.RunloopRuntimeOperations] = []
      if _MODE.value == "sim":
        operations.append(
            RateLimiterOperation(env.device_layer.control_timestep)
        )
      runloop = runloop_lib.Runloop(
          environment=env,
          policy=policy,
          loggers=loggers,
          handle_sigint=True,
          runloop_runtime_operations=operations,
      )
      runloop.run(num_episodes=_NUM_EPISODES.value)


if __name__ == "__main__":
  app.run(main)
