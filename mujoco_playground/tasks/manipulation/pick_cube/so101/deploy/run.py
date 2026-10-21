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
"""

from collections.abc import Callable
from collections.abc import Mapping
import copy
import dataclasses
import os
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
    "A model_*.pt checkpoint, or a run directory to take the latest from."
    " Without one the policy commands zero actions, holding the hover pose.",
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


def resolve_checkpoint(checkpoint: str | None) -> epath.Path | None:
  """Expands `--checkpoint` and checks it exists, touching no hardware."""
  if checkpoint is None:
    return None
  path = epath.Path(checkpoint).expanduser()
  if not path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {path}")
  return path


def load_actor(
    env: torch_environment.Environment,
    checkpoint: epath.Path | str,
    env_id: str = "SO101PickCubeVision",
) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
  """Loads the trained actor from a checkpoint as an observations -> action map."""
  from mujoco_playground._src.rl.rsl_rl import checkpoint as checkpoint_lib  # pylint: disable=g-import-not-at-top
  from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper  # pylint: disable=g-import-not-at-top
  from rsl_rl import runners  # pylint: disable=g-import-not-at-top
  import tensordict  # pylint: disable=g-import-not-at-top

  device = env.xpu_device
  entry = rl.env_registry.get(env_id)
  runner_cfg = copy.deepcopy(entry.runner_cfg)
  runner_cfg.nworld = 1
  # Deployment observes no privileged state, so the critic — which is not
  # loaded, and not run — is pointed at the policy's own observations, but unused.
  runner_cfg.obs_groups = {"actor": ["policy", "camera"], "critic": ["policy"]}

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
  actor = runner.get_inference_policy(device=device)

  return lambda obs: actor(
      tensordict.TensorDict(dict(obs), batch_size=[1])  # pyrefly: ignore[bad-argument-type, bad-specialization]
  )


def create_policy(
    env: torch_environment.Environment,
    checkpoint: epath.Path | None,
    action_scale: float = 1.0,
) -> TorchPolicy:
  """The trained actor, or a zero-action stand-in when no checkpoint is given."""
  action_spec = env.action_spec()
  assert isinstance(action_spec, Mapping)
  if checkpoint is None:
    print("No checkpoint: commanding zero actions.")
    (spec,) = action_spec.values()
    zeros = torch.zeros(spec.shape, device=env.xpu_device)
    network = lambda obs: zeros
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

  env = create_env(control_freq_hz=_CONTROL_FREQ_HZ.value)
  policy = create_policy(env, checkpoint, action_scale=_ACTION_SCALE.value)
  log_dir = _resolve_log_dir()
  loggers = [create_logger(env, policy)] if log_dir is not None else []

  if _VIEWER.value:
    deploy_viewer.launch(env, policy, loggers=loggers, gfx=_GFX.value)
  else:
    operations: list[runloop_lib.RunloopRuntimeOperations] = []
    if _MODE.value == "sim":
      operations.append(RateLimiterOperation(env.device_layer.control_timestep))
    runloop = runloop_lib.Runloop(
        environment=env,
        policy=policy,
        loggers=loggers,
        handle_sigint=True,
        runloop_runtime_operations=operations,
    )
    runloop.run(num_episodes=_NUM_EPISODES.value)

  for closeable in (*loggers, env):
    closeable.close()


if __name__ == "__main__":
  app.run(main)
