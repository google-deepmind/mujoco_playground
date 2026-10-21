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

"""Script to play RL agent with RSL-RL."""

import dataclasses
import json
from typing import Callable

from absl import app
from absl import flags
import dm_env
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl.rsl_rl import checkpoint
from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
from mujoco_playground._src.rl.rsl_rl.viewer.plugins import policy as policy_plugin
from reaf.simulation import torch as reaf_torch
from reaf.simulation.torch import viewer
from rsl_rl import runners as rls_rl_runners
import tensordict
import torch
import warp as wp

_RESUME_PATH = flags.DEFINE_string(
    "resume_path", None, "Path to the checkpoint to resume from."
)
_ENV_OVERRIDES = flags.DEFINE_string(
    "env_overrides",
    None,
    "JSON dict string for environment config overrides.",
)
_ENV = flags.DEFINE_string("env", "SO101PickCube", "Environment to play.")
_ENV_ID = flags.DEFINE_integer("env_id", 0, "Environment ID to play.")
_NWORLD = flags.DEFINE_integer("nworld", 1, "Number of worlds to play.")
_WP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "wp_kernel_cache_dir",
    "/tmp/wp_kernel_cache",
    "Path to the Warp kernel cache directory.",
)
_DEVICE = flags.DEFINE_string(
    "device",
    None,
    "Torch/sim device, e.g. cuda:0 or cpu (default: cuda:0 if available).",
)
_ZERO_ACTIONS = flags.DEFINE_bool(
    "zero_actions",
    False,
    "Ignore the policy and always send zero actions (useful to inspect the"
    " environment / reference motion without a trained agent).",
)
# TODO(taylorhowell, matijak): Consider a more user-friendly flag name than gfx
# (e.g. graphics_mode).
_GFX = flags.DEFINE_string(
    "gfx", "", 'Graphics mode ("" for native, "web" for browser).'
)
_PORT = flags.DEFINE_integer(
    "port", 0, "Web viewer port (0 picks first free port >= 8080)."
)
_HOT_RELOAD = flags.DEFINE_string(
    "hot_reload",
    "",
    "What is reloaded when modified. Options are '', 'plugins', 'all',"
    " 'Plugin1,Plugin2,...'. Default is '', meaning nothing.",
)


class _RslRlPolicy(gdmr_policy.Policy[torch.Tensor]):
  """Adapts an RSL-RL inference policy to the gdm_robotics policy interface."""

  def __init__(
      self,
      policy: Callable[..., torch.Tensor],
      action_spec: gdmr_types.ActionSpec,  # pyrefly: ignore[invalid-type-var]
      nworld: int,
  ):
    self._policy = policy
    self._action_key = next(iter(action_spec))  # pyrefly: ignore[bad-assignment]
    self._nworld = nworld

  def forward(self, *args, **kwargs) -> torch.Tensor:
    return self._policy(*args, **kwargs)

  def step(self, timestep: dm_env.TimeStep, prev_state=()):
    obs = timestep.observation
    if not isinstance(obs, dict):
      obs = {"policy": obs}
    action = self.forward(
        tensordict.TensorDict(
            obs, batch_size=[self._nworld]  # pyrefly: ignore[bad-argument-type]
        )
    )
    return ({self._action_key: action}, {}), prev_state

  def initial_state(self):
    if hasattr(self._policy, "reset"):
      self._policy.reset()
    return ()

  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec):
    del timestep_spec


def main(argv, device: str | None = None):
  del argv
  device = (
      device
      or _DEVICE.value
      or ("cuda:0" if torch.cuda.is_available() else "cpu")
  )
  if _WP_KERNEL_CACHE_DIR.value:
    wp.config.kernel_cache_dir = _WP_KERNEL_CACHE_DIR.value

  entry = env_registry.get(_ENV.value)
  agent_cfg = entry.runner_cfg
  agent_cfg.nworld = _NWORLD.value

  env_overrides = (
      json.loads(_ENV_OVERRIDES.value) if _ENV_OVERRIDES.value else {}
  )

  env = entry.env_builder(
      nworld=_NWORLD.value,
      device=device,
      **env_overrides,
  )
  env = vecenv_wrapper.RslRlVecEnvWrapper(
      env,
      clip_actions=agent_cfg.clip_actions,
      cfg=entry.runner_cfg.vecenv_wrapper_cfg,
  )

  runner = rls_rl_runners.OnPolicyRunner(
      env, dataclasses.asdict(agent_cfg), device=device
  )

  def load_checkpoint(path: str) -> None:
    runner.load(
        checkpoint.read_checkpoint(path),  # pyrefly: ignore[bad-argument-type]
        map_location=device,
    )

  reaf_env = env.unwrapped
  # Re-enable auto-reset, which RslRlVecEnvWrapper disables.
  reaf_env.auto_reset_mode = reaf_torch.AutoResetMode.SYNC

  policy = None
  load_checkpoint_fn = None
  if not _ZERO_ACTIONS.value:
    if _RESUME_PATH.value:
      load_checkpoint(_RESUME_PATH.value)
    policy = _RslRlPolicy(
        runner.get_inference_policy(device=device),
        reaf_env.action_spec(),
        _NWORLD.value,
    )
    load_checkpoint_fn = load_checkpoint

  viewer.launch(
      reaf_env,
      simulator=reaf_env.device_layer.sim,
      policy=policy,
      main_world_id=_ENV_ID.value,
      gfx=_GFX.value,
      http_port=_PORT.value,
      hot_reload=_HOT_RELOAD.value,
      plugins=[
          policy_plugin.PolicyPlugin(
              policy=policy,
              resume_path=_RESUME_PATH.value,
              load_checkpoint_fn=load_checkpoint_fn,
          )
      ],
  )

  env.close()


if __name__ == "__main__":
  app.run(main)
