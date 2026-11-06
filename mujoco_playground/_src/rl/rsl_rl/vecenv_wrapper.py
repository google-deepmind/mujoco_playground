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

"""RSL-RL VecEnv wrapper for reaf-sim environments."""

from collections.abc import Mapping
from typing import Any

import dm_env
from mujoco_playground._src.rl import rng
from mujoco_playground._src.rl.rsl_rl import rsl_config as rsl_config
from reaf.simulation.torch import environment as reaf_env
from reaf.simulation.torch import profile
from reaf.simulation.torch import timeout_termination_checker
from rsl_rl import env as rsl_rl_env
import tensordict
import torch

# TODO: fix the blanket attribute-error disable.


def _dict_reward_from_timestep(timestep: Any) -> dict[str, Any]:
  reward_dict = timestep.reward
  if not isinstance(reward_dict, dict):
    reward_dict = {"reward": reward_dict, "terms": {}}
  return reward_dict


def _derive_max_episode_length(env: reaf_env.Environment) -> int:
  """Derives max_episode_length from TimeoutTerminationChecker."""
  task_layer = env.task_layer
  for checker in task_layer._termination_checkers:  # pylint: disable=protected-access
    if isinstance(
        checker, timeout_termination_checker.TimeoutTerminationChecker
    ):
      control_dt = env.device_layer.control_timestep
      return int(checker.timeout / control_dt)
  raise ValueError("No TimeoutTerminationChecker found in the environment.")


class RslRlVecEnvWrapper(rsl_rl_env.VecEnv):
  """Wraps reaf-sim environment for the RSL-RL library."""

  def __init__(
      self,
      env: reaf_env.Environment,
      clip_actions: float | None = None,
      cfg: rsl_config.RslRlVecEnvWrappercfg | None = None,
      max_episode_length: int | None = None,
  ):
    if clip_actions:
      raise ValueError("clip_actions is not supported.")

    self.env = env
    # RslRlVecEnvWrapper does gym-style resets, so we disable
    # the auto-reset behavior from REAF `Environment.step()`.
    self.env.auto_reset_mode = reaf_env.AutoResetMode.DISABLED
    self.clip_actions = clip_actions
    self.num_envs = self.env.nworld
    self.device = torch.device(self.env.xpu_device)
    self._cfg = cfg

    # max_episode_length is required by RSL-RL.
    # Derive max_episode_length from TimeoutTerminationChecker if present.
    self._max_episode_length = max_episode_length or _derive_max_episode_length(
        self.env
    )
    self._control_timestep = self.env.device_layer.control_timestep
    # Track episode length per env for RSL-RL's init_at_random_ep_len
    self._episode_length_buf = torch.zeros(
        self.num_envs, device=self.device, dtype=torch.long
    )

    # TODO: how do we make action_spec NOT return a dict?
    # RSL-RL does not want a dict for the action...
    self.num_actions = sum(
        a.shape[-1] for a in self.unwrapped.action_spec().values()  # pyrefly: ignore[missing-attribute]
    )
    self._modify_action_space()

    # Reset at the start since rsl_rl does not call reset.
    self._timestep = self.env.reset()
    self._episode_sums = {}

  @property
  def cfg(self):  # pyrefly: ignore[bad-override]
    """Returns the VecEnv wrapper config."""
    return self._cfg

  def render_mode(self) -> str | None:
    # return self.env.render_mode
    return None

  def observation_spec(self):
    return self.env.observation_spec()

  def action_spec(self):
    return self.env.action_spec()

  def reward_spec(self):
    return self.env.reward_spec()

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> reaf_env.Environment:
    return self.env

  # Properties.

  @property
  def max_episode_length(self) -> int:  # pyrefly: ignore[bad-override]
    """Maximum episode length in steps, required by RSL-RL."""
    return self._max_episode_length

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self._episode_length_buf

  @episode_length_buf.setter
  @torch.inference_mode()
  def episode_length_buf(self, value: torch.Tensor) -> None:  # type: ignore
    self._episode_length_buf = value
    # Sync data.time with the new episode lengths.
    # This converts step count to time and writes to data.time directly.
    episode_time = value.float() * self._control_timestep
    self.unwrapped.device_layer.sim.data.time[:] = episode_time

  def seed(self, seed: int = -1) -> int:
    """Sets the global random seed for reproducibility."""
    if seed >= 0:
      rng.seed(seed)
    return seed

  def _get_flat_obs(self, timestep: Any) -> dict[str, Any]:
    obs_dict = timestep.observation
    if not isinstance(obs_dict, dict):
      obs_dict = {"policy": obs_dict}
    return obs_dict

  @profile.trace("VecEnv.get_observations", color=profile.Color.DEEP_SKY_BLUE)
  def get_observations(self) -> tensordict.TensorDict:
    obs_dict = self._get_flat_obs(self._timestep)
    return tensordict.TensorDict(
        obs_dict, batch_size=[self.unwrapped.nworld]  # pyrefly: ignore[bad-argument-type]
    )

  @profile.trace("VecEnv.reset", color=profile.Color.MEDIUM_PURPLE)
  def reset(self) -> tuple[tensordict.TensorDict, dict]:
    self._timestep = self.env.reset()
    # self._reset_episode_stats(torch.arange(self.num_envs, device=self.device))
    obs_dict = self._get_flat_obs(self._timestep)
    extras = {
        "step_type": self._timestep.step_type,
    }
    return (
        tensordict.TensorDict(
            obs_dict, batch_size=[self.unwrapped.nworld]  # pyrefly: ignore[bad-argument-type]
        ),
        extras,
    )

  @profile.trace("VecEnv.step", color=profile.Color.DODGER_BLUE)
  def step(
      self, actions: torch.Tensor
  ) -> tuple[tensordict.TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      raise ValueError("clip_actions is not supported.")
      # actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

    # TODO: we need to make sure actions are not dictionaries,
    # or we need a generic way to convert back to dicts for REAF, so that
    # the action_spec is compatible with RSL-RL.
    action_spec = self.action_spec()
    assert isinstance(action_spec, Mapping)
    action_key = list(action_spec.keys())[0]
    action_dict = {action_key: actions}

    self._timestep = self.env.step(action_dict)
    rew_dict = _dict_reward_from_timestep(self._timestep)
    rew = rew_dict["reward"]

    # Increment episode length buffer for this step.
    self._episode_length_buf += 1

    is_last = self._timestep.step_type == dm_env.StepType.LAST
    terminated = is_last & (self._timestep.discount == 0)
    truncated = is_last & (self._timestep.discount > 0)
    term_or_trunc = terminated | truncated
    dones = term_or_trunc.to(dtype=torch.long)
    done_mask = term_or_trunc.bool()

    # Log rewards for RSL-RL.
    reward_terms = rew_dict["terms"]
    extras = {
        "step_type": self._timestep.step_type,
        "log": {},
        "reward_terms_per_world": reward_terms,
    }
    for k, v in reward_terms.items():
      extras["log"]["reward_step/" + k] = v.mean()
      if k not in self._episode_sums:
        self._episode_sums[k] = torch.zeros_like(v)
      self._episode_sums[k] += v

    # For terminated envs, get S_0 observations. This is the auto-reset behavior
    # that RSL-RL expects.
    if done_mask.any():
      # Snapshot episode metrics before the auto-reset overwrites them.
      for k, val in self.env.task_layer.get_episode_metrics(done_mask).items():
        extras["log"][f"Metrics/{k}"] = val
      for k in self._episode_sums:
        extras["log"]["reward_episode/" + k] = self._episode_sums[k][
            done_mask
        ].mean()
        self._episode_sums[k][done_mask] = 0.0
      # Reset episode length for done envs.
      self._episode_length_buf[done_mask] = 0
      self._timestep = self.env.reset_with_options(
          options=self.env.default_reset_options(),
          env_mask=done_mask,
      )
      extras["step_type"] = self._timestep.step_type

    # get the observation dict after the conditional auto-reset
    obs_dict = self._get_flat_obs(self._timestep)

    if self.cfg is not None and not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated

    return (
        tensordict.TensorDict(
            obs_dict, batch_size=[self.num_envs]  # pyrefly: ignore[bad-argument-type]
        ),
        rew,
        dones,
        extras,
    )

  def close(self) -> None:
    self.unwrapped.close()

  def _modify_action_space(self) -> None:
    if self.clip_actions is None:
      return
    raise ValueError("clip_actions is not supported.")
    # TODO: support clipped actions? Or should we just let REAF handle
    # this? This is coming from the Isaac code.
