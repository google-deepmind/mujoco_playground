# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrappers for MuJoCo Playground environments that interop with torch."""

from collections import deque
import functools
import os
from typing import Any

import jax
import numpy as np

try:
  from rsl_rl.env import VecEnv  # pytype: disable=import-error
except ImportError:
  VecEnv = object
try:
  import torch  # pytype: disable=import-error
except ImportError:
  torch = None

from mujoco_playground._src import wrapper
try:
  from tensordict import TensorDict  # pytype: disable=import-error
except ImportError:
  TensorDict = None


def _jax_to_torch(tensor):
  import torch.utils.dlpack as tpack  # pytype: disable=import-error # pylint: disable=import-outside-toplevel

  tensor = tpack.from_dlpack(tensor)
  return tensor


def _torch_to_jax(tensor):
  from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel

  tensor = from_dlpack(tensor)
  return tensor


def get_load_path(root, load_run=-1, checkpoint=-1):
  try:
    runs = os.listdir(root)
    # TODO sort by date to handle change of month
    runs.sort()
    if "exported" in runs:
      runs.remove("exported")
    last_run = os.path.join(root, runs[-1])
  except Exception as exc:
    raise ValueError("No runs in this directory: " + root) from exc
  if load_run == -1 or load_run == "-1":
    load_run = last_run
  else:
    load_run = os.path.join(root, load_run)

  if checkpoint == -1:
    models = [file for file in os.listdir(load_run) if "model" in file]
    models.sort(key=lambda m: m.zfill(15))
    model = models[-1]
  else:
    model = f"model_{checkpoint}.pt"

  load_path = os.path.join(load_run, model)
  return load_path


class RSLRLBraxWrapper(VecEnv):
  """Wrapper for Brax environments that interop with torch."""

  def __init__(
      self,
      env,
      num_actors,
      seed,
      episode_length,
      action_repeat,
      randomization_fn=None,
      render_callback=None,
      device_rank=None,
  ):
    import torch  # pytype: disable=import-error # pylint: disable=redefined-outer-name,unused-import,import-outside-toplevel

    self.seed = seed
    self.batch_size = num_actors
    self.num_envs = num_actors

    self.key = jax.random.PRNGKey(self.seed)

    if device_rank is not None:
      gpu_devices = jax.devices("gpu")
      self.key = jax.device_put(self.key, gpu_devices[device_rank])
      self.device = f"cuda:{device_rank}"
      print(f"Device -- {gpu_devices[device_rank]}")
      print(f"Key device -- {self.key.devices()}")

    # split key into two for reset and randomization
    key_reset, key_randomization = jax.random.split(self.key)

    self.key_reset = jax.random.split(key_reset, self.batch_size)

    if randomization_fn is not None:
      randomization_rng = jax.random.split(key_randomization, self.batch_size)
      v_randomization_fn = functools.partial(
          randomization_fn, rng=randomization_rng
      )
    else:
      v_randomization_fn = None

    self.env = wrapper.wrap_for_brax_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    self.render_callback = render_callback

    self.asymmetric_obs = False
    obs_shape = self.env.env.unwrapped.observation_size
    print(f"obs_shape: {obs_shape}")

    if isinstance(obs_shape, dict):
      print("Asymmetric observation space")
      self.asymmetric_obs = True
      self.num_obs = obs_shape["state"]
      self.num_privileged_obs = obs_shape["privileged_state"]
    else:
      self.num_obs = obs_shape
      self.num_privileged_obs = None

    self.num_actions = self.env.env.unwrapped.action_size

    self.max_episode_length = episode_length

    # todo -- specific to leap environment
    self.success_queue = deque(maxlen=100)

    print("JITing reset and step")
    self.reset_fn = jax.jit(self.env.reset)
    self.step_fn = jax.jit(self.env.step)
    print("Done JITing reset and step")
    self.env_state = None

  def step(self, action):
    action = torch.clip(action, -1.0, 1.0)  # pytype: disable=attribute-error
    action = _torch_to_jax(action)
    self.env_state = self.step_fn(self.env_state, action)
    critic_obs = None
    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
      obs = {"state": obs, "privileged_state": critic_obs}
    else:
      obs = _jax_to_torch(self.env_state.obs)
      obs = {"state": obs}
    reward = _jax_to_torch(self.env_state.reward)
    done = _jax_to_torch(self.env_state.done)
    info = self.env_state.info
    truncation = _jax_to_torch(info["truncation"])

    info_ret = {
        "time_outs": truncation,
        "observations": {"critic": critic_obs},
        "log": {},
    }

    if "last_episode_success_count" in info:
      last_episode_success_count = (
          _jax_to_torch(info["last_episode_success_count"])[done > 0]  # pylint: disable=unsubscriptable-object
          .float()
          .tolist()
      )
      if len(last_episode_success_count) > 0:
        self.success_queue.extend(last_episode_success_count)
      info_ret["log"]["last_episode_success_count"] = np.mean(
          self.success_queue
      )

    for k, v in self.env_state.metrics.items():
      if k not in info_ret["log"]:
        info_ret["log"][k] = _jax_to_torch(v).float().mean().item()

    obs = TensorDict(obs, batch_size=[self.num_envs])
    return obs, reward, done, info_ret

  def reset(self):
    # todo add random init like in collab examples?
    self.env_state = self.reset_fn(self.key_reset)

    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
      obs = {"state": obs, "privileged_state": critic_obs}
    else:
      obs = _jax_to_torch(self.env_state.obs)
      obs = {"state": obs}
    return TensorDict(obs, batch_size=[self.num_envs])

  def get_observations(self):
   return self.reset()

  def render(self, mode="human"):  # pylint: disable=unused-argument
    if self.render_callback is not None:
      self.render_callback(self.env.env.env, self.env_state)
    else:
      raise ValueError("No render callback specified")

  def get_number_of_agents(self):
    return 1

  def get_env_info(self):
    info = {}
    info["action_space"] = self.action_space  # pytype: disable=attribute-error
    info["observation_space"] = (
        self.observation_space  # pytype: disable=attribute-error
    )
    return info
