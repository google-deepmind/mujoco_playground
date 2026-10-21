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

"""Environment and config registries for RL."""

import dataclasses
from typing import Any
from typing import Callable
from typing import Dict

from mujoco_playground._src.rl.rsl_rl import env_rl_config
from mujoco_playground.tasks.dm_control_suite import cartpole
from mujoco_playground.tasks.dm_control_suite import cartpole_vision
from mujoco_playground.tasks.manipulation.pick_cube.so101 import task as so101_pick_cube
from mujoco_playground.tasks.manipulation.pick_cube.trossen_wxai import task as trossen_pick_cube
from mujoco_playground.tasks.whole_body.tracking import task as whole_body_tracking
from reaf.simulation.torch import environment as reaf_env

EnvBuilder = Callable[..., reaf_env.Environment]


@dataclasses.dataclass
class RegistryEntry:
  env_builder: EnvBuilder
  runner_cfg: Any
  env_cfg_fn: Callable


_REGISTRY: Dict[str, RegistryEntry] = {}


def register(
    env_id: str,
    runner_cfg: Any,
    env_cfg_fn: Callable,
) -> None:
  """Registers an env builder under a single env_id.

  Args:
    env_id: Unique identifier for the environment.
    runner_cfg: Runner configuration dataclass.
    env_cfg_fn: A callable returning the env config. Called lazily to avoid file
      access at import time.
  """
  if env_id in _REGISTRY:
    raise ValueError(f"Environment '{env_id}' already registered.")

  def build(
      nworld: int = 1,
      device: str = "cuda:0",
      **kwargs,
  ) -> reaf_env.Environment:
    """Builds an environment from a config object."""
    from reaf.simulation.torch import flat_scene as build_env

    impl = kwargs.pop("impl", None)
    env_cfg = env_cfg_fn(**kwargs)
    env_cfg.sim.nworld = nworld
    env_cfg.sim.xpu_device = device
    env_cfg.sim.impl = impl or ("warp" if device.startswith("cuda") else "cpp")
    return build_env.build_env(env_cfg)

  _REGISTRY[env_id] = RegistryEntry(
      env_builder=build,
      runner_cfg=runner_cfg,
      env_cfg_fn=env_cfg_fn,
  )


def get(env_id: str) -> RegistryEntry:
  """Retrieves an entry from the registry."""
  if env_id not in _REGISTRY:
    raise ValueError(
        f"Environment '{env_id}' not found in registry."
        f" Available: {list(_REGISTRY.keys())}"
    )
  return _REGISTRY[env_id]


# TODO(btaba): mark these environments specific to the RL package.
register(
    "SO101PickCube",
    env_rl_config.PickCubeRslRlOnPolicyRunnerCfg(),
    env_cfg_fn=so101_pick_cube.so101_pick_cube_env,
)
register(
    "SO101PickCubeVision",
    env_rl_config.PickCubeVisionRslRlOnPolicyRunnerCfg(),
    env_cfg_fn=so101_pick_cube.so101_pick_cube_pixels_env,
)
register(
    "TrossenPickCube",
    env_rl_config.PickCubeRslRlOnPolicyRunnerCfg(),
    env_cfg_fn=trossen_pick_cube.right_arm_pick_cube_env,
)
register(
    "TrossenPickCubeVision",
    env_rl_config.PickCubeVisionRslRlOnPolicyRunnerCfg(),
    env_cfg_fn=trossen_pick_cube.right_arm_pick_cube_pixels_env,
)
register(
    "CartpoleBalance",
    env_rl_config.CartpoleRslRlOnPolicyRunnerCfg(),
    env_cfg_fn=cartpole.cartpole_balance_env,
)
register(
    "CartpoleSwingUp",
    env_rl_config.CartpoleRslRlOnPolicyRunnerCfg(),
    env_cfg_fn=cartpole.cartpole_swingup_env,
)
register(
    "CartpoleBalanceVision",
    env_rl_config.CartpoleVisionRslRlOnPolicyRunnerCfg(),
    env_cfg_fn=cartpole_vision.cartpole_balance_vision_env,
)
register(
    "G1WholeBodyTracking",
    env_rl_config.G1WholeBodyTrackingRslRlOnPolicyRunnerCfg(),
    env_cfg_fn=whole_body_tracking.g1_env,
)
