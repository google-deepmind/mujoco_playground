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
"""Module for manipulation environments."""
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import jax
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aloha import handover as aloha_handover
from mujoco_playground._src.manipulation.aloha import single_peg_insertion as aloha_peg
from mujoco_playground._src.manipulation.franka_emika_panda import open_cabinet as panda_open_cabinet
from mujoco_playground._src.manipulation.franka_emika_panda import pick as panda_pick
from mujoco_playground._src.manipulation.franka_emika_panda import pick_cartesian as panda_pick_cartesian
from mujoco_playground._src.manipulation.franka_emika_panda_robotiq import push_cube as robotiq_push_cube
from mujoco_playground._src.manipulation.leap_hand import reorient as leap_cube_reorient
from mujoco_playground._src.manipulation.leap_hand import rotate_z as leap_rotate_z


_envs = {
    "AlohaHandOver": aloha_handover.HandOver,
    "AlohaSinglePegInsertion": aloha_peg.SinglePegInsertion,
    "PandaPickCube": panda_pick.PandaPickCube,
    "PandaPickCubeOrientation": panda_pick.PandaPickCubeOrientation,
    "PandaPickCubeCartesian": panda_pick_cartesian.PandaPickCubeCartesian,
    "PandaOpenCabinet": panda_open_cabinet.PandaOpenCabinet,
    "PandaRobotiqPushCube": robotiq_push_cube.PandaRobotiqPushCube,
    "LeapCubeReorient": leap_cube_reorient.CubeReorient,
    "LeapCubeRotateZAxis": leap_rotate_z.CubeRotateZAxis,
}

_cfgs = {
    "AlohaHandOver": aloha_handover.default_config,
    "AlohaSinglePegInsertion": aloha_peg.default_config,
    "PandaPickCube": panda_pick.default_config,
    "PandaPickCubeOrientation": panda_pick.default_config,
    "PandaPickCubeCartesian": panda_pick_cartesian.default_config,
    "PandaOpenCabinet": panda_open_cabinet.default_config,
    "PandaRobotiqPushCube": robotiq_push_cube.default_config,
    "LeapCubeReorient": leap_cube_reorient.default_config,
    "LeapCubeRotateZAxis": leap_rotate_z.default_config,
}

_randomizer = {
    "LeapCubeRotateZAxis": leap_rotate_z.domain_randomize,
    "LeapCubeReorient": leap_cube_reorient.domain_randomize,
}


def __getattr__(name):
  if name == "ALL_ENVS":
    return tuple(_envs.keys())
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def register_environment(
    env_name: str,
    env_class: Type[mjx_env.MjxEnv],
    cfg_class: Callable[[], config_dict.ConfigDict],
) -> None:
  """Register a new environment.

  Args:
      env_name: The name of the environment.
      env_class: The environment class.
      cfg_class: The default configuration.
  """
  _envs[env_name] = env_class
  _cfgs[env_name] = cfg_class


def get_default_config(env_name: str) -> config_dict.ConfigDict:
  """Get the default configuration for an environment."""
  if env_name not in _cfgs:
    raise ValueError(
        f"Env '{env_name}' not found in default configs. Available configs:"
        f" {list(_cfgs.keys())}"
    )
  return _cfgs[env_name]()


def load(
    env_name: str,
    config: Optional[config_dict.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
) -> mjx_env.MjxEnv:
  """Get an environment instance with the given configuration.

  Args:
      env_name: The name of the environment.
      config: The configuration to use. If not provided, the default
        configuration is used.
      config_overrides: A dictionary of overrides for the configuration.

  Returns:
      An instance of the environment.
  """
  mjx_env.ensure_menagerie_exists()  # Ensure menagerie exists when environment is loaded.
  if env_name not in _envs:
    raise ValueError(
        f"Env '{env_name}' not found. Available envs: {_cfgs.keys()}"
    )
  config = config or get_default_config(env_name)
  return _envs[env_name](config=config, config_overrides=config_overrides)


def get_domain_randomizer(
    env_name: str,
) -> Optional[Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]]:
  """Get the default domain randomizer for an environment."""
  if env_name not in _randomizer:
    print(
        f"Env '{env_name}' does not have a domain randomizer in the"
        " manipulation registry."
    )
    return None
  return _randomizer[env_name]
