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
"""Locomotion environments."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import jax
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.apollo import joystick as apollo_joystick
from mujoco_playground._src.locomotion.barkour import joystick as barkour_joystick
from mujoco_playground._src.locomotion.berkeley_humanoid import joystick as berkeley_humanoid_joystick
from mujoco_playground._src.locomotion.berkeley_humanoid import randomize as berkeley_humanoid_randomize
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize
from mujoco_playground._src.locomotion.go1 import getup as go1_getup
from mujoco_playground._src.locomotion.go1 import handstand as go1_handstand
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick
from mujoco_playground._src.locomotion.go1 import randomize as go1_randomize
from mujoco_playground._src.locomotion.h1 import inplace_gait_tracking as h1_inplace_gait_tracking
from mujoco_playground._src.locomotion.h1 import joystick_gait_tracking as h1_joystick_gait_tracking
from mujoco_playground._src.locomotion.op3 import joystick as op3_joystick
from mujoco_playground._src.locomotion.spot import getup as spot_getup
from mujoco_playground._src.locomotion.spot import joystick as spot_joystick
from mujoco_playground._src.locomotion.spot import joystick_gait_tracking as spot_joystick_gait_tracking
from mujoco_playground._src.locomotion.t1 import joystick as t1_joystick
from mujoco_playground._src.locomotion.t1 import randomize as t1_randomize


_envs = {
    "ApolloJoystickFlatTerrain": functools.partial(
        apollo_joystick.Joystick, task="flat_terrain"
    ),
    "BarkourJoystick": barkour_joystick.Joystick,
    "BerkeleyHumanoidJoystickFlatTerrain": functools.partial(
        berkeley_humanoid_joystick.Joystick, task="flat_terrain"
    ),
    "BerkeleyHumanoidJoystickRoughTerrain": functools.partial(
        berkeley_humanoid_joystick.Joystick, task="rough_terrain"
    ),
    "G1JoystickFlatTerrain": functools.partial(
        g1_joystick.Joystick, task="flat_terrain"
    ),
    "G1JoystickRoughTerrain": functools.partial(
        g1_joystick.Joystick, task="rough_terrain"
    ),
    "Go1JoystickFlatTerrain": functools.partial(
        go1_joystick.Joystick, task="flat_terrain"
    ),
    "Go1JoystickRoughTerrain": functools.partial(
        go1_joystick.Joystick, task="rough_terrain"
    ),
    "Go1Getup": go1_getup.Getup,
    "Go1Handstand": go1_handstand.Handstand,
    "Go1Footstand": go1_handstand.Footstand,
    "H1InplaceGaitTracking": h1_inplace_gait_tracking.InplaceGaitTracking,
    "H1JoystickGaitTracking": h1_joystick_gait_tracking.JoystickGaitTracking,
    "Op3Joystick": op3_joystick.Joystick,
    "SpotFlatTerrainJoystick": functools.partial(
        spot_joystick.Joystick, task="flat_terrain"
    ),
    "SpotGetup": spot_getup.Getup,
    "SpotJoystickGaitTracking": (
        spot_joystick_gait_tracking.JoystickGaitTracking
    ),
    "T1JoystickFlatTerrain": functools.partial(
        t1_joystick.Joystick, task="flat_terrain"
    ),
    "T1JoystickRoughTerrain": functools.partial(
        t1_joystick.Joystick, task="rough_terrain"
    ),
}

_cfgs = {
    "ApolloJoystickFlatTerrain": apollo_joystick.default_config,
    "BarkourJoystick": barkour_joystick.default_config,
    "BerkeleyHumanoidJoystickFlatTerrain": (
        berkeley_humanoid_joystick.default_config
    ),
    "BerkeleyHumanoidJoystickRoughTerrain": (
        berkeley_humanoid_joystick.default_config
    ),
    "G1JoystickFlatTerrain": g1_joystick.default_config,
    "G1JoystickRoughTerrain": g1_joystick.default_config,
    "Go1JoystickFlatTerrain": go1_joystick.default_config,
    "Go1JoystickRoughTerrain": go1_joystick.default_config,
    "Go1Getup": go1_getup.default_config,
    "Go1Handstand": go1_handstand.default_config,
    "Go1Footstand": go1_handstand.default_config,
    "H1InplaceGaitTracking": h1_inplace_gait_tracking.default_config,
    "H1JoystickGaitTracking": h1_joystick_gait_tracking.default_config,
    "Op3Joystick": op3_joystick.default_config,
    "SpotFlatTerrainJoystick": spot_joystick.default_config,
    "SpotGetup": spot_getup.default_config,
    "SpotJoystickGaitTracking": spot_joystick_gait_tracking.default_config,
    "T1JoystickFlatTerrain": t1_joystick.default_config,
    "T1JoystickRoughTerrain": t1_joystick.default_config,
}

_randomizer = {
    "BerkeleyHumanoidJoystickFlatTerrain": (
        berkeley_humanoid_randomize.domain_randomize
    ),
    "BerkeleyHumanoidJoystickRoughTerrain": (
        berkeley_humanoid_randomize.domain_randomize
    ),
    "G1JoystickFlatTerrain": g1_randomize.domain_randomize,
    "G1JoystickRoughTerrain": g1_randomize.domain_randomize,
    "Go1JoystickFlatTerrain": go1_randomize.domain_randomize,
    "Go1JoystickRoughTerrain": go1_randomize.domain_randomize,
    "Go1Getup": go1_randomize.domain_randomize,
    "Go1Handstand": go1_randomize.domain_randomize,
    "Go1Footstand": go1_randomize.domain_randomize,
    "T1JoystickFlatTerrain": t1_randomize.domain_randomize,
    "T1JoystickRoughTerrain": t1_randomize.domain_randomize,
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
        f"Env '{env_name}' does not have a domain randomizer in the locomotion"
        " registry."
    )
    return None
  return _randomizer[env_name]
