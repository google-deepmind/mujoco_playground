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
"""DeepMind Control Suite Environments."""

from functools import partial  # pylint: disable=g-importing-member
from typing import Any, Callable, Dict, Optional, Type, Union

from ml_collections import config_dict

from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import acrobot
from mujoco_playground._src.dm_control_suite import ball_in_cup
from mujoco_playground._src.dm_control_suite import cartpole
from mujoco_playground._src.dm_control_suite import cheetah
from mujoco_playground._src.dm_control_suite import finger
from mujoco_playground._src.dm_control_suite import fish
from mujoco_playground._src.dm_control_suite import hopper
from mujoco_playground._src.dm_control_suite import humanoid
from mujoco_playground._src.dm_control_suite import pendulum
from mujoco_playground._src.dm_control_suite import point_mass
from mujoco_playground._src.dm_control_suite import reacher
from mujoco_playground._src.dm_control_suite import swimmer
from mujoco_playground._src.dm_control_suite import walker

_envs = {
    "AcrobotSwingup": partial(acrobot.Balance, sparse=False),
    "AcrobotSwingupSparse": partial(acrobot.Balance, sparse=True),
    "BallInCup": ball_in_cup.BallInCup,
    "CartpoleBalance": partial(cartpole.Balance, swing_up=False, sparse=False),
    "CartpoleBalanceSparse": partial(
        cartpole.Balance, swing_up=False, sparse=True
    ),
    "CartpoleSwingup": partial(cartpole.Balance, swing_up=True, sparse=False),
    "CartpoleSwingupSparse": partial(
        cartpole.Balance, swing_up=True, sparse=True
    ),
    "CheetahRun": cheetah.Run,
    "FingerSpin": finger.Spin,
    "FingerTurnEasy": partial(
        finger.Turn, target_radius=finger.EASY_TARGET_SIZE
    ),
    "FingerTurnHard": partial(
        finger.Turn, target_radius=finger.HARD_TARGET_SIZE
    ),
    "FishSwim": fish.Swim,
    "HopperHop": partial(hopper.Hopper, hopping=True),
    "HopperStand": partial(hopper.Hopper, hopping=False),
    "HumanoidStand": partial(humanoid.Humanoid, move_speed=0.0),
    "HumanoidWalk": partial(humanoid.Humanoid, move_speed=humanoid.WALK_SPEED),
    "HumanoidRun": partial(humanoid.Humanoid, move_speed=humanoid.RUN_SPEED),
    "PendulumSwingup": pendulum.SwingUp,
    "PointMass": point_mass.PointMass,
    "ReacherEasy": partial(reacher.Reacher, target_size=reacher.BIG_TARGET),
    "ReacherHard": partial(reacher.Reacher, target_size=reacher.SMALL_TARGET),
    "SwimmerSwimmer6": partial(swimmer.Swim, n_links=6),
    "WalkerRun": partial(walker.PlanarWalker, move_speed=walker.RUN_SPEED),
    "WalkerStand": partial(walker.PlanarWalker, move_speed=0.0),
    "WalkerWalk": partial(walker.PlanarWalker, move_speed=walker.WALK_SPEED),
}

_cfgs = {
    # go/keep-sorted start
    "AcrobotSwingup": acrobot.default_config,
    "AcrobotSwingupSparse": acrobot.default_config,
    "BallInCup": ball_in_cup.default_config,
    "CartpoleBalance": cartpole.default_config,
    "CartpoleBalanceSparse": cartpole.default_config,
    "CartpoleSwingup": cartpole.default_config,
    "CartpoleSwingupSparse": cartpole.default_config,
    "CheetahRun": cheetah.default_config,
    "FingerSpin": finger.default_config,
    "FingerTurnEasy": finger.default_config,
    "FingerTurnHard": finger.default_config,
    "FishSwim": fish.default_config,
    "HopperHop": hopper.default_config,
    "HopperStand": hopper.default_config,
    "HumanoidRun": humanoid.default_config,
    "HumanoidStand": humanoid.default_config,
    "HumanoidWalk": humanoid.default_config,
    "PendulumSwingup": pendulum.default_config,
    "PointMass": point_mass.default_config,
    "ReacherEasy": reacher.default_config,
    "ReacherHard": reacher.default_config,
    "SwimmerSwimmer6": swimmer.default_config,
    "WalkerRun": walker.default_config,
    "WalkerStand": walker.default_config,
    "WalkerWalk": walker.default_config,
    # go/keep-sorted end
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
      cfg_class: The default configuration
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
  if env_name not in _envs:
    raise ValueError(f"Env '{env_name}' not found. Available envs: {_cfgs.keys()}")
  config = config or get_default_config(env_name)
  return _envs[env_name](config=config, config_overrides=config_overrides)
