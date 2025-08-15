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
"""Franka Emika Panda base class."""

from typing import Any, Dict, Optional, Union

from etils import epath
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
GRIPPER_GEOMS = [
    "left_coupler_col_1",
    "left_coupler_col_2",
    "left_follower_pad2",
    "right_coupler_col_1",
    "right_coupler_col_2",
    "right_follower_pad2",
]
FINGER_JOINTS = ["right_driver_joint", "left_driver_joint"]
GEAR = np.array([150.0, 150.0, 150.0, 150.0, 20.0, 20.0, 20.0])
_MENAGERIE_FRANKA_DIR = "franka_emika_panda"
_MENAGERIE_GRIPPER_DIR = "robotiq_2f85_v4"
_ENV_DIR = mjx_env.ROOT_PATH / "manipulation/franka_emika_panda_robotiq"


def get_assets() -> Dict[str, bytes]:
  assets = {}
  path = mjx_env.MENAGERIE_PATH / _MENAGERIE_FRANKA_DIR
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  path = mjx_env.MENAGERIE_PATH / _MENAGERIE_GRIPPER_DIR
  mjx_env.update_assets(assets, path / "assets")
  mjx_env.update_assets(assets, _ENV_DIR / "xmls", "*.xml")
  mjx_env.update_assets(assets, _ENV_DIR / "assets", "*")
  return assets


class PandaRobotiqBase(mjx_env.MjxEnv):
  """Base environment for Franka Emika Panda and Robotiq gripper."""

  def __init__(
      self,
      config: config_dict.ConfigDict,
      xml_path: epath.Path,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    self._xml_path = xml_path.as_posix()
    xml = xml_path.read_text()
    self._model_assets = get_assets()
    mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
    mj_model.opt.timestep = self.sim_dt

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)

  def _post_init(self, obj_name: str, keyframe: str):
    all_joints = ARM_JOINTS + FINGER_JOINTS
    self._robot_arm_qposadr = np.array([
        self.mj_model.jnt_qposadr[self.mj_model.joint(j).id] for j in ARM_JOINTS
    ])
    self._robot_qposadr = np.array([
        self.mj_model.jnt_qposadr[self.mj_model.joint(j).id] for j in all_joints
    ])
    self._gripper_site = self.mj_model.site("gripper").id
    self._left_finger_geom = self.mj_model.geom("left_finger_pad").id
    self._right_finger_geom = self.mj_model.geom("right_finger_pad").id
    self._hand_geom = self.mj_model.geom("hand_capsule").id
    self._gripper_geoms = [self.mj_model.geom(n).id for n in GRIPPER_GEOMS]
    self._obj_body = self.mj_model.body(obj_name).id
    self._obj_geom = self.mj_model.geom(obj_name).id
    self._obj_qposadr = self.mj_model.jnt_qposadr[
        self.mj_model.body(obj_name).jntadr[0]
    ]
    self._mocap_target = self.mj_model.body("mocap_target").mocapid
    self._floor_geom = self.mj_model.geom("floor").id
    self._wall_geom = self.mj_model.geom("wall").id
    self._init_q = self.mj_model.keyframe(keyframe).qpos.copy()
    self._init_obj_pos = np.array(
        self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
        dtype=np.float32,
    )
    self._init_obj_quat = np.array(
        self._init_q[self._obj_qposadr + 3 : self._obj_qposadr + 7],
        dtype=np.float32,
    )
    self._init_ctrl = self.mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T
    self._q_low_joint_pos_index = 0
    self._q_upper_joint_pos_index = 7
    self._qd_low_joint_pos_index = 0
    self._qd_upper_joint_pos_index = 7
    self._gear = GEAR
    self._joint_limit_percentage = 0.9
    self._joint_vel_limit_percentage = 0.9
    self._jnt_range = np.array(self.jnt_range())
    self._jnt_vel_range = np.array(self.jnt_vel_range())
    self._joint_range_init_percent_limit = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
    )
    self._max_torque = 8.0
    self._gripper_obj_normal_sensor = [
        self.mj_model.sensor(geom + "_" + obj_name + "_normal").id
        for geom in GRIPPER_GEOMS
    ]
    hand_geoms = [
        "left_finger_pad",
        "right_finger_pad",
        "hand_capsule",
    ]
    self._hand_wall_found_sensor = [
        self.mj_model.sensor("wall_" + hand_geom + "_found").id
        for hand_geom in hand_geoms
    ]
    self._hand_floor_found_sensor = [
        self.mj_model.sensor("floor_" + hand_geom + "_found").id
        for hand_geom in hand_geoms
    ]

  def jnt_range(self):
    # TODO(siholt): Use joint limits from XML.
    return [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ]

  def jnt_vel_range(self):
    return [
        [-2.1750, 2.1750],
        [-2.1750, 2.1750],
        [-2.1750, 2.1750],
        [-2.1750, 2.1750],
        [-2.6100, 2.6100],
        [-2.6100, 2.6100],
        [-2.6100, 2.6100],
    ]

  def ctrl_range(self):
    return [
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
    ]

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
