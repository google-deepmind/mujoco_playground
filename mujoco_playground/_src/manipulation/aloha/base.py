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
"""Base class for ALOHA."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import collision
from mujoco.mjx._src.support import contact_force
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aloha import aloha_constants as consts


def get_assets() -> Dict[str, bytes]:
  """Returns a dictionary of all assets used by the environment."""
  assets = {}
  path = mjx_env.MENAGERIE_PATH / "aloha"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  path = mjx_env.ROOT_PATH / "manipulation" / "aloha" / "xmls"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class AlohaEnv(mjx_env.MjxEnv):
  """Base class for ALOHA environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=get_assets()
    )
    self._mj_model.opt.timestep = self._config.sim_dt

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path

  def _post_init_aloha(self, keyframe: str = "home"):
    """Initializes helpful robot properties."""
    self._left_gripper_site = self._mj_model.site("left/gripper").id
    self._right_gripper_site = self._mj_model.site("right/gripper").id
    self._table_geom = self._mj_model.geom("table").id
    self._finger_geoms = [
        self._mj_model.geom(geom_id).id for geom_id in consts.FINGER_GEOMS
    ]
    self._init_q = jp.array(self._mj_model.keyframe(keyframe).qpos)
    self._init_ctrl = jp.array(self._mj_model.keyframe(keyframe).ctrl)
    self._lowers, self._uppers = self.mj_model.actuator_ctrlrange.T
    arm_joint_ids = [self._mj_model.joint(j).id for j in consts.ARM_JOINTS]
    self._arm_qadr = jp.array(
        [self._mj_model.jnt_qposadr[joint_id] for joint_id in arm_joint_ids]
    )
    self._finger_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in consts.FINGER_JOINTS
    ])

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  def hand_table_collision(self, data) -> jp.ndarray:
    # Check for collisions with the floor.
    hand_table_collisions = [
        collision.geoms_colliding(data, self._table_geom, g)
        for g in self._finger_geoms
    ]
    return (sum(hand_table_collisions) > 0).astype(float)

  def is_grasped(self, data, hand):
      # Grasped if both fingers have applied forces > 5.
      t_f = 2.5  # min force. Don't need to squeeze so hard!
      # 3D vec; top and bottom collision bodies
      f_lfing = self.get_finger_force(data, hand, "left")
      f_rfing = self.get_finger_force(data, hand, "right")
      d_lfing = self.get_finger_dir(data, hand, "left")
      d_rfing = -1 * d_lfing
      l_d_flag = self.check_dir(f_lfing, d_lfing)
      l_f_flag = (jp.linalg.norm(f_lfing) > t_f).astype(float)
      r_d_flag = self.check_dir(f_rfing, d_rfing)
      r_f_flag = (jp.linalg.norm(f_rfing) > t_f).astype(float)
      grasped = jp.all(jp.array([l_d_flag, l_f_flag, r_d_flag, r_f_flag])).astype(float)
      return grasped

  def get_finger_force(self, data, hand, finger):
      """
      Sum up the 3D force vectors across bottom and top collision primitives
      """
      ids = jp.array([self._mj_model.geom(f"{hand}/{finger}_finger_{pos}").id
             for pos in ["top", "bottom"]]) # 2
      contact_forces = [contact_force(self._mjx_model, data, i, True)[None, :3] # 1, 3
                        for i in np.arange(data.ncon)]
      contact_forces = jp.concat(contact_forces, axis=0) # ncon, 3
      matches = jp.isin(data.contact.geom, ids).any(axis=1) # ncon
      dist_mask = (data.contact.dist < 0) # ncon
      # Sum
      return jp.sum(contact_forces * (matches * dist_mask)[:, None], axis=0)

  def get_finger_dir(self, data, hand, finger):
      """
      A vector pointing from `finger` to the other finger.
      """
      other = "left" if finger == "right" else "right"
      site_fing = mujoco.mj_name2id(
          self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f'{hand}/{finger}_finger')
      site_ofing = mujoco.mj_name2id(
          self.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f'{hand}/{other}_finger')
      v = data.site_xpos[site_ofing] - data.site_xpos[site_fing]
      return v / (jp.linalg.norm(v) + 1e-7)

  def check_dir(self, v1, v2, t_align=jp.deg2rad(55)) -> float:
      m = jp.linalg.norm(v1) * jp.linalg.norm(v2)
      return (jp.arccos(jp.dot(v1, v2) / (m + 1e-7)) < t_align).astype(float)
