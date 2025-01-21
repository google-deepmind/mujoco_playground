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
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.manipulation.leap_hand import leap_hand_constants
from mujoco_playground._src.manipulation.leap_hand.base import get_assets
from mujoco_playground._src.mjx_env import get_qpos_ids
from mujoco_playground._src.mjx_env import get_qvel_ids

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"


class OnnxController:
  """ONNX controller for the Leap hand."""

  def __init__(
      self,
      policy_path: str,
      hand_qids: np.ndarray,
      hand_dqids: np.ndarray,
      ctrl_init: np.ndarray,
      lowers: np.ndarray,
      uppers: np.ndarray,
      n_substeps: int,
      action_scale: float = 0.5,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._hand_qids = hand_qids
    self._hand_dqids = hand_dqids
    self._action_scale = action_scale
    self._last_action = np.zeros_like(hand_qids, dtype=np.float32)
    self._motor_targets = ctrl_init.copy()
    self._lowers = lowers
    self._uppers = uppers

    self._counter = 0
    self._n_substeps = n_substeps

  def get_obs(self, model, data) -> np.ndarray:  # pylint: disable=unused-argument
    joint_angles = data.qpos[self._hand_qids]
    qpos_error = joint_angles - self._motor_targets
    cube_pos_error = (
        data.sensor("palm_position").data - data.sensor("cube_position").data
    )

    cube_quat = data.sensor("cube_orientation").data
    goal_quat = data.sensor("cube_goal_orientation").data
    goal_quat_inv = np.zeros(4)
    mujoco.mju_negQuat(goal_quat_inv, goal_quat)
    quat_diff = np.zeros(4)
    mujoco.mju_mulQuat(quat_diff, cube_quat, goal_quat_inv)
    mat_diff = np.zeros(9)
    mujoco.mju_quat2Mat(mat_diff, quat_diff)
    cube_ori_error = mat_diff[3:]
    obs = np.hstack([
        joint_angles,
        qpos_error,
        cube_pos_error,
        cube_ori_error,
        self._last_action,
    ])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      delta = onnx_pred * self._action_scale
      data.ctrl[:] += +delta
      np.clip(data.ctrl, self._lowers, self._uppers, out=data.ctrl)
      self._motor_targets = data.ctrl.copy()
      self._last_action = onnx_pred.copy()


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      leap_hand_constants.CUBE_XML.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.05
  sim_dt = 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  hand_qids = get_qpos_ids(model, leap_hand_constants.JOINT_NAMES)
  hand_dqids = get_qvel_ids(model, leap_hand_constants.JOINT_NAMES)

  policy = OnnxController(
      policy_path=(_ONNX_DIR / "leap_reorient_policy.onnx").as_posix(),
      hand_qids=hand_qids,
      hand_dqids=hand_dqids,
      n_substeps=n_substeps,
      action_scale=0.5,
      ctrl_init=data.ctrl,
      lowers=model.actuator_ctrlrange[:, 0],
      uppers=model.actuator_ctrlrange[:, 1],
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
