# Copyright 2024 DeepMind Technologies Limited
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

import mujoco
import numpy as np
import onnxruntime as rt
from etils import epath
from mujoco import viewer

from mujoco_playground._src.locomotion.apollo import constants as apollo_constants
from mujoco_playground._src.locomotion.apollo.base import get_assets
from mujoco_playground.experimental.sim2sim.gamepad_reader import Gamepad

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"


class OnnxController:
  """ONNX controller for the Booster Apollo humanoid."""

  def __init__(
    self,
    policy_path: str,
    default_angles: np.ndarray,
    ctrl_dt: float,
    n_substeps: int,
    action_scale: float = 0.5,
    vel_scale_x: float = 1.0,
    vel_scale_y: float = 1.0,
    vel_scale_rot: float = 1.0,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)

    self._counter = 0
    self._n_substeps = n_substeps
    self._ctrl_dt = ctrl_dt

    self._phase = np.array([0.0, np.pi])
    self._base_phase_dt = 2 * np.pi * ctrl_dt  # Store base phase_dt without frequency

    self._joystick = Gamepad(
      vel_scale_x=vel_scale_x,
      vel_scale_y=vel_scale_y,
      vel_scale_rot=vel_scale_rot,
      deadzone=0.03,
    )

  def get_obs(self, model, data) -> np.ndarray:
    linvel = data.sensor("local_linvel").data
    gyro = data.sensor("gyro").data
    imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    command = self._joystick.get_command()
    ph = self._phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
    phase = np.concatenate([np.cos(ph), np.sin(ph)])
    obs = np.hstack(
      [
        linvel,
        gyro,
        gravity,
        command,
        joint_angles,
        joint_velocities,
        self._last_action,
        phase,
      ]
    )
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      self._last_action = onnx_pred.copy()
      data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
      command = self._joystick.get_command()
      cmd_magnitude = np.linalg.norm(command)
      if cmd_magnitude < 0.01:
        gait_freq = 1.25
      else:
        gait_freq = 1.25 + 0.5 * min(cmd_magnitude, 1.5) / 1.5
      phase_dt = self._base_phase_dt * gait_freq
      phase_tp1 = self._phase + phase_dt
      self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


def load_callback(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
    apollo_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
    assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.02
  sim_dt = 0.005
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  policy = OnnxController(
    policy_path=(_ONNX_DIR / "apollo_policy.onnx").as_posix(),
    default_angles=np.array(model.keyframe("knees_bent").qpos[7:]),
    ctrl_dt=ctrl_dt,
    n_substeps=n_substeps,
    action_scale=0.5,
    vel_scale_x=1.5,
    vel_scale_y=0.8,
    vel_scale_rot=1.5,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
