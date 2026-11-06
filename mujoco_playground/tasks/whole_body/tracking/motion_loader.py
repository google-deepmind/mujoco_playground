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

"""Reference motion trajectory loader for whole-body tracking.

Expected NPZ keys and shapes:
  joint_pos:      (T, num_joints)   actuated joint positions, model order
  joint_vel:      (T, num_joints)   actuated joint velocities
  body_pos_w:     (T, num_bodies, 3)  world positions of all bodies [1:]
  body_quat_w:    (T, num_bodies, 4)  world quaternions (w, x, y, z)
  body_lin_vel_w: (T, num_bodies, 3)  world linear velocities at body origin
  body_ang_vel_w: (T, num_bodies, 3)  world angular velocities

Body arrays cover every robot body excluding the worldbody, in model body
order (index 0 = pelvis); `body_indexes` selects the tracked subset.
Generate files with csv_to_npz.py using the same robot model.
"""

import io

from etils import epath
import numpy as np
import torch


class MotionLoader:
  """Loads a reference motion npz and selects tracked bodies."""

  def __init__(
      self,
      motion_file: str,
      body_indexes: torch.Tensor,
      device: str = "cpu",
  ) -> None:
    with epath.Path(motion_file).open("rb") as f:
      file_bytes = f.read()
    data = np.load(io.BytesIO(file_bytes))
    self.joint_pos = torch.tensor(
        data["joint_pos"], dtype=torch.float32, device=device
    )
    self.joint_vel = torch.tensor(
        data["joint_vel"], dtype=torch.float32, device=device
    )
    self._body_pos_w = torch.tensor(
        data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
        data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
        data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
        data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]

    if torch.isnan(self.joint_pos).any():
      raise ValueError(f"Motion file {motion_file} contains NaN values.")

  def to(self, device: str | torch.device) -> "MotionLoader":
    """Moves all tensors to the given device in place."""
    for name in (
        "joint_pos",
        "joint_vel",
        "_body_pos_w",
        "_body_quat_w",
        "_body_lin_vel_w",
        "_body_ang_vel_w",
        "_body_indexes",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
      setattr(self, name, getattr(self, name).to(device))
    return self


def make_dummy_motion(
    num_tracked_bodies: int,
    num_joints: int,
    num_steps: int = 100,
    root_height: float = 0.76,
    device: str = "cpu",
) -> MotionLoader:
  """Creates a standing-still dummy motion (for tests / smoke runs)."""
  loader = MotionLoader.__new__(MotionLoader)
  loader.joint_pos = torch.zeros(num_steps, num_joints, device=device)
  loader.joint_vel = torch.zeros(num_steps, num_joints, device=device)
  body_pos = torch.zeros(num_steps, num_tracked_bodies, 3, device=device)
  body_pos[:, :, 2] = root_height
  body_quat = torch.zeros(num_steps, num_tracked_bodies, 4, device=device)
  body_quat[:, :, 0] = 1.0
  loader._body_pos_w = body_pos  # pylint: disable=protected-access
  loader._body_quat_w = body_quat  # pylint: disable=protected-access
  loader._body_lin_vel_w = torch.zeros_like(body_pos)  # pylint: disable=protected-access
  loader._body_ang_vel_w = torch.zeros_like(body_pos)  # pylint: disable=protected-access
  loader._body_indexes = torch.arange(num_tracked_bodies, device=device)  # pylint: disable=protected-access
  loader.body_pos_w = body_pos
  loader.body_quat_w = body_quat
  loader.body_lin_vel_w = torch.zeros_like(body_pos)
  loader.body_ang_vel_w = torch.zeros_like(body_pos)
  loader.time_step_total = num_steps
  return loader
