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

r"""Converts a retargeted motion capture CSV into an NPZ trajectory.

The CSV file should contain motion capture data retargeted to the G1 robot.
You can obtain such datasets from Hugging Face, for example:
https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1

Format of the CSV:
- Each row represents a frame.
- Columns:
  - 0:3 - Base position (x, y, z)
  - 3:7 - Base orientation quaternion (x, y, z, w)
  - 7:end - Joint positions (matching the robot's actuated joints)
"""

from absl import app
from absl import flags
from etils import epath
import mujoco
from mujoco_playground.robots.g1.sim import scene as g1_scene
import numpy as np
import scipy.interpolate as scipy_interpolate
import scipy.spatial.transform as scipy_transform

interp1d = scipy_interpolate.interp1d
R = scipy_transform.Rotation
Slerp = scipy_transform.Slerp

_INPUT_CSV = flags.DEFINE_string(
    "input_csv", None, "Path to input CSV file.", required=True
)
_OUTPUT_NPZ = flags.DEFINE_string(
    "output_npz", None, "Path to output NPZ file.", required=True
)
_INPUT_FPS = flags.DEFINE_float("input_fps", 30.0, "FPS of the input CSV.")
_OUTPUT_FPS = flags.DEFINE_float(
    "output_fps", 50.0, "Desired FPS for the output NPZ."
)


def load_and_interpolate(csv_path, in_fps, out_fps):
  """Loads a motion CSV file and interpolates it to the target FPS.

  Args:
    csv_path: Path to the input CSV file.
    in_fps: FPS of the input motion data.
    out_fps: Desired FPS for the output motion data.

  Returns:
    A tuple of (pos, quat, joints, linvel, angvel, jvel) containing the
    interpolated positions, orientations, joint angles, and their respective
    velocities.
  """
  csv_path = epath.Path(csv_path)
  with csv_path.open("r") as f:
    motion = np.loadtxt(f, delimiter=",")

  # Standard format: 0:3 base pos, 3:7 base quat (xyzw), 7:end joints
  base_pos = motion[:, :3]
  base_quat_xyzw = motion[:, 3:7]
  joint_pos = motion[:, 7:]

  frames = motion.shape[0]
  duration = (frames - 1) / in_fps

  t_in = np.linspace(0, duration, frames)
  t_out = np.arange(0, duration, 1.0 / out_fps)

  # Linear interpolation for positions
  pos_interp = interp1d(t_in, base_pos, axis=0, kind="linear")(t_out)
  joint_interp = interp1d(t_in, joint_pos, axis=0, kind="linear")(t_out)

  # Spherical linear interpolation for quaternions
  rots = R.from_quat(base_quat_xyzw)
  slerp = Slerp(t_in, rots)
  rot_interp_xyzw = slerp(t_out).as_quat()
  # Convert xyzw to MuJoCo's wxyz
  rot_interp_wxyz = rot_interp_xyzw[:, [3, 0, 1, 2]]

  # Compute velocities via finite differences
  dt = 1.0 / out_fps
  base_linvel = np.gradient(pos_interp, dt, axis=0)
  joint_vel = np.gradient(joint_interp, dt, axis=0)

  # Compute angular velocity from consecutive quaternions
  rot_diff = (
      R.from_quat(rot_interp_xyzw[1:]) * R.from_quat(rot_interp_xyzw[:-1]).inv()
  )
  angvel = rot_diff.as_rotvec() / dt
  # Pad the first frame to maintain array length
  angvel = np.concatenate([angvel[:1], angvel], axis=0)

  return (
      pos_interp,
      rot_interp_wxyz,
      joint_interp,
      base_linvel,
      angvel,
      joint_vel,
  )


def main(argv):
  del argv

  print(f"Loading motion from {_INPUT_CSV.value}...")
  pos, quat, joints, linvel, angvel, jvel = load_and_interpolate(
      _INPUT_CSV.value, _INPUT_FPS.value, _OUTPUT_FPS.value
  )

  spec = g1_scene.g1_robot_spec()
  model = spec.compile()
  data = mujoco.MjData(model)

  log = {
      "fps": np.array([_OUTPUT_FPS.value]),
      "joint_pos": [],
      "joint_vel": [],
      "body_pos_w": [],
      "body_quat_w": [],
      "body_lin_vel_w": [],
      "body_ang_vel_w": [],
  }

  num_frames = pos.shape[0]
  print(f"Unrolling {num_frames} frames to compute world-frame kinematics...")

  for i in range(num_frames):
    data.qpos[:3] = pos[i]
    data.qpos[3:7] = quat[i]

    nj = min(joints.shape[1], model.nq - 7)
    data.qpos[7 : 7 + nj] = joints[i, :nj]

    data.qvel[:3] = linvel[i]
    data.qvel[3:6] = angvel[i]
    data.qvel[6 : 6 + nj] = jvel[i, :nj]

    # Forward kinematics and body velocities (cvel needs mj_comVel).
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)

    log["joint_pos"].append(data.qpos[7 : 7 + nj].copy())
    log["joint_vel"].append(data.qvel[6 : 6 + nj].copy())
    # Slice [1:] to exclude the worldbody (index 0). This makes pelvis index 0.
    log["body_pos_w"].append(data.xpos[1:].copy())
    log["body_quat_w"].append(data.xquat[1:].copy())
    # Shift cvel (expressed at the subtree CoM of the body's kinematic root)
    # to the body frame origin.
    ref_point = data.subtree_com[model.body_rootid[1:]]
    ang_vel = data.cvel[1:, :3]
    lin_vel = data.cvel[1:, 3:] - np.cross(ang_vel, ref_point - data.xpos[1:])
    log["body_lin_vel_w"].append(lin_vel.copy())
    log["body_ang_vel_w"].append(ang_vel.copy())

  for k in log:
    if k != "fps":
      log[k] = np.stack(log[k], axis=0)  # pyrefly: ignore[no-matching-overload]

  print(f"Saving NPZ to {_OUTPUT_NPZ.value}...")
  output_npz = epath.Path(_OUTPUT_NPZ.value)
  with output_npz.open("wb") as f:
    np.savez(f, **log)
  print("Done!")


if __name__ == "__main__":
  app.run(main)
