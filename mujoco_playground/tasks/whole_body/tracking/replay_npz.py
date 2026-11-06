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

r"""Replay a reference trajectory (.npz) in the MuJoCo viewer.

This is a standard kinematic replay utility that loads joint positions and
body poses from an NPZ file and visualizes them using the raw MuJoCo passive
viewer.
"""

import time

from absl import app
from absl import flags
from etils import epath
import mujoco
import mujoco.viewer
from mujoco_playground import asset
from mujoco_playground.robots.g1.sim import scene as g1_scene
import numpy as np

_TRAJECTORY_PATH = flags.DEFINE_string(
    "trajectory_path",
    None,
    "Path to the .npz reference trajectory file.",
    required=True,
)

_XML_PATH = flags.DEFINE_string(
    "xml_path",
    "",
    "Path to the MuJoCo XML scene file. Empty (default) uses the"
    " G1 scene (g1_scene.g1_scene_spec).",
)


def main(argv):
  del argv

  if _XML_PATH.value:
    print(f"Loading model from: {_XML_PATH.value}")
    spec = asset.get(_XML_PATH.value).spec()
  else:
    print("Loading G1 scene...")
    spec = g1_scene.g1_scene_spec()
  model = spec.compile()
  data = mujoco.MjData(model)

  print(f"Loading trajectory from: {_TRAJECTORY_PATH.value}")
  trajectory_path = epath.Path(_TRAJECTORY_PATH.value)
  with trajectory_path.open("rb") as f:
    npz_data = np.load(f)
    joint_pos = npz_data["joint_pos"]
    body_pos_w = npz_data.get("body_pos_w", None)
    body_quat_w = npz_data.get("body_quat_w", None)

  print("Launching passive viewer...")
  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      for frame in range(joint_pos.shape[0]):
        if not viewer.is_running():
          break

        if body_pos_w is not None and body_quat_w is not None:
          # Assumes 0th index tracks the root/pelvis
          data.qpos[:3] = (
              body_pos_w[frame, 0]
              if body_pos_w.ndim == 3
              else body_pos_w[frame]
          )
          data.qpos[3:7] = (
              body_quat_w[frame, 0]
              if body_quat_w.ndim == 3
              else body_quat_w[frame]
          )

        # Set actuated joints safely
        nj = min(joint_pos.shape[1], model.nq - 7)

        # Datasets like HybridRobotics often export 36 DoF for a 29 DoF robot
        # (keeping root in joints array).
        if joint_pos.shape[1] == 36 and nj == 29:
          data.qpos[7 : 7 + nj] = joint_pos[frame, 7:]
        else:
          data.qpos[7 : 7 + nj] = joint_pos[frame, :nj]

        mujoco.mj_forward(model, data)
        viewer.sync()

        time.sleep(0.02)  # Playback at ~50 FPS

      print("Trajectory ended, looping...")
      time.sleep(1.0)


if __name__ == "__main__":
  app.run(main)
