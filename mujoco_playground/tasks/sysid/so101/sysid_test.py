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

"""Unit tests for SO101 system identification using mujoco.sysid."""

import tempfile
import unittest

from etils import epath
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.sim import constant as sim_constant
from mujoco_playground.tasks.sysid.so101 import data_loader
from mujoco_playground.tasks.sysid.so101 import sysid
import numpy as np


class SysIdTest(unittest.TestCase):

  def test_parameter_setup(self):
    params = sysid.build_parameters()
    for joint in constant.JOINTS:
      self.assertIn(f"{joint}_armature", params)
      self.assertIn(f"{joint}_frictionloss", params)
      self.assertIn(f"{joint}_stiffness", params)
      self.assertIn(f"{joint}_damping", params)

  def test_end_to_end_sysid(self):
    # Generate synthetic trajectory
    n_steps = 20
    dt = 0.02
    timestamps = np.arange(n_steps) * dt

    ctrl = np.zeros((n_steps, len(constant.JOINTS)))
    for step in range(n_steps):
      ctrl[step, 0] = 0.2 * np.sin(2 * np.pi * step / n_steps)

    meas = np.zeros((n_steps, len(constant.JOINTS))) + 0.05

    traj = data_loader.Trajectory(
        path=epath.Path("test_ep.mcap"),
        timestamps=timestamps,
        ctrl_timestamps=timestamps + 0.004,
        dt=dt,
        measured_qpos=meas,
        commanded_ctrl=ctrl,
        measured_qvel=np.zeros((n_steps, len(constant.JOINTS))),
        leader_timestamps_ns=np.arange(n_steps, dtype=np.int64) * 20_000_000,
        follower_timestamps_ns=(
            np.arange(n_steps, dtype=np.int64) * 20_000_000 + 4_000_000
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      save_path = epath.Path(tmp_dir)
      opt_params, opt_result, report_builder = sysid.run_sysid(
          trajectories=[traj],
          save_path=save_path,
          max_iters=3,
          optimizer="mujoco",
          generate_videos=False,
      )

      self.assertIsNotNone(opt_params)
      self.assertIsNotNone(opt_result)
      self.assertTrue((save_path / "report.html").is_file())


if __name__ == "__main__":
  unittest.main()
