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

"""Tests for the robot-agnostic Cartesian IK processor."""

from absl.testing import absltest
from absl.testing import parameterized
from mujoco_playground.tasks.agentic.tabletop_v0 import agent
from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment
from mujoco_playground.tasks.agentic.tabletop_v0 import ik_controller
import numpy as np
import torch

# A top-down target per robot, inside its spawn region.
_TARGETS = {
    "so101": (0.25, -0.06, 0.03),
    "yam": (0.34, 0.08, 0.05),
}


def _yaw_err(a: float, b: float) -> float:
  return abs(embodiment.fold(a - b))


class IKControllerTest(parameterized.TestCase):

  def _proc(self, robot):
    return ik_controller.SmoothCartesianIKProcessor(robot=robot)

  def _drive(self, proc, action, steps) -> np.ndarray:
    """Feeds a constant action; the processor sees itself perfectly tracked."""
    keys = proc.embodiment.goal_position_keys
    ctrl = np.zeros(len(keys), dtype=np.float32)
    for _ in range(steps):
      out = proc.process_commands(
          {"action": torch.tensor(action, dtype=torch.float32)}
      )
      ctrl = np.array([float(out[k][0]) for k in keys], dtype=np.float32)
      proc._measured_qpos[0] = ctrl  # pylint: disable=protected-access
    return ctrl

  @parameterized.product(robot=("so101", "yam"), yaw=(0.4, -1.2, 1.5))
  def test_reaches_target_with_requested_yaw_and_pitch(self, robot, yaw):
    proc = self._proc(robot)
    emb = proc.embodiment
    target = [*_TARGETS[robot], 1.0, yaw, ik_controller.PITCH_DOWN]
    ctrl = self._drive(proc, target, steps=150)
    q = proc._current_qpos[0, : emb.n_arm]  # pylint: disable=protected-access
    pos, rot = proc._ik.fk(q)  # pylint: disable=protected-access
    np.testing.assert_allclose(pos, target[:3], atol=3e-3)
    got_yaw, got_pitch = emb.yaw_pitch(pos, rot)
    self.assertLess(_yaw_err(got_yaw, yaw), 0.03)
    self.assertAlmostEqual(got_pitch, ik_controller.PITCH_DOWN, delta=0.05)
    self.assertAlmostEqual(ctrl[-1], emb.gripper_cmd_range[1], delta=1e-5)
    self.assertLen(ctrl, emb.n_arm + 1)

  @parameterized.parameters(
      ("so101", (0.24, 0.0, 0.16)),
      ("yam", (0.34, 0.02, 0.26)),
      ("yam", (0.33, 0.01, 0.20)),
  )
  def test_high_lookout_keeps_position_when_wrist_saturates(self, robot, xyz):
    proc = self._proc(robot)
    emb = proc.embodiment
    yaw = emb.heading(*xyz[:2]) + emb.wrist_yaw_offset
    self._drive(proc, [*xyz, 1.0, yaw, ik_controller.PITCH_DOWN], steps=150)
    q = proc._current_qpos[0, : emb.n_arm]  # pylint: disable=protected-access
    pos, rot = proc._ik.fk(q)  # pylint: disable=protected-access
    np.testing.assert_allclose(pos, xyz, atol=3e-3)

  @parameterized.parameters(
      ("so101", (0.16, 0.08, 0.02)),
      ("yam", (0.246, 0.02, 0.017)),
      ("yam", (0.262, -0.097, 0.017)),
  )
  def test_near_table_height_grasp_is_reached(self, robot, xyz):
    proc = self._proc(robot)
    emb = proc.embodiment
    yaw = emb.heading(*xyz[:2]) + emb.wrist_yaw_offset
    pitch = emb.grasp_pitch(*xyz[:2])
    self._drive(proc, [*xyz[:2], 0.12, 1.0, yaw, pitch], steps=150)
    self._drive(proc, [*xyz, 1.0, yaw, pitch], steps=150)
    q = proc._current_qpos[0, : emb.n_arm]  # pylint: disable=protected-access
    pos, rot = proc._ik.fk(q)  # pylint: disable=protected-access
    np.testing.assert_allclose(pos, xyz, atol=3e-3)

  @parameterized.parameters("so101", "yam")
  def test_executor_plans_track_through_ik(self, robot):
    """Plan executor + IK in closed loop (perfect servos) reach every goal."""
    proc = self._proc(robot)
    emb = proc.embodiment
    ex = agent.PlanExecutor(emb)
    x0 = float(np.mean(emb.spawn_x))

    def obs():
      q = proc._current_qpos[0]  # pylint: disable=protected-access
      pos, rot = proc._ik.fk(q[: emb.n_arm])  # pylint: disable=protected-access
      yaw, pitch = emb.yaw_pitch(pos, rot)
      return {
          "ee_pos": torch.tensor([pos], dtype=torch.float32),
          "ee_ori": torch.tensor([[yaw, pitch]], dtype=torch.float32),
          "gripper_state": torch.tensor([[0.3]], dtype=torch.float32),
          "joint_pos": torch.tensor([q], dtype=torch.float32),
      }

    ex.sync(obs())
    for goal in (
        {"op": "goto", "x": x0, "y": 0.01, "z": 0.12, "pitch": 1.57},
        {"op": "goto", "x": x0 - 0.03, "y": -0.08, "z": 0.1, "yaw_deg": 80},
        {"op": "home"},
    ):
      ex.load([goal], obs())
      for _ in range(1000):
        a = ex.step(obs())
        self._drive(proc, a, steps=1)
        if not ex.busy:
          break
      for _ in range(30):  # Let the rate limiter catch up.
        self._drive(proc, a, steps=1)
      got = obs()["ee_pos"][0].numpy()
      np.testing.assert_allclose(got, a[:3], atol=4e-3, err_msg=str(goal))

  @parameterized.parameters("so101", "yam")
  def test_rate_limits_cartesian_jumps(self, robot):
    proc = self._proc(robot)
    # pylint: disable=protected-access
    start = proc._pose[0, :3].copy()
    far = [*proc.embodiment.workspace_max, 1.0, 0.0, ik_controller.PITCH_DOWN]
    self._drive(proc, far, steps=1)
    step = float(np.linalg.norm(proc._pose[0, :3] - start))
    self.assertLessEqual(
        step, ik_controller.DEFAULT_MAX_LINEAR_SPEED * 0.02 + 1e-9
    )

  @parameterized.parameters("so101", "yam")
  def test_clamps_to_workspace(self, robot):
    proc = self._proc(robot)
    self._drive(
        proc, [0.0, 0.0, -0.1, 0.0, 0.0, ik_controller.PITCH_DOWN], steps=200
    )
    target = proc._pose[0, :3]  # pylint: disable=protected-access
    np.testing.assert_array_less(
        np.asarray(proc.embodiment.workspace_min) - 1e-9, target
    )

  @parameterized.parameters("so101", "yam")
  def test_gripper_command_maps_to_joint_range(self, robot):
    proc = self._proc(robot)
    lo, hi = proc.embodiment.gripper_cmd_range
    home = proc.embodiment.home()
    closed = self._drive(proc, [*home[:3], 0.0, *home[3:]], steps=1)
    self.assertAlmostEqual(closed[-1], lo, delta=1e-6)
    half = self._drive(proc, [*home[:3], 0.5, *home[3:]], steps=1)
    self.assertAlmostEqual(half[-1], 0.5 * (lo + hi), delta=1e-6)

  @parameterized.parameters(
      (0.32, 0.08, 0.06, 0.0, np.deg2rad(20.0), 0.0),
      (0.30, -0.06, 0.07, np.deg2rad(90.0), np.deg2rad(25.0), np.deg2rad(15.0)),
  )
  def test_yam_reaches_full_6dof_non_topdown_pose(
      self, x, y, z, roll, pitch, azimuth
  ):
    proc = self._proc("yam")
    emb = proc.embodiment
    yaw = embodiment.fold(azimuth + roll)
    az_off = embodiment.wrap_pi(azimuth - emb.heading(x, y))
    target = [x, y, z, 1.0, yaw, pitch, az_off]
    self._drive(proc, target, steps=200)
    q = proc._current_qpos[0, : emb.n_arm]  # pylint: disable=protected-access
    pos, rot = proc._ik.fk(q)  # pylint: disable=protected-access
    np.testing.assert_allclose(pos, [x, y, z], atol=3e-3)
    got_yaw, got_pitch, got_az = emb.orientation(pos, rot)
    self.assertLess(_yaw_err(got_yaw, yaw), 0.03)
    self.assertAlmostEqual(got_pitch, pitch, delta=0.03)
    self.assertLess(abs(embodiment.wrap_pi(got_az - azimuth)), 0.03)


if __name__ == "__main__":
  absltest.main()
