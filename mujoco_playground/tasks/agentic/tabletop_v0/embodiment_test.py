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

"""Consistency checks every registered embodiment must pass."""

from absl.testing import absltest
from absl.testing import parameterized
from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment
from mujoco_playground.tasks.agentic.tabletop_v0 import ik_controller
import numpy as np

_ROBOTS = sorted(embodiment.EMBODIMENTS)


class EmbodimentTest(parameterized.TestCase):

  @parameterized.product(
      robot=_ROBOTS, yaw=(-1.4, -0.3, 0.0, 0.9), pitch=(1.15, 1.4, np.pi / 2)
  )
  def test_tool_rotation_round_trips(self, robot, yaw, pitch):
    emb = embodiment.get(robot)
    pos = np.array([np.mean(emb.spawn_x), 0.05, 0.05])
    rot = emb.tool_rotation(pos[0], pos[1], yaw, pitch)
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-9)
    self.assertAlmostEqual(np.linalg.det(rot), 1.0, places=9)
    got_yaw, got_pitch = emb.yaw_pitch(pos, rot)
    self.assertLess(abs(embodiment.fold(got_yaw - yaw)), 1e-6)
    self.assertAlmostEqual(got_pitch, pitch, places=6)

  @parameterized.parameters(*_ROBOTS)
  def test_model_has_named_parts(self, robot):
    emb = embodiment.get(robot)
    model = emb.spec_fn().compile()
    for j in emb.mj_joints:
      model.joint(j)
    for j in emb.joints:
      model.actuator(j)
    model.site(emb.pinch_site)
    model.camera(emb.wrist_cam)
    self.assertLen(emb.home_qpos, len(emb.joints))
    self.assertLen(emb.rest_qpos, len(emb.joints))

  @parameterized.parameters(*_ROBOTS)
  def test_home_pose_is_reachable_from_reset(self, robot):
    """`AgentPolicy` moves from `home_qpos` to `home()` after every reset."""
    emb = embodiment.get(robot)
    ik = ik_controller.PinchIK(emb)
    q = np.asarray(emb.home_qpos[: emb.n_arm], dtype=np.float64)
    x, y, z, yaw, pitch = emb.home()
    target = np.array([x, y, z])
    rot = emb.tool_rotation(x, y, yaw, pitch)
    start, _ = ik.fk(q)
    for t in np.linspace(0.0, 1.0, 40):
      q = ik.solve(q, (1 - t) * start + t * target, rot)
    for _ in range(5):
      q = ik.solve(q, target, rot)
    pos, got_rot = ik.fk(q)
    np.testing.assert_allclose(pos, target, atol=0.015)
    got_yaw, got_pitch = emb.yaw_pitch(pos, got_rot)
    self.assertLess(abs(embodiment.fold(got_yaw - yaw)), 0.1)
    self.assertAlmostEqual(got_pitch, pitch, delta=0.15)

  @parameterized.parameters(*_ROBOTS)
  def test_spawn_region_is_reachable_top_down(self, robot):
    emb = embodiment.get(robot)
    ik = ik_controller.PinchIK(emb)
    home = np.asarray(emb.home_qpos[: emb.n_arm])
    for x in emb.spawn_x:
      for y in emb.spawn_y:
        if emb.reach(x, y) > emb.max_reach:
          continue
        z = emb.workspace_min[2] + 0.01
        pitch = emb.grasp_pitch(x, y)
        target = np.array([x, y, z])
        rot = emb.tool_rotation(x, y, emb.heading(x, y), pitch)
        # Walk the target from home like the rate-limited controller does.
        q = home
        start, _ = ik.fk(home)
        for t in np.linspace(0.0, 1.0, 30):
          q = ik.solve(q, (1 - t) * start + t * target, rot)
        pos, _ = ik.fk(q)
        np.testing.assert_allclose(pos, target, atol=5e-3, err_msg=f"{x, y}")


if __name__ == "__main__":
  absltest.main()
