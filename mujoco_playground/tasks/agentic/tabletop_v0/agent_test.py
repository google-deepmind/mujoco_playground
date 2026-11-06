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

"""Tests for tabletop_v0 plan compilation, execution, and oracle simulation."""

import json
import os
import signal
import sys
import tempfile

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
  os.environ["MUJOCO_GL"] = "osmesa"
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/wp_kernel_cache")

# pylint: disable=g-import-not-at-top
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from mujoco_playground.tasks.agentic.tabletop_v0 import agent
from mujoco_playground.tasks.agentic.tabletop_v0 import calibration
from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment
from mujoco_playground.tasks.agentic.tabletop_v0 import env as agentic_env
from mujoco_playground.tasks.agentic.tabletop_v0 import task as agentic_task
import numpy as np
from PIL import Image
import torch

# pylint: enable=g-import-not-at-top


def run_oracle_episode(env, task, robot: str) -> tuple[bool, int]:
  """Runs one deterministic oracle episode; returns (success, steps)."""
  backend = agent.OracleBackend(env, task)
  policy = agent.AgentPolicy(
      backend,
      robot,
      dt=env.device_layer.control_timestep,
  )
  torch.manual_seed(0)
  ts = env.reset()
  policy.reset(ts.observation)
  steps, success, idle = 0, False, 0
  while not ts.last():
    ts = env.step({"action": policy.step(ts.observation, success)})
    steps += 1
    success |= float(np.asarray(ts.reward["reward"]).reshape(-1)[0]) >= 1.0
    idle = idle + 1 if backend.exhausted and not policy.executor.busy else 0
    if success or idle > 100:
      break
  return success, steps


class PlanExecutorTest(parameterized.TestCase):

  def _obs(
      self, xyz, yaw=0.0, pitch=agent.PITCH_DOWN, grip=1.0
  ) -> dict[str, object]:
    return {
        "ee_pos": torch.tensor(np.asarray([xyz], dtype=np.float32)),
        "ee_ori": torch.tensor([[yaw, pitch]], dtype=torch.float32),
        "gripper_state": torch.tensor([[grip]], dtype=torch.float32),
        "joint_pos": torch.zeros((1, 6)),
    }

  def test_grasp_expands_to_smooth_vertical_approach(self):
    ex = agent.PlanExecutor("so101")
    obs = self._obs([0.2, 0.0, 0.15])
    ex.sync(obs)
    ex.load([{"op": "grasp", "x": 0.25, "y": 0.05, "z": 0.015}], obs)
    actions = []
    while ex.busy:
      pose = ex._pose[:3]  # pylint: disable=protected-access
      actions.append(ex.step(self._obs(pose, grip=0.2)))
    actions = np.stack(actions)
    self.assertEqual(ex.status, "done")
    low = actions[actions[:, 2] < 0.05]
    np.testing.assert_allclose(low[:, :2], [[0.25, 0.05]] * len(low), atol=1e-6)
    closed = actions[actions[:, 3] < 0.5]
    self.assertGreater(len(closed), 0)
    self.assertLessEqual(closed[0, 2], 0.02 + 1e-6)
    self.assertLess(actions[-1, 3], 0.5)
    speed = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1) / ex.dt
    self.assertLess(speed.max(), 1.9 * agent.LINEAR_SPEED)

  def test_empty_grasp_aborts_and_reopens(self):
    ex = agent.PlanExecutor("so101")
    obs = self._obs([0.2, 0.0, 0.15])
    ex.sync(obs)
    ex.load(
        [
            {
                "op": "grasp",
                "x": 0.25,
                "y": 0.0,
                "z": 0.015,
                "verify_grasp": True,
            },
            {"op": "home"},
        ],
        obs,
    )
    a = np.zeros(7)
    while ex.busy:
      pose = ex._pose[:3]  # pylint: disable=protected-access
      a = ex.step(self._obs(pose, grip=0.03))
    self.assertEqual(ex.status, "grasp_failed")
    self.assertEqual(a[3], 1.0)
    self.assertGreater(a[2], 0.05)

  def test_code_as_policies_compiles_loops_and_yaw(self):
    code = """```python
bx, by = 0.21, -0.05
for i, (cx, cy, yaw_deg) in enumerate([(0.23, 0.02, 15), (0.19, 0.07, -35)], 1):
  robot.grasp(x=cx, y=cy, z=0.015, yaw_deg=yaw_deg)
  robot.place(x=bx, y=by, z=0.015 + 0.03 * i)
robot.look_at(u=0.5, v=0.5, z=0.12)
robot.home()
```"""
    plan = agent.parse_plan(code)
    self.assertLen(plan, 6)
    self.assertEqual(plan[0]["op"], "grasp")
    self.assertEqual(plan[0]["yaw_deg"], 15)
    self.assertAlmostEqual(plan[1]["z"], 0.045)
    self.assertAlmostEqual(plan[3]["z"], 0.075)
    self.assertEqual(plan[4]["op"], "goto")
    self.assertAlmostEqual(plan[4]["pitch"], agent.PITCH_DOWN)
    self.assertEqual(plan[5], {"op": "home"})

  @parameterized.parameters("so101", "yam")
  def test_resolve_pixels_and_kinematics_round_trip(self, robot):
    emb = embodiment.get(robot)
    qpos = np.asarray(emb.home_qpos)
    kin = agent.kinematics(emb)

    # Pixel <-> world round-trip at the home configuration.
    u0, v0 = kin.image_size[0] * 0.5, kin.image_size[1] * 0.5
    xy = kin.deproject(u0, v0, qpos, z_plane=0.015)
    proj = kin.project(np.array([xy[0], xy[1], 0.015]), qpos)
    assert proj is not None
    np.testing.assert_allclose(proj, (u0, v0), atol=1e-2)

    plan = [
        {"op": "grasp", "u": 330, "v": 250, "z": 0.015, "yaw_deg": 10},
        {"op": "place", "u": 0.5, "v": 0.5, "z": 0.045},
        {"op": "goto", "u": 0.5, "v": 0.5, "z": 0.14},
        {"op": "home"},
    ]
    out = agent.resolve_pixels(plan, qpos, emb)
    np.testing.assert_allclose(
        [out[0]["x"], out[0]["y"]], kin.deproject(330, 250, qpos, 0.015)
    )
    np.testing.assert_allclose(
        [out[1]["x"], out[1]["y"]], kin.deproject(0.5, 0.5, qpos, 0.03)
    )
    np.testing.assert_allclose(
        [out[2]["x"], out[2]["y"]], kin.deproject(0.5, 0.5, qpos, agent.GRID_Z)
    )
    self.assertNotIn("u", out[0])
    self.assertEqual(out[0]["yaw_deg"], 10)
    self.assertEqual(out[3], {"op": "home"})

  def test_move_by_and_gripper_primitives(self):
    ex = agent.PlanExecutor("yam")
    obs = self._obs([0.2, 0.0, 0.12], grip=0.25)
    ex.sync(obs)
    ex.load(
        agent.parse_plan(
            "robot.move_by(dx=0.02, dy=-0.01, dz=-0.03, dyaw_deg=15,"
            " dpitch_deg=-10, speed=0.04)\n"
            "robot.open_gripper()\n"
            "robot.close_gripper()\n"
            "robot.done()"
        ),
        obs,
    )
    actions = []
    while ex.busy:
      pose = ex._pose[:3]  # pylint: disable=protected-access
      ori = ex._pose[3:5]  # pylint: disable=protected-access
      actions.append(
          ex.step(self._obs(pose, yaw=ori[0], pitch=ori[1], grip=0.25))
      )
    actions = np.stack(actions)
    self.assertEqual(ex.status, "done")
    np.testing.assert_allclose(
        ex._pose[:3], [0.22, -0.01, 0.09], atol=1e-5  # pylint: disable=protected-access
    )
    self.assertTrue(np.any(actions[:, 3] > 0.9))
    self.assertEqual(float(actions[-1, 3]), 0.0)

  @parameterized.parameters("so101", "yam")
  def test_verify_grasp_compares_measured_and_commanded_opening(self, robot):
    emb = embodiment.get(robot)
    closed = emb.commanded_opening(0.0)

    def run(code, grip):
      ex = agent.PlanExecutor(robot)
      obs = self._obs([0.22, 0.0, 0.12])
      ex.sync(obs)
      ex.load(agent.parse_plan(code), obs)
      a = np.zeros(7)
      while ex.busy:
        pose = ex._pose[:3]  # pylint: disable=protected-access
        a = ex.step(self._obs(pose, grip=grip))
      return ex, float(a[3])

    ex, a = run("robot.close_gripper(verify_grasp=True)", closed)
    self.assertEqual(ex.status, "grasp_failed")
    self.assertEqual(a, 1.0)
    ex, a = run("robot.close_gripper()", closed)
    self.assertEqual(ex.status, "done")
    self.assertEqual(a, 0.0)
    held = closed + 2 * agent.GRASP_THRESHOLD
    ex, _ = run("robot.close_gripper(verify_grasp=True)", held)
    self.assertEqual(ex.status, "done")
    self.assertTrue(ex.holding(self._obs([0.22, 0.0, 0.12], grip=held)))
    ex, a = run("robot.gripper(0.4)", emb.commanded_opening(0.4))
    self.assertEqual(ex.status, "done")
    self.assertAlmostEqual(a, 0.4, places=6)

  def test_grasp_and_place_opening_and_clearance(self):
    ex = agent.PlanExecutor("yam")
    obs = self._obs([0.30, 0.0, 0.12], grip=0.2)
    ex.sync(obs)
    plan = agent.parse_plan(
        "robot.grasp(0.30, 0.0, 0.015, opening=0.4)\n"
        "robot.place(0.30, 0.1, 0.03, opening=0.3, clearance=0.002)"
    )
    ex.load(plan, obs)
    segs = ex._segments  # pylint: disable=protected-access
    grip = {s.label: s.gripper for s in segs}
    self.assertEqual(grip["grasp:transit"], 0.4)
    self.assertEqual(grip["grasp:descend"], 0.4)
    self.assertEqual(grip["grasp:gripper"], 0.0)
    self.assertEqual(grip["place:descend"], 0.0)
    self.assertEqual(grip["place:gripper"], 0.3)
    descend = next(s for s in segs if s.label == "place:descend")
    self.assertAlmostEqual(
        float(descend.pose[2]),
        max(0.03 + 0.002, ex.embodiment.workspace_min[2]),
        places=5,
    )

  def test_move_until_contact_and_place_until_contact(self):
    ex = agent.PlanExecutor("so101")
    obs = self._obs([0.22, 0.0, 0.10], grip=0.2)
    ex.sync(obs)
    ex.load(
        agent.parse_plan(
            "robot.move_until_contact(dz=-0.08, speed=0.04, max_lag=0.012,"
            " relief=0.002)"
        ),
        obs,
    )
    a = np.zeros(7)
    while ex.busy:
      pose = ex._pose[:3].copy()  # pylint: disable=protected-access
      pose[2] = max(pose[2], 0.05)
      a = ex.step(self._obs(pose, grip=0.2))
    self.assertEqual(ex.status, "contact")
    self.assertTrue(ex.stopped_on_contact)
    self.assertEqual(ex.contact_pos, [0.22, 0.0, 0.05])
    self.assertAlmostEqual(float(a[2]), 0.052, places=3)

  def test_inflight_grasp_slip_during_place_transit(self):
    ex = agent.PlanExecutor("so101")
    obs = self._obs([0.20, -0.05, 0.12], grip=0.2)
    ex.sync(obs)
    ex.load(
        agent.parse_plan(
            "robot.close_gripper(verify_grasp=True)\n"
            "robot.place(x=0.25, y=0.06, z=0.075)"
        ),
        obs,
    )
    ticks, labels = 0, []
    a = np.zeros(7)
    while ex.busy:
      segs = ex._segments  # pylint: disable=protected-access
      labels.append(segs[0].label if segs else "")
      ticks += labels[-1].startswith("place:")
      pose = ex._pose[:3]  # pylint: disable=protected-access
      grip = 0.02 if ticks > 6 else 0.2
      a = ex.step(self._obs(pose, grip=grip))
    self.assertEqual(ex.status, "grasp_failed")
    self.assertEqual(float(a[3]), 1.0)
    self.assertNotIn("place:descend", labels)

  def test_bridge_and_calibration_card(self):
    bridge_dir = self.enterContext(tempfile.TemporaryDirectory())
    cal_out = calibration.generate_card(
        "so101", os.path.join(bridge_dir, "card.png")
    )
    with Image.open(cal_out) as img:
      self.assertEqual(img.size, (640, 480))

    backend = agent.BridgeBackend(
        "test task", bridge_dir, http_port=0, emb="so101", max_turns=1
    )
    try:
      obs = self._obs([0.2, 0.0, 0.15])
      obs["wrist_rgb"] = np.zeros((240, 320, 3), dtype=np.uint8)
      backend.on_reset(obs)
      self.assertFalse(backend.exhausted)
      with backend._lock:  # pylint: disable=protected-access
        backend._pending = [{"op": "home"}]  # pylint: disable=protected-access
      self.assertEqual(backend.plan(obs), [{"op": "home"}])
      self.assertTrue(backend.exhausted)
      with open(os.path.join(bridge_dir, "obs.json"), encoding="utf-8") as f:
        payload = json.load(f)
      self.assertIn("calibration_image", payload)
    finally:
      backend.close()

  def test_shutdown_ignores_sigint_and_keeps_torque_on_failure(self):
    emb = embodiment.get("so101")
    follower = mock.MagicMock()
    follower.get_measurements.return_value = {
        k: np.float32(0.0) for k in emb.joint_position_keys
    }
    camera = mock.MagicMock()
    seen_sigint = []

    def _on_set_commands(cmds):
      del cmds
      seen_sigint.append(signal.getsignal(signal.SIGINT))

    follower.set_commands.side_effect = _on_set_commands
    orig_sigint = signal.getsignal(signal.SIGINT)
    closer = agentic_env.Shutdown(
        follower=follower, emb=emb, camera=camera, duration_secs=0.01, dt=0.01
    )
    closer.close()
    self.assertEqual(seen_sigint, [signal.SIG_IGN])
    self.assertEqual(signal.getsignal(signal.SIGINT), orig_sigint)
    follower.close.assert_called_once()
    camera.close.assert_called_once()

    follower.reset_mock()
    camera.reset_mock()
    follower.get_measurements.side_effect = RuntimeError("bus error")
    closer.close()
    follower.close.assert_not_called()
    camera.close.assert_called_once()


class OracleTest(parameterized.TestCase):

  def _create_sim_env(
      self, *, task: str, robot: str, episode_length_secs: float
  ):
    orig_env_config = agentic_task.env_config

    def _headless_env_config(*args, **kwargs):
      cfg = orig_env_config(*args, **kwargs)
      cfg.device_layer.devices.pop("wrist_cam", None)
      cfg.observation_adapter.pop("wrist_rgb", None)
      return cfg

    with mock.patch.object(
        agentic_task, "env_config", side_effect=_headless_env_config
    ):
      return agentic_env.create_sim_env(
          task=task, robot=robot, episode_length_secs=episode_length_secs
      )

  @parameterized.parameters(*sorted(embodiment.EMBODIMENTS))
  def test_oracle_solves_cube_in_bowl(self, robot):
    task = agentic_task.TASKS["cube_in_bowl"]
    env = self._create_sim_env(
        task="cube_in_bowl", robot=robot, episode_length_secs=60.0
    )
    try:
      success, steps = run_oracle_episode(env, task, robot)
    finally:
      env.close()
    self.assertTrue(
        success, f"{robot}/cube_in_bowl not solved in {steps} steps"
    )


if __name__ == "__main__":
  absltest.main()
