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

"""Tests for whole-body tracking REAF environment and components."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujoco_playground.robots.g1.sim import device as g1_device
from mujoco_playground.robots.g1.sim import scene as g1_scene
from mujoco_playground.tasks.whole_body.tracking import motion_command
from mujoco_playground.tasks.whole_body.tracking import task
import numpy as np
from reaf.simulation.torch import flat_scene as build
import torch

os.environ["MUJOCO_GL"] = "osmesa"


def _make_standing_motion_npz(path: str, num_steps: int = 100) -> None:
  """Writes a standing-still motion in the NPZ format."""
  spec = g1_scene.g1_robot_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)

  nbody = model.nbody - 1
  nj = g1_device.NUM_JOINTS
  joint_pos = np.tile(data.qpos[7:].copy(), (num_steps, 1)).astype(np.float32)
  body_pos = np.tile(data.xpos[1:].copy(), (num_steps, 1, 1)).astype(np.float32)
  body_quat = np.tile(data.xquat[1:].copy(), (num_steps, 1, 1)).astype(
      np.float32
  )
  with open(path, "wb") as f:
    np.savez(
        f,
        fps=np.array([50.0]),
        joint_pos=joint_pos,
        joint_vel=np.zeros((num_steps, nj), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((num_steps, nbody, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((num_steps, nbody, 3), dtype=np.float32),
    )


class TrackingEnvTest(parameterized.TestCase):

  def _build(self, num_steps: int = 100, **env_kwargs):
    tmpdir = tempfile.mkdtemp()
    npz_path = os.path.join(tmpdir, "standing.npz")
    _make_standing_motion_npz(npz_path, num_steps=num_steps)
    env_cfg = task.g1_env(trajectory_path=npz_path, **env_kwargs)
    env_cfg.sim.nworld = 1
    env_cfg.sim.xpu_device = "cpu"
    env_cfg.sim.impl = "cpp"
    return build.build_env(env_cfg)

  def test_tracking_env_compiles_and_steps(self):
    built_env = self._build()
    action_spec = built_env.action_spec()
    assert isinstance(action_spec, dict)
    self.assertIn("action", action_spec)
    self.assertEqual(action_spec["action"].shape, (1, 29))

    timestep = built_env.reset()
    obs = timestep.observation
    self.assertIn("policy", obs)
    self.assertIn("critic", obs)
    # Actor: command 58 + anchor_pos_b 3 + anchor_ori_b 6 + lin 3 + ang 3
    # + joint_pos 29 + joint_vel 29 + actions 29 = 160.
    self.assertEqual(obs["policy"].shape, (1, 160))
    # Critic adds body_pos 42 + body_ori 84 = 286.
    self.assertEqual(obs["critic"].shape, (1, 286))

    for _ in range(5):
      action = {"action": torch.zeros((1, 29), dtype=torch.float32)}
      timestep = built_env.step(action)
    self.assertIsNotNone(timestep)
    reward = timestep.reward
    if isinstance(reward, dict):
      reward = reward["reward"]
    self.assertTrue(torch.isfinite(reward).all())

    motion_ref = built_env.task_layer.get_component_by_name("motion")
    self.assertIsInstance(motion_ref, motion_command.MotionReferenceProducer)
    self.assertIn("error_anchor_pos", motion_ref.metrics)
    self.assertIn("sampling_entropy", motion_ref.metrics)

  def test_play_and_ghost_modes(self):
    built_env = self._build(play=True, enable_ghost=True)
    timestep = built_env.reset()
    self.assertIsNotNone(timestep)
    action = {"action": torch.zeros((1, 29), dtype=torch.float32)}
    timestep = built_env.step(action)
    self.assertIsNotNone(timestep)

  def test_without_state_estimation(self):
    built_env = self._build(has_state_estimation=False)
    timestep = built_env.reset()
    obs = timestep.observation
    # Drops anchor_pos_b (3) and base_lin_vel (3) -> 160 - 6 = 154.
    self.assertEqual(obs["policy"].shape, (1, 154))
    self.assertEqual(obs["critic"].shape, (1, 286))

  def test_clip_end_wraps_time_steps(self):
    built_env = self._build(num_steps=4, play=True)
    built_env.reset()
    motion_ref = built_env.task_layer.get_component_by_name("motion")
    self.assertEqual(int(motion_ref.time_steps.item()), 0)
    action = {"action": torch.zeros((1, 29), dtype=torch.float32)}
    for expected_t in (1, 2, 3, 0):
      built_env.step(action)
      self.assertEqual(int(motion_ref.time_steps.item()), expected_t)

  def test_standing_motion_no_immediate_termination(self):
    """Zero actions on a standing reference must survive the short term."""
    torch.manual_seed(0)
    built_env = self._build(play=True)
    built_env.reset()
    for _ in range(20):
      action = {"action": torch.zeros((1, 29), dtype=torch.float32)}
      timestep = built_env.step(action)
      self.assertFalse(
          bool(timestep.step_type == 2),  # LAST
          "standing reference should not terminate immediately",
      )

  def test_ghost_visualization(self):
    built_env = self._build(enable_ghost=True)
    mj_model = built_env.device_layer.sim.mj_model

    ghost_materials = [
        i
        for i in range(mj_model.nmat)
        if mj_model.material(i).name.endswith("_ghost")
    ]
    self.assertNotEmpty(ghost_materials)
    for i in ghost_materials:
      self.assertAlmostEqual(float(mj_model.mat_rgba[i, 3]), 0.3, places=5)

    robot_materials = [
        i
        for i in range(mj_model.nmat)
        if not mj_model.material(i).name.endswith("_ghost")
    ]
    self.assertNotEmpty(robot_materials)
    for i in robot_materials:
      self.assertAlmostEqual(float(mj_model.mat_rgba[i, 3]), 1.0, places=5)


if __name__ == "__main__":
  absltest.main()
