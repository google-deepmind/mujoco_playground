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

"""Tests the camera observation keeps the aspect ratio the policy trained on."""

from absl.testing import absltest
from mujoco_playground.tasks.manipulation.pick_cube import mdp
import torch

_KEY = "cam_rgb"
_OBS_SIZE = (16, 16)


def camera_obs(rgb: torch.Tensor) -> torch.Tensor:
  return mdp.obs_camera({_KEY: rgb}, camera_key=_KEY, downsample_size=_OBS_SIZE)


class ObsCameraTest(absltest.TestCase):

  def test_square_frame_is_untouched_by_the_crop(self):
    rgb = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)

    obs = camera_obs(rgb)

    self.assertEqual(obs.shape, (1, 3, *_OBS_SIZE))

  def test_wide_frame_is_cropped_not_squashed(self):
    # A 4:3 frame, as the real wrist camera captures it, with the periphery
    # the simulated square render never sees filled in differently.
    rgb = torch.zeros((1, 240, 320, 3), dtype=torch.uint8)
    rgb[:, :, 40:280, :] = 255

    obs = camera_obs(rgb)

    # Only the central square survives, so the observation is uniform rather
    # than a horizontally compressed copy of the whole frame.
    torch.testing.assert_close(obs, torch.ones_like(obs))

  def test_crop_matches_resizing_the_central_square_directly(self):
    rgb = torch.randint(0, 256, (1, 240, 320, 3), dtype=torch.uint8)

    obs = camera_obs(rgb)

    torch.testing.assert_close(obs, camera_obs(rgb[:, :, 40:280, :]))


class RewardTest(absltest.TestCase):

  def test_reward_gripper_close_only_when_near_cube(self):
    features = {
        "ee/xpos": torch.tensor([[0.25, 0.0, 0.02], [0.25, 0.20, 0.02]]),
        "object/xpos": torch.tensor([[0.25, 0.0, 0.02], [0.25, 0.0, 0.02]]),
        "robot_state/joint_pos": torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.15],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.15],
        ]),
    }

    rew = mdp.reward_gripper_close(features)

    self.assertAlmostEqual(float(rew[0]), 1.0, places=4)
    self.assertLess(float(rew[1]), 1e-4)

  def test_reward_gripper_close_zero_when_open(self):
    features = {
        "ee/xpos": torch.tensor([[0.25, 0.0, 0.02]]),
        "object/xpos": torch.tensor([[0.25, 0.0, 0.02]]),
        "robot_state/joint_pos": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.80]]
        ),
    }

    rew = mdp.reward_gripper_close(features)

    self.assertAlmostEqual(float(rew[0]), 0.0, places=5)

  def test_reward_cube_lift_zero_on_table_and_positive_when_lifted(self):
    features = {
        "object/xpos": torch.tensor([[0.25, 0.0, 0.018], [0.25, 0.0, 0.030]]),
        "target/mocap_pos": torch.tensor(
            [[0.25, 0.0, 0.10], [0.25, 0.0, 0.10]]
        ),
    }

    lift_rew = mdp.reward_cube_lift(features, table_z=0.018)
    bring_rew = mdp.reward_cube_to_target(features, table_z=0.018)

    self.assertAlmostEqual(float(lift_rew[0]), 0.0, places=5)
    self.assertGreater(float(lift_rew[1]), 0.1)
    self.assertAlmostEqual(float(bring_rew[0]), 0.0, places=5)
    self.assertGreater(float(bring_rew[1]), 0.0)

  def test_metric_distance_and_cube_lifted(self):
    features = {
        "ee/xpos": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        "object/xpos": torch.tensor(
            [[0.0, 3.0, 4.0], [1.0, 0.0, 0.02], [0.0, 0.0, 0.10]]
        ),
        "target/mocap_pos": torch.tensor(
            [[0.0, 3.0, 4.05], [1.0, 0.0, 0.02], [0.0, 0.0, 0.30]]
        ),
    }
    dist = mdp.metric_distance(
        features, pos_a_key="ee/xpos", pos_b_key="object/xpos"
    )
    lifted = mdp.metric_cube_lifted(
        features, cube_xpos_key="object/xpos", lift_height_threshold=0.04
    )
    success = mdp.reward_success(
        features, distance_threshold=0.08, lift_height_threshold=0.04
    )
    torch.testing.assert_close(dist, torch.tensor([5.0, 0.02, 0.10]))
    torch.testing.assert_close(lifted, torch.tensor([1.0, 0.0, 1.0]))
    # World 0: lifted & close -> 1; world 1: close but not lifted -> 0;
    # world 2: lifted but far -> 0.
    torch.testing.assert_close(success, torch.tensor([1.0, 0.0, 0.0]))


if __name__ == "__main__":
  absltest.main()
