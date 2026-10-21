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

"""Tests for cosine_interpolate."""

from absl.testing import absltest
from mujoco_playground._src.real.common import motion
import numpy as np


class MotionTest(absltest.TestCase):

  def test_cosine_interpolate_numpy_arrays(self):
    start = np.array([0.0, 10.0], dtype=np.float32)
    target = np.array([1.0, 20.0], dtype=np.float32)
    written: list[np.ndarray] = []

    motion.cosine_interpolate(
        written.append,
        start_pose=start,
        target_pose=target,
        duration_secs=0.1,
        control_timestep=0.02,
        real_time=False,
    )

    self.assertNotEmpty(written)
    self.assertEqual(len(written), 5)
    np.testing.assert_allclose(written[-1], target)

  def test_cosine_interpolate_sequences(self):
    start = [0.0, 5.0]
    target = [2.5, -5.0]
    written: list[np.ndarray] = []

    motion.cosine_interpolate(
        written.append,
        start_pose=start,
        target_pose=target,
        duration_secs=0.05,
        control_timestep=0.01,
        real_time=False,
    )

    self.assertNotEmpty(written)
    np.testing.assert_allclose(written[-1], np.array(target, dtype=np.float32))


if __name__ == "__main__":
  absltest.main()
