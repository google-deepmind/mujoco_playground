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

"""CPU test for train.py execution."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import flagsaver
from mujoco_playground.rl.rsl_rl import train


class CpuTrainTest(absltest.TestCase):

  def test_train_one_step_cpu_device(self) -> None:
    """Test train.py runs successfully when explicitly targeting CPU."""
    tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    with flagsaver.flagsaver(
        env="CartpoleBalance",
        device="cpu",
        logdir=os.path.join(tmpdir, "logdir"),
        overrides='{"max_iterations": 1, "nworld": 2, "num_steps_per_env": 8}',
        wp_kernel_cache_dir=os.path.join(tmpdir, "wp_kernel_cache"),
    ):
      train.main([])


if __name__ == "__main__":
  absltest.main()
