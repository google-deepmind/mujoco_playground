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

"""Tests the SO-101 pick-cube deploy script."""

import tempfile
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import run as deploy_run


class MainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    flags.FLAGS.mark_as_parsed()
    self.enterContext(
        flagsaver.flagsaver(
            checkpoint=None,
            viewer=True,
            mcap_log_dir=self.enterContext(tempfile.TemporaryDirectory()),
        )
    )

  def test_closes_the_env_when_the_policy_fails_to_load(self):
    env = mock.Mock()
    with (
        mock.patch.object(deploy_run, "create_env", return_value=env),
        mock.patch.object(
            deploy_run,
            "create_policy",
            side_effect=ValueError("bad checkpoint"),
        ),
    ):
      with self.assertRaisesRegex(ValueError, "bad checkpoint"):
        deploy_run.main([])
    env.close.assert_called_once()

  def test_closes_the_env_when_a_logger_fails_to_close(self):
    env = mock.Mock()
    logger = mock.Mock(**{"close.side_effect": OSError("disk full")})
    with (
        mock.patch.object(deploy_run, "create_env", return_value=env),
        mock.patch.object(deploy_run, "create_policy"),
        mock.patch.object(deploy_run, "create_logger", return_value=logger),
        mock.patch.object(deploy_run.deploy_viewer, "launch"),
    ):
      with self.assertRaisesRegex(OSError, "disk full"):
        deploy_run.main([])
    env.close.assert_called_once()


if __name__ == "__main__":
  absltest.main()
