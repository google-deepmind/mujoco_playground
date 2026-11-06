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

"""Tests for the keyboard foot pedal."""

import contextlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from mujoco_playground._src.real.common import foot_pedal


@contextlib.contextmanager
def _fake_terminal():
  """Stands in for the tty `start` puts in cbreak mode, and its reader."""
  stdin = mock.MagicMock(isatty=lambda: True, fileno=lambda: 0)
  with mock.patch.multiple(
      foot_pedal,
      sys=mock.MagicMock(stdin=stdin),
      termios=mock.DEFAULT,
      tty=mock.DEFAULT,
      threading=mock.DEFAULT,
  ):
    yield


class KeyboardFootPedalTest(parameterized.TestCase):

  def test_press_is_latched_then_cleared_by_reading(self):
    pedal = foot_pedal.KeyboardFootPedal()
    pedal.press("d")

    self.assertTrue(pedal.state.right_pressed)
    self.assertFalse(pedal.state.right_pressed)

  def test_every_press_between_reads_is_reported(self):
    # A read covers a window of time, so the pedals are not mutually exclusive.
    pedal = foot_pedal.KeyboardFootPedal()
    pedal.press("a")
    pedal.press("s")

    state = pedal.state

    self.assertTrue(state.left_pressed)
    self.assertTrue(state.middle_pressed)
    self.assertFalse(state.right_pressed)

  def test_unmapped_key_is_ignored(self):
    pedal = foot_pedal.KeyboardFootPedal()
    pedal.press("x")

    state = pedal.state

    self.assertFalse(
        state.left_pressed or state.middle_pressed or state.right_pressed
    )

  def test_custom_keys_map_by_role_ignoring_case(self):
    pedal = foot_pedal.KeyboardFootPedal(
        keys={"left": "1", "middle": "S", "right": "3"}
    )
    pedal.press("s")

    self.assertTrue(pedal.state.middle_pressed)

  @parameterized.named_parameters(
      ("missing_role", {"left": "a", "middle": "s"}, "Expected a key"),
      ("shared_key", {"left": "a", "middle": "a", "right": "d"}, "distinct"),
  )
  def test_bad_key_mapping_raises(self, keys, message):
    with self.assertRaisesRegex(ValueError, message):
      foot_pedal.KeyboardFootPedal(keys=keys)

  def test_start_requires_a_tty(self):
    pedal = foot_pedal.KeyboardFootPedal()

    with mock.patch.object(foot_pedal.sys.stdin, "isatty", return_value=False):
      with self.assertRaisesRegex(RuntimeError, "tty"):
        pedal.start()

  def test_starting_twice_raises(self):
    # Otherwise the second start saves the already-cbreak attributes as the
    # ones to restore, and stop leaves the terminal with no echo.
    pedal = foot_pedal.KeyboardFootPedal()

    with _fake_terminal():
      pedal.start()

      with self.assertRaisesRegex(RuntimeError, "already started"):
        pedal.start()

  def test_can_be_restarted_after_stopping(self):
    pedal = foot_pedal.KeyboardFootPedal()

    with _fake_terminal():
      pedal.start()
      pedal.stop()
      pedal.start()  # Must not raise: the reader's stop signal was cleared.

      self.assertFalse(pedal._stop.is_set())  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
