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

"""Tests for the episode viewer."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
from mujoco_playground import paths
from mujoco_playground._src.real.common import view_log
import numpy as np


def setUpModule():
  os.environ.setdefault("MUJOCO_GL", "osmesa")


class _Feature:
  """Stands in for the one thing a `tf.Example` feature is asked here."""

  def __init__(self, kind: str, value: tuple[bytes, ...] = ()):
    self._kind = kind
    if kind == "bytes_list":
      self.bytes_list = mock.NonCallableMagicMock()
      self.bytes_list.value = value

  def WhichOneof(self, _) -> str:  # pylint: disable=invalid-name
    return self._kind


class LatestEpisodeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.log_dir = epath.Path(tmp.name)
    patcher = mock.patch.object(paths, "local_dir", return_value=self.log_dir)
    patcher.start()
    self.addCleanup(patcher.stop)

  def _episode(self, name: str, mtime: float) -> epath.Path:
    path = self.log_dir / "2026" / "07" / "27" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"mcap")
    os.utime(path, (mtime, mtime))
    return path

  def test_picks_the_most_recent_episode(self):
    # Episode files are named after a uuid, so the newest is not the last
    # alphabetically — sorting by name would pick an arbitrary episode.
    self._episode("episode_ffff_shard0.mcap", mtime=1000)
    newest = self._episode("episode_0000_shard0.mcap", mtime=2000)

    self.assertEqual(view_log.latest_episode(), newest)

  def test_no_episodes_names_the_directory(self):
    with self.assertRaisesRegex(FileNotFoundError, str(self.log_dir)):
      view_log.latest_episode()


class ViewTypeTest(absltest.TestCase):
  """What a feature's shape and contents say about how to show it."""

  def test_three_dimensional_features_are_images(self):
    # A jpeg wrist frame and a policy's normalized stack, respectively.
    self.assertEqual(
        view_log.view_type(_Feature("bytes_list"), (16, 16, 3)), "image"
    )
    self.assertEqual(
        view_log.view_type(_Feature("float_list"), (9, 16, 16)), "image"
    )

  def test_compressed_jpeg_bytes_are_images_even_with_empty_shape(self):
    self.assertEqual(
        view_log.view_type(
            _Feature("bytes_list", (b"\xff\xd8\xff\xe0\x00\x10JFIF",)), ()
        ),
        "image",
    )

  def test_shapeless_bytes_are_documents(self):
    self.assertEqual(
        view_log.view_type(
            _Feature("bytes_list", (b"pick up the blue cube",)), ()
        ),
        "text",
    )

  def test_everything_else_is_plotted(self):
    self.assertEqual(view_log.view_type(_Feature("float_list"), (21,)), "plot")
    self.assertEqual(view_log.view_type(_Feature("int64_list"), ()), "plot")


class NewestFrameTest(absltest.TestCase):

  def test_takes_the_leading_channels_as_rgb(self):
    # Two 1x1 RGB frames, the current one leading.
    frame = view_log.newest_frame([1.0, 0.5, 0.0, 0.9, 0.9, 0.9], (6, 1, 1))

    self.assertEqual(frame.dtype, np.uint8)
    np.testing.assert_array_equal(frame, [[[255, 127, 0]]])

  def test_handles_hwc_format(self):
    frame = view_log.newest_frame([1.0, 0.5, 0.0], (1, 1, 3))
    self.assertEqual(frame.dtype, np.uint8)
    np.testing.assert_array_equal(frame, [[[255, 127, 0]]])


if __name__ == "__main__":
  absltest.main()
