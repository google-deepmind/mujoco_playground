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
"""Tests for the REAF Studio ReafViewerPlugin."""

import itertools
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from mujoco.studio import messages
from mujoco.studio import viewer_app
from mujoco.studio import viewer_protocol
import numpy as np
from reaf.simulation.viewer import messages as reaf_messages
from reaf.simulation.viewer import viewer_plugin
import torch

from mujoco.studio.dear_imgui import dear_imgui as imgui  # isort: skip

# A nested spec, as returned by the Torch FunctionalRewardProvider.
_NESTED_SPEC = {
    'reward': dm_env.specs.Array((), np.float32),
    'terms': {
        'height': dm_env.specs.Array((), np.float32),
        'weighted/height': dm_env.specs.Array((), np.float32),
    },
}


def _timestep(obs: Any) -> dm_env.TimeStep:
  return dm_env.TimeStep(dm_env.StepType.FIRST, None, None, obs)


class RewardLegendsTest(absltest.TestCase):

  def test_scalar(self):
    for shape in [(), (1,)]:
      self.assertEqual(
          viewer_plugin.reward_spec_legends(
              dm_env.specs.Array(shape, np.float32)
          ),
          ['reward'],
      )

  def test_vector(self):
    self.assertEqual(
        viewer_plugin.reward_spec_legends(dm_env.specs.Array((3,), np.float32)),
        ['[0]', '[1]', '[2]'],
    )

  def test_nested_dict(self):
    self.assertEqual(
        viewer_plugin.reward_spec_legends(_NESTED_SPEC),
        ['reward', 'terms/height', 'terms/weighted/height'],
    )

  def test_dict_with_vector(self):
    self.assertEqual(
        viewer_plugin.reward_spec_legends(
            {'a': dm_env.specs.Array((2,), np.float32)}
        ),
        ['a[0]', 'a[1]'],
    )

  def test_skips_multidimensional_leaves(self):
    spec = {
        'image': dm_env.specs.Array((2, 2), np.float32),
        'total': dm_env.specs.Array((), np.float32),
    }
    self.assertEqual(viewer_plugin.reward_spec_legends(spec), ['total'])

  def test_keys_select_and_order_legends(self):
    self.assertEqual(
        viewer_plugin.reward_spec_legends(
            _NESTED_SPEC, keys=['terms/height', 'reward']
        ),
        ['terms/height', 'reward'],
    )

  def test_unknown_keys_raise(self):
    with self.assertRaisesRegex(ValueError, 'height'):
      viewer_plugin.reward_spec_legends(_NESTED_SPEC, keys=['height'])


class ReafViewerPluginTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    next_id = itertools.count(1)
    self.viewer = mock.create_autospec(viewer_protocol.Viewer, instance=True)
    self.viewer.upload_image.side_effect = (
        lambda tex_id, img, width, height, bpp: tex_id or next(next_id)
    )
    self.app = viewer_app.ViewerApp()
    self.app.viewer = self.viewer
    self.plugin = viewer_plugin.ReafViewerPlugin()
    self.plugin.on_viewer_app_init(
        viewer_app.ViewerAppInitEvent(viewer_app=self.app)
    )

    self.mock_context = self.enter_context(
        mock.patch.object(imgui, 'GetCurrentContext', return_value=True)
    )
    mock_io = mock.MagicMock()
    mock_io.WantCaptureKeyboard = False
    self.enter_context(mock.patch.object(imgui, 'GetIO', return_value=mock_io))
    self.enter_context(
        mock.patch.object(imgui, 'IsKeyChordPressed', return_value=False)
    )
    self.enter_context(
        mock.patch.object(imgui, 'GetWindowDockID', return_value=42)
    )
    self.mock_set_dock = self.enter_context(
        mock.patch.object(imgui, 'SetNextWindowDockID')
    )
    self.mock_set_focus = self.enter_context(
        mock.patch.object(imgui, 'SetNextWindowFocus')
    )
    self.mock_begin = self.enter_context(
        mock.patch.object(imgui, 'Begin', return_value=(True, True))
    )
    self.mock_end = self.enter_context(mock.patch.object(imgui, 'End'))
    self.mock_tree_node = self.enter_context(
        mock.patch.object(imgui, 'TreeNodeEx', return_value=True)
    )
    self.enter_context(mock.patch.object(imgui, 'TreePop'))
    self.mock_slider_int = self.enter_context(
        mock.patch.object(
            imgui, 'SliderInt', side_effect=lambda label, v, *args: (False, v)
        )
    )
    self.mock_image = self.enter_context(mock.patch.object(imgui, 'Image'))

  def _step(self, obs: Any = None) -> None:
    if obs is None:
      obs = {'camera': torch.zeros((1, 16, 24, 3), dtype=torch.uint8)}
    self.plugin.on_timestep(reaf_messages.TimestepEvent(_timestep(obs)))
    self.plugin.on_update(messages.UpdateEvent())
    self.plugin.build_reaf_gui(messages.BuildGuiEvent())

  def test_no_plot_without_legends(self):
    plugin = viewer_plugin.ReafViewerPlugin()
    plugin.on_timestep(
        reaf_messages.TimestepEvent(dm_env.transition(1.0, None))
    )
    self.assertIsNone(plugin.reward_plot)
    self.assertEqual(plugin.n_steps, 1)

  def test_counts_steps_and_episodes_and_plots_rewards(self):
    plugin = viewer_plugin.ReafViewerPlugin(
        reward_legends=viewer_plugin.reward_spec_legends(_NESTED_SPEC)
    )
    reward = {
        'reward': np.float32(1.0),
        'terms': {'height': np.float32(2.0), 'weighted/height': np.float32(3)},
    }
    events = [
        dm_env.restart(None),
        dm_env.transition(reward, None),
        dm_env.termination(reward, None),
    ]
    for timestep in events:
      plugin.on_timestep(reaf_messages.TimestepEvent(timestep))

    self.assertEqual(plugin.episode_count, 1)
    self.assertEqual(plugin.n_steps, 2)
    assert plugin.reward_plot is not None
    fig = plugin.reward_plot.figure
    self.assertEqual([int(fig.linepnt[i]) for i in range(3)], [2, 2, 2])
    # Each line's latest y value matches its reward entry.
    self.assertEqual([float(fig.linedata[i, 3]) for i in range(3)], [1, 2, 3])

    plugin.on_timestep(reaf_messages.TimestepEvent(dm_env.restart(None)))
    self.assertEqual(plugin.episode_count, 2)
    self.assertEqual(plugin.n_steps, 0)
    self.assertEqual(int(plugin.reward_plot.figure.linepnt[0]), 0)

  def test_timestep_uploads_in_update_and_renders_raw_pixels_in_sidebar(self):
    self._step()
    self.viewer.upload_image.assert_called_once_with(
        0, bytes(24 * 16 * 3), 24, 16, bpp=3
    )
    self.mock_set_dock.assert_called_once_with(42, imgui.Cond.Once)
    self.mock_set_focus.assert_called_once()
    self.mock_begin.assert_any_call('Inspector')
    self.mock_begin.assert_called_with('Obs 2D')
    self.mock_image.assert_called_once()
    tex_id, size = self.mock_image.call_args.args
    self.assertEqual(tex_id, 1)
    self.assertEqual((size.x, size.y), (24, 16))

    # Second step updates existing texture ID in-place without re-querying dock.
    self.mock_begin.reset_mock()
    self._step()
    self.assertEqual(self.viewer.upload_image.call_count, 2)
    self.assertEqual(self.viewer.upload_image.call_args.args[0], 1)
    self.mock_begin.assert_called_once_with('Obs 2D')
    self.mock_set_dock.assert_called_once()
    self.assertEqual(self.mock_set_focus.call_count, 2)

    # Subsequent frame without a timestep event does not upload or refocus.
    self.plugin.on_update(messages.UpdateEvent())
    self.plugin.build_reaf_gui(messages.BuildGuiEvent())
    self.assertEqual(self.viewer.upload_image.call_count, 2)
    self.assertEqual(self.mock_set_focus.call_count, 2)

  def test_multiple_streams_and_lazy_upload(self):
    # Frame 1: cam_a is open, cam_b is collapsed in the sidebar.
    self.mock_tree_node.side_effect = lambda name, flags: name == 'cam_a'
    obs = {
        'cam_a': torch.zeros((1, 16, 16, 3), dtype=torch.uint8),
        'cam_b': torch.zeros((1, 16, 16, 3), dtype=torch.uint8),
    }
    self._step(obs)
    self.assertEqual(self.viewer.upload_image.call_count, 2)

    # Frame 2: only cam_a is open, so a new step only uploads cam_a.
    self._step(obs)
    self.assertEqual(self.viewer.upload_image.call_count, 3)

    # Frame 3: expanding cam_b uploads dirty cam_b on the next update.
    self.mock_tree_node.side_effect = None
    self.mock_tree_node.return_value = True
    self.plugin.build_reaf_gui(messages.BuildGuiEvent())
    self.plugin.on_update(messages.UpdateEvent())
    self.plugin.on_update(messages.UpdateEvent())
    self.assertEqual(self.viewer.upload_image.call_count, 4)

  def test_nearest_neighbor_scaling(self):
    raw = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((1, 2, 3, 3))
    self._step({'camera': raw})
    self.viewer.upload_image.reset_mock()
    self.mock_image.reset_mock()

    # Adjusting the Scale slider marks streams dirty and re-uploads on update.
    self.mock_slider_int.side_effect = None
    self.mock_slider_int.return_value = (True, 4)
    self.plugin.build_reaf_gui(messages.BuildGuiEvent())
    self.mock_slider_int.side_effect = lambda label, v, *args: (False, v)

    self.plugin.on_update(messages.UpdateEvent())
    self.plugin.build_reaf_gui(messages.BuildGuiEvent())

    expected = np.repeat(np.repeat(raw[0], 4, axis=0), 4, axis=1)
    self.viewer.upload_image.assert_called_once_with(
        1, expected.tobytes(), 12, 8, bpp=3
    )
    _, size = self.mock_image.call_args.args
    self.assertEqual((size.x, size.y), (12, 8))

  @parameterized.named_parameters(
      dict(
          testcase_name='rgb_uint8',
          img=torch.full((2, 10, 20, 3), 99, dtype=torch.uint8),
          expected_shape=(20, 10, 3),
          expected_val=99,
      ),
      dict(
          testcase_name='chw_float',
          img=torch.full((2, 3, 16, 16), 0.5, dtype=torch.float32),
          expected_shape=(16, 16, 3),
          expected_val=127,
      ),
      dict(
          testcase_name='chw_stacked',
          img=torch.ones((2, 9, 12, 16), dtype=torch.float32),
          expected_shape=(16, 12, 3),
          expected_val=255,
      ),
      dict(
          testcase_name='gray_2d',
          img=np.full((12, 18), 42, dtype=np.uint8),
          expected_shape=(18, 12, 3),
          expected_val=42,
      ),
      dict(
          testcase_name='gray_3d',
          img=torch.full((2, 8, 14, 1), 7, dtype=torch.uint8),
          expected_shape=(14, 8, 3),
          expected_val=7,
      ),
      dict(
          testcase_name='rgba',
          img=np.full((2, 6, 12, 4), 200, dtype=np.uint8),
          expected_shape=(12, 6, 4),
          expected_val=200,
      ),
      dict(
          testcase_name='extra_leading_dim',
          img=torch.full((2, 1, 8, 10, 3), 55, dtype=torch.uint8),
          expected_shape=(10, 8, 3),
          expected_val=55,
      ),
  )
  def test_image2d_conversion(self, img, expected_shape, expected_val):
    image2d = viewer_plugin.Image2D()
    image2d.set_images({'stream': img})
    image2d.update(self.viewer)

    self.viewer.upload_image.assert_called_once()
    _, img_bytes, width, height = self.viewer.upload_image.call_args.args
    bpp = self.viewer.upload_image.call_args.kwargs['bpp']
    self.assertEqual((width, height, bpp), expected_shape)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    self.assertEqual(
        (int(arr.min()), int(arr.max())), (expected_val, expected_val)
    )

  def test_filtering_and_clearing(self):
    image2d = viewer_plugin.Image2D()
    image2d.set_images({
        'valid': np.zeros((4, 4, 3), dtype=np.uint8),
        'vec1d': np.zeros((7,), dtype=np.uint8),
        'empty': np.zeros((0, 4, 3), dtype=np.uint8),
        'scalar': 1.0,
    })
    image2d.update(self.viewer)
    image2d.build_gui()
    self.assertEqual(
        [c.args[0] for c in self.mock_tree_node.call_args_list], ['valid']
    )

    # Dropping all streams clears the window.
    self.mock_tree_node.reset_mock()
    image2d.set_images({})
    image2d.update(self.viewer)
    image2d.build_gui()
    self.mock_tree_node.assert_not_called()

  def test_camera_rgb_observation_filtering(self):
    self._step({
        'policy': torch.zeros((1, 18), dtype=torch.float32),
        'camera': torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        'front_rgb': torch.zeros((1, 8, 8, 3), dtype=torch.uint8),
        'front_depth': torch.zeros((1, 8, 8), dtype=torch.float32),
        'meta': 'not_an_array',
    })
    self.assertEqual(
        [c.args[0] for c in self.mock_tree_node.call_args_list],
        ['camera', 'front_rgb', 'front_depth'],
    )

  @parameterized.named_parameters(
      dict(testcase_name='single_world_small_grid', shape=(1, 4, 4)),
      dict(testcase_name='three_worlds_three_by_three', shape=(3, 3, 3)),
      dict(testcase_name='four_worlds_four_by_four', shape=(4, 4, 4)),
      dict(testcase_name='multi_world_rectangular_grid', shape=(2, 6, 8)),
  )
  def test_batched_2d_observation_unwrapping(self, shape):
    raw = torch.full(shape, 42, dtype=torch.uint8)
    self._step({'tactile': raw})
    self.viewer.upload_image.assert_called_once()
    _, img_bytes, width, height = self.viewer.upload_image.call_args.args
    self.assertEqual((width, height), (shape[2], shape[1]))
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    self.assertEqual(int(arr.min()), 42)
    self.assertEqual(int(arr.max()), 42)

    # Non-mapping observation produces no window.
    self.mock_tree_node.reset_mock()
    self._step({})
    self._step(torch.zeros((1, 18), dtype=torch.float32))
    self.mock_tree_node.assert_not_called()

  def test_build_gui_early_returns_and_collapsed_window(self):
    # No ImGui context: build_reaf_gui skips Begin.
    self.mock_context.return_value = False
    self._step()
    self.mock_begin.assert_not_called()

    # Collapsed window (Begin=(False, True)) skips TreeNodeEx/Image and clears shown.
    self.mock_context.return_value = True
    self.mock_begin.side_effect = lambda title: (title == 'Inspector', True)
    self.plugin.build_reaf_gui(messages.BuildGuiEvent())
    self.viewer.upload_image.reset_mock()
    self._step()
    self.viewer.upload_image.assert_not_called()
    self.mock_image.assert_not_called()

    # Expanding the window marks streams shown so the next update uploads.
    self.mock_begin.side_effect = None
    self.mock_begin.return_value = (True, True)
    self.plugin.build_reaf_gui(messages.BuildGuiEvent())
    self.plugin.on_update(messages.UpdateEvent())
    self.viewer.upload_image.assert_called_once()
    self.mock_image.reset_mock()
    self.plugin.build_reaf_gui(messages.BuildGuiEvent())
    self.mock_image.assert_called_once()


if __name__ == '__main__':
  absltest.main()
