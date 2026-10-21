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

"""Tests for ActionHistory."""

from absl.testing import absltest
from reaf.simulation.torch._src.common import action_history
import torch


class ActionHistoryTest(absltest.TestCase):

  def setUp(self):
    self.xpu_device = "cpu"
    self.nworld = 2
    self.history_length = 3
    self.feature_shape = (4,)
    self.key = "test_ctrl"

    self.history = action_history.ActionHistory(
        name="test_history",
        key=self.key,
        history_length=self.history_length,
        feature_shape=self.feature_shape,
        nworld=self.nworld,
        xpu_device=self.xpu_device,
    )

  def test_consumed_commands_spec(self):
    spec = self.history.consumed_commands_spec()
    self.assertEqual(set(spec.keys()), {self.key})
    self.assertEqual(spec[self.key].shape, self.feature_shape)

  def test_produced_commands_keys(self):
    keys = self.history.produced_commands_keys()
    self.assertEqual(keys, {self.key})

  def test_produced_features_spec(self):
    spec = self.history.produced_features_spec()
    history_key = f"{self.key}_history"
    self.assertEqual(set(spec.keys()), {self.key, history_key})
    self.assertEqual(spec[self.key].shape, self.feature_shape)
    self.assertEqual(
        spec[history_key].shape,
        (self.history_length, *self.feature_shape),
    )

  def test_required_features_keys(self):
    keys = self.history.required_features_keys()
    self.assertEqual(keys, set())

  def test_process_commands_passthrough(self):
    action = torch.ones(
        self.nworld, *self.feature_shape, device=self.xpu_device
    )
    result = self.history.process_commands({self.key: action})
    self.assertEqual(set(result.keys()), {self.key})
    torch.testing.assert_close(result[self.key], action)

  def test_features_after_steps(self):
    """Verify current action and history buffer after processing actions."""
    # Process 3 different actions.
    actions = [torch.full((self.nworld, 4), float(i)) for i in range(1, 4)]
    for a in actions:
      self.history.process_commands({self.key: a})

    features = self.history.produce_features({})
    history_key = f"{self.key}_history"

    # Current action should be the latest (3.0).
    torch.testing.assert_close(features[self.key][0], torch.full((4,), 3.0))

    # History contains PAST raw actions only (not current).
    # After processing [1, 2, 3]: current=3, history=[2, 1, 0].
    history = features[history_key]
    self.assertEqual(history.shape, (self.nworld, self.history_length, 4))
    torch.testing.assert_close(history[0, 0], torch.full((4,), 2.0))
    torch.testing.assert_close(history[0, 1], torch.full((4,), 1.0))
    torch.testing.assert_close(history[0, 2], torch.zeros(4))

  def test_reset_clears_buffer(self):
    action = torch.ones(self.nworld, *self.feature_shape)
    # Process twice so history buffer has nonzero data.
    self.history.process_commands({self.key: action})
    self.history.process_commands({self.key: action * 2.0})

    # Reset env 0 only.
    self.history.reset(
        env_mask=torch.tensor([True, False], device=self.xpu_device)
    )

    features = self.history.produce_features({})
    history_key = f"{self.key}_history"

    # Env 0 current and history should be zeros.
    torch.testing.assert_close(features[self.key][0], torch.zeros(4))
    torch.testing.assert_close(
        features[history_key][0],
        torch.zeros(self.history_length, *self.feature_shape),
    )
    # Env 1 should still have data.
    self.assertGreater(features[history_key][1].sum().item(), 0)

  def test_reset_all_clears_all(self):
    action = torch.ones(self.nworld, *self.feature_shape)
    self.history.process_commands({self.key: action})

    self.history.reset(env_mask=None)

    features = self.history.produce_features({})
    history_key = f"{self.key}_history"
    torch.testing.assert_close(
        features[self.key],
        torch.zeros(self.nworld, *self.feature_shape),
    )
    torch.testing.assert_close(
        features[history_key],
        torch.zeros(self.nworld, self.history_length, *self.feature_shape),
    )

  def test_reset_action_property(self):
    self.assertEqual(
        self.history.reset_action.shape, (self.nworld, *self.feature_shape)
    )
    torch.testing.assert_close(
        self.history.reset_action,
        torch.zeros((self.nworld, *self.feature_shape), device=self.xpu_device),
    )

  def test_reset_fills_with_custom_reset_action(self):
    custom_action = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device=self.xpu_device
    )
    self.history.reset_action.copy_(custom_action)

    action = torch.ones(
        self.nworld, *self.feature_shape, device=self.xpu_device
    )
    self.history.process_commands({self.key: action})
    self.history.process_commands({self.key: action * 2.0})

    # Reset env 0 only.
    self.history.reset(
        env_mask=torch.tensor([True, False], device=self.xpu_device)
    )

    features = self.history.produce_features({})
    history_key = f"{self.key}_history"

    # Env 0 current and history should match custom_action[0].
    torch.testing.assert_close(features[self.key][0], custom_action[0])
    torch.testing.assert_close(
        features[history_key][0],
        custom_action[0, None, ...].expand(
            self.history_length, *self.feature_shape
        ),
    )
    # Env 1 should still have data from process_commands.
    torch.testing.assert_close(features[self.key][1], action[1] * 2.0)

    # Reset all envs.
    self.history.reset(env_mask=None)
    features = self.history.produce_features({})
    torch.testing.assert_close(features[self.key], custom_action)
    torch.testing.assert_close(
        features[history_key],
        custom_action[:, None, ...].expand(
            self.nworld, self.history_length, *self.feature_shape
        ),
    )

  def test_init_with_reset_action(self):
    custom_action = torch.tensor(
        [[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], device=self.xpu_device
    )
    h = action_history.ActionHistory(
        key=self.key,
        feature_shape=self.feature_shape,
        nworld=self.nworld,
        xpu_device=self.xpu_device,
        history_length=self.history_length,
        reset_action=custom_action,
    )
    h.reset()
    features = h.produce_features({})
    history_key = f"{self.key}_history"
    torch.testing.assert_close(features[self.key], custom_action)
    torch.testing.assert_close(
        features[history_key],
        custom_action[:, None, ...].expand(
            self.nworld, self.history_length, *self.feature_shape
        ),
    )

  def test_init_with_wrong_shape_raises(self):
    wrong_shape_action = torch.tensor(
        [1.0, 2.0, 3.0, 4.0], device=self.xpu_device
    )
    with self.assertRaisesRegex(ValueError, "expected"):
      action_history.ActionHistory(
          key=self.key,
          feature_shape=self.feature_shape,
          nworld=self.nworld,
          xpu_device=self.xpu_device,
          reset_action=wrong_shape_action,
      )

  def test_reset_action_partial_envs(self):
    action_env1 = torch.tensor([99.0, 88.0, 77.0, 66.0], device=self.xpu_device)
    self.history.reset_action[1] = action_env1

    # Reset only env 1
    self.history.reset(
        env_mask=torch.tensor([False, True], device=self.xpu_device)
    )
    features = self.history.produce_features({})
    history_key = f"{self.key}_history"

    torch.testing.assert_close(features[self.key][1], action_env1)
    torch.testing.assert_close(
        features[history_key][1],
        action_env1[None, ...].expand(self.history_length, *self.feature_shape),
    )

  def test_no_history(self):
    """history_length=0 produces only the current action feature."""
    h = action_history.ActionHistory(
        key="a",
        feature_shape=(2,),
        nworld=1,
        xpu_device="cpu",
        history_length=0,
    )
    spec = h.produced_features_spec()
    self.assertEqual(set(spec.keys()), {"a"})

    h.process_commands({"a": torch.ones(1, 2)})
    features = h.produce_features({})
    self.assertEqual(set(features.keys()), {"a"})
    torch.testing.assert_close(features["a"], torch.ones(1, 2))


if __name__ == "__main__":
  absltest.main()
