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

"""Tests that all registered environments can be loaded."""

import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from mujoco_playground._src.rl import env_registry
import torch


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class EnvRegistryTest(parameterized.TestCase):

  @parameterized.parameters(*env_registry._REGISTRY.keys())
  def test_load_and_step_env(self, env_id: str):
    entry = env_registry.get(env_id)
    env = entry.env_builder(nworld=1, device="cuda:0")
    self.assertIsNotNone(env)

    env.reset()

    action_spec = env.action_spec()
    assert isinstance(action_spec, dict)
    actions = {}
    for key, spec in action_spec.items():
      actions[key] = torch.zeros(spec.shape, device="cuda:0")

    env.step(actions)


class NoCudaTest(absltest.TestCase):

  @mock.patch.object(torch.cuda, "is_available", return_value=False)
  def test_vision_env_on_cpu(self, unused_is_available):
    # Without CUDA the vision env must use the CPU camera.
    entry = env_registry.get("SO101PickCubeVision")
    env = entry.env_builder(nworld=1, device="cpu")
    env.reset()

    action_spec = env.action_spec()
    assert isinstance(action_spec, dict)
    actions = {}
    for key, spec in action_spec.items():
      actions[key] = torch.zeros(spec.shape)

    env.step(actions)


if __name__ == "__main__":
  absltest.main()
