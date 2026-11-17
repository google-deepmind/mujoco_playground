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
import dm_env
from mujoco_playground._src.rl import env_registry
from reaf.simulation.torch import environment as reaf_env
import torch
import warp as wp


def _cuda_tensors(root: object) -> dict[str, torch.Tensor]:
  tensors: dict[str, torch.Tensor] = {}
  seen: set[int] = set()

  def _walk(obj: object, path: str) -> None:
    if id(obj) in seen:
      return
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
      tensors[path] = obj
      return
    if isinstance(obj, (str, bytes, int, float, bool)):
      return
    seen.add(id(obj))
    if isinstance(obj, dict):
      for k, v in obj.items():
        _walk(v, f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
      for i, v in enumerate(obj):
        _walk(v, f"{path}[{i}]")
    elif hasattr(obj, "__dict__"):
      for k, v in vars(obj).items():
        if not k.startswith("__"):
          _walk(v, f"{path}.{k}")

  _walk(root, "env")
  return tensors


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class EnvRegistryTest(parameterized.TestCase):

  @unittest.skip("Enable once all RL tasks are sync-free")
  def _check_zero_host_syncs(
      self,
      env: reaf_env.Environment,
      actions: dict[str, torch.Tensor],
      mask: torch.Tensor,
  ) -> None:
    options = env.default_reset_options()
    env.step(actions)
    env.reset_with_options(options=options, env_mask=mask)

    torch.cuda.synchronize()
    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
      env.step(actions)
      env.reset_with_options(options=options, env_mask=mask)
    finally:
      torch.cuda.set_sync_debug_mode(prev_mode)

  @unittest.skip("Enable once all RL tasks preserve tensor pointers")
  def _check_state_pointer_invariance(
      self,
      env: reaf_env.Environment,
      actions: dict[str, torch.Tensor],
      mask: torch.Tensor,
  ) -> None:
    options = env.default_reset_options()
    env.step(actions)
    env.reset_with_options(options=options, env_mask=mask)

    before = _cuda_tensors(env)
    self.assertNotEmpty(before)
    before_ptrs = {k: t.data_ptr() for k, t in before.items()}

    env.step(actions)
    self.assertEqual(
        before_ptrs, {k: t.data_ptr() for k, t in _cuda_tensors(env).items()}
    )

    env.reset_with_options(options=options, env_mask=mask)
    self.assertEqual(
        before_ptrs, {k: t.data_ptr() for k, t in _cuda_tensors(env).items()}
    )

  @unittest.skip("Enable once all RL tasks support CUDA graphs")
  def _check_cuda_graph_replay(
      self,
      env: reaf_env.Environment,
      actions: dict[str, torch.Tensor],
  ) -> None:
    graph_actions = {k: v.clone() for k, v in actions.items()}
    step_actions = [
        {k: torch.full_like(v, 0.1 * (i + 1)) for k, v in actions.items()}
        for i in range(2)
    ]
    torch.manual_seed(0)
    ts0 = env.reset()
    ts0.step_type[0] = dm_env.StepType.LAST
    eager_out = []
    for act in step_actions:
      ts = env.step(act)
      eager_out.append(
          torch.utils._pytree.tree_map(
              lambda x: x.clone(),
              (ts.step_type, ts.reward, ts.discount, ts.observation),
          )
      )

    stream = torch.cuda.Stream(device="cuda:0")
    wp_stream = wp.stream_from_torch(stream)
    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.stream(stream),
        wp.ScopedDevice("cuda:0"),
        wp.ScopedStream(wp_stream),
        torch.cuda.graph(
            graph, stream=stream, capture_error_mode="thread_local"
        ),
        wp.ScopedCapture(
            stream=wp_stream,
            external=True,
            capture_mode=wp.CaptureMode.THREAD_LOCAL,
        ) as warp_capture,
    ):
      captured_ts = env.step(graph_actions)
    stream.synchronize()
    _ = warp_capture.graph

    torch.manual_seed(0)
    ts0 = env.reset()
    ts0.step_type[0] = dm_env.StepType.LAST
    for act, expected in zip(step_actions, eager_out):
      for k, v in act.items():
        graph_actions[k].copy_(v)
      graph.replay()
      actual = (
          captured_ts.step_type,
          captured_ts.reward,
          captured_ts.discount,
          captured_ts.observation,
      )
      torch.testing.assert_close(actual, expected)

  @parameterized.parameters(*env_registry._REGISTRY.keys())
  def test_load_and_step_env(self, env_id: str) -> None:
    entry = env_registry.get(env_id)
    env = entry.env_builder(nworld=2, device="cuda:0")
    self.assertIsNotNone(env)

    env.reset()

    action_spec = env.action_spec()
    assert isinstance(action_spec, dict)
    actions = {
        key: torch.zeros(spec.shape, device="cuda:0")
        for key, spec in action_spec.items()
    }
    mask = torch.tensor([True, False], device="cuda:0")

    env.step(actions)

    # Default AutoResetMode.SYNC checks `is_last_timestep.any()` (a D2H sync)
    # at the end of step(); switch to NO_SYNC after testing the default
    # env_builder() mode so the CUDA-graph checks run without auto-reset syncs.
    env.auto_reset_mode = reaf_env.AutoResetMode.NO_SYNC
    with self.subTest("zero_host_syncs"):
      self._check_zero_host_syncs(env, actions, mask)
    with self.subTest("state_pointer_invariance"):
      self._check_state_pointer_invariance(env, actions, mask)
    with self.subTest("cuda_graph_replay"):
      self._check_cuda_graph_replay(env, actions)


class NoCudaTest(absltest.TestCase):

  @mock.patch.object(torch.cuda, "is_available", return_value=False)
  def test_vision_env_on_cpu(self, unused_is_available) -> None:
    # Without CUDA the vision env must use the CPU camera and default to CPU.
    entry = env_registry.get("SO101PickCubeVision")
    env = entry.env_builder(nworld=1)
    self.assertEqual(env.xpu_device, "cpu")
    env.reset()

    action_spec = env.action_spec()
    assert isinstance(action_spec, dict)
    actions = {}
    for key, spec in action_spec.items():
      actions[key] = torch.zeros(spec.shape)

    env.step(actions)


if __name__ == "__main__":
  absltest.main()
