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

"""Tests episode metric reductions and masked resets."""

from absl.testing import absltest
from reaf.simulation.torch._src.builders.flat_scene import config
from reaf.simulation.torch._src.common import functional_metric
from reaf.simulation.torch._src.core.types import MetricReduceMode
import torch


def _scaled(features, *, key: str, scale: float) -> torch.Tensor:
  return features[key] * scale


class FunctionalMetricProviderTest(absltest.TestCase):

  def test_empty_terms_is_noop(self) -> None:
    provider = functional_metric.FunctionalMetricProvider(
        terms={}, nworld=2, xpu_device="cpu"
    )
    mask = torch.tensor([True, False])
    provider.update_metrics({"x": torch.tensor([1.0, 2.0])})
    self.assertEqual(provider.get_episode_metrics(mask), {})
    provider.reset()
    provider.reset(mask)
    self.assertEqual(provider.get_episode_metrics(mask), {})

  def test_reductions_key_order_and_partial_reset(self) -> None:
    # Interleave reduce modes to verify insertion-order preservation and
    # contiguous (K, nworld) buffer packing.
    provider = functional_metric.FunctionalMetricProvider(
        terms={
            "last_m": config.Metric("x", reduce="last"),
            "mean_m": config.Metric("x", reduce="mean"),
            "max_m": config.Metric("x", reduce="max"),
            "sum_m": config.Metric("x", reduce="sum"),
        },
        nworld=2,
        xpu_device="cpu",
    )
    self.assertEqual(provider._last_names, ("last_m",))
    assert provider._mean_sums is not None
    self.assertEqual(provider._mean_sums.shape, (1, 2))
    assert provider._sums is not None
    self.assertEqual(provider._sums.shape, (1, 2))

    w0 = torch.tensor([True, False])
    w1 = torch.tensor([False, True])
    both = torch.tensor([True, True])
    none = torch.tensor([False, False])

    # Before any steps (_maxs == -inf, _step_counts == 0), all modes return 0-D
    # 0.0 without NaN or -inf.
    ep_init = provider.get_episode_metrics(both)
    self.assertEqual(
        list(ep_init.keys()), ["last_m", "mean_m", "max_m", "sum_m"]
    )
    for val in ep_init.values():
      self.assertEqual(val.shape, ())
      torch.testing.assert_close(val, torch.tensor(0.0))

    # Step 1: world 0 = 2.0, world 1 = 10.0
    res = provider.update_metrics({"x": torch.tensor([2.0, 10.0])})
    self.assertIsNone(res)

    # Step 2: world 0 = 6.0, world 1 = 4.0
    provider.update_metrics({"x": torch.tensor([6.0, 4.0])})

    # Per-world values via single-world masks and cross-world mean via `both`.
    ep_w0 = provider.get_episode_metrics(w0)
    self.assertEqual(list(ep_w0.keys()), ["last_m", "mean_m", "max_m", "sum_m"])
    torch.testing.assert_close(ep_w0["mean_m"], torch.tensor(4.0))
    torch.testing.assert_close(ep_w0["sum_m"], torch.tensor(8.0))
    torch.testing.assert_close(ep_w0["max_m"], torch.tensor(6.0))
    torch.testing.assert_close(ep_w0["last_m"], torch.tensor(6.0))

    ep_w1 = provider.get_episode_metrics(w1)
    torch.testing.assert_close(ep_w1["mean_m"], torch.tensor(7.0))
    torch.testing.assert_close(ep_w1["sum_m"], torch.tensor(14.0))
    torch.testing.assert_close(ep_w1["max_m"], torch.tensor(10.0))
    torch.testing.assert_close(ep_w1["last_m"], torch.tensor(4.0))

    ep_both = provider.get_episode_metrics(both)
    torch.testing.assert_close(ep_both["mean_m"], torch.tensor(5.5))
    torch.testing.assert_close(ep_both["sum_m"], torch.tensor(11.0))
    torch.testing.assert_close(ep_both["max_m"], torch.tensor(8.0))
    torch.testing.assert_close(ep_both["last_m"], torch.tensor(5.0))

    # Empty mask (e.g. NO_SYNC step with zero dones) returns finite 0.0 scalars.
    ep_none = provider.get_episode_metrics(none)
    for val in ep_none.values():
      torch.testing.assert_close(val, torch.tensor(0.0))

    # Reset world 0 only (_maxs[:, 0] == -inf -> mapped to 0.0 on read).
    provider.reset(w0)
    for val in provider.get_episode_metrics(w0).values():
      torch.testing.assert_close(val, torch.tensor(0.0))
    ep_w1_after = provider.get_episode_metrics(w1)
    torch.testing.assert_close(ep_w1_after["mean_m"], torch.tensor(7.0))
    torch.testing.assert_close(ep_w1_after["sum_m"], torch.tensor(14.0))
    torch.testing.assert_close(ep_w1_after["max_m"], torch.tensor(10.0))
    torch.testing.assert_close(ep_w1_after["last_m"], torch.tensor(4.0))
    # Previous snapshot `ep_w0` before reset must remain uncorrupted.
    torch.testing.assert_close(ep_w0["sum_m"], torch.tensor(8.0))
    torch.testing.assert_close(ep_w0["last_m"], torch.tensor(6.0))

    # Step 3: world 0 (step 1 of ep 2) = -2.0, world 1 (step 3 of ep 1) = 1.0
    provider.update_metrics({"x": torch.tensor([-2.0, 1.0])})
    ep_w0_s3 = provider.get_episode_metrics(w0)
    torch.testing.assert_close(ep_w0_s3["mean_m"], torch.tensor(-2.0))
    torch.testing.assert_close(ep_w0_s3["sum_m"], torch.tensor(-2.0))
    torch.testing.assert_close(ep_w0_s3["max_m"], torch.tensor(-2.0))
    torch.testing.assert_close(ep_w0_s3["last_m"], torch.tensor(-2.0))

    ep_w1_s3 = provider.get_episode_metrics(w1)
    torch.testing.assert_close(ep_w1_s3["mean_m"], torch.tensor(5.0))
    torch.testing.assert_close(ep_w1_s3["sum_m"], torch.tensor(15.0))
    torch.testing.assert_close(ep_w1_s3["max_m"], torch.tensor(10.0))
    torch.testing.assert_close(ep_w1_s3["last_m"], torch.tensor(1.0))

    # Cross-world mean with unequal step counts (w0=1 step, w1=3 steps):
    # mean of per-world means is (-2.0 + 5.0) / 2 = 1.5 (not pooled 13.0 / 4).
    ep_both_s3 = provider.get_episode_metrics(both)
    torch.testing.assert_close(ep_both_s3["mean_m"], torch.tensor(1.5))
    torch.testing.assert_close(ep_both_s3["sum_m"], torch.tensor(6.5))
    torch.testing.assert_close(ep_both_s3["max_m"], torch.tensor(4.0))
    torch.testing.assert_close(ep_both_s3["last_m"], torch.tensor(-0.5))

  def test_multiple_terms_per_mode_callable_params_and_full_reset(self) -> None:
    scales: dict[str, tuple[float, MetricReduceMode]] = {
        "max_a": (-1.0, "max"),
        "mean_a": (1.0, "mean"),
        "sum_a": (3.0, "sum"),
        "last_a": (6.0, "last"),
        "mean_b": (2.0, "mean"),
        "max_b": (5.0, "max"),
        "last_b": (-2.0, "last"),
        "sum_b": (4.0, "sum"),
    }
    provider = functional_metric.FunctionalMetricProvider(
        terms={
            k: config.Metric(_scaled, key="x", scale=s, reduce=r)
            for k, (s, r) in scales.items()
        },
        nworld=2,
        xpu_device="cpu",
    )
    w0 = torch.tensor([True, False])
    w1 = torch.tensor([False, True])
    provider.update_metrics({"x": torch.tensor([2.0, 5.0])})
    provider.update_metrics({"x": torch.tensor([3.0, 1.0])})
    expected_ep1 = {
        "max_a": (-2.0, -1.0),
        "mean_a": (2.5, 3.0),
        "sum_a": (15.0, 18.0),
        "last_a": (18.0, 6.0),
        "mean_b": (5.0, 6.0),
        "max_b": (15.0, 25.0),
        "last_b": (-6.0, -2.0),
        "sum_b": (20.0, 24.0),
    }
    for mask, i in ((w0, 0), (w1, 1)):
      ep = provider.get_episode_metrics(mask)
      self.assertEqual(list(ep.keys()), list(scales.keys()))
      for k, v in expected_ep1.items():
        torch.testing.assert_close(ep[k], torch.tensor(v[i]))

    # Full reset (env_mask=None) clears all 4 mode buffers and step counts.
    provider.reset()
    for val in provider.get_episode_metrics(
        torch.tensor([True, True])
    ).values():
      torch.testing.assert_close(val, torch.tensor(0.0))

    provider.update_metrics({"x": torch.tensor([1.0, -4.0])})
    expected_ep2 = {
        "max_a": (-1.0, 4.0),
        "mean_a": (1.0, -4.0),
        "sum_a": (3.0, -12.0),
        "last_a": (6.0, -24.0),
        "mean_b": (2.0, -8.0),
        "max_b": (5.0, -20.0),
        "last_b": (-2.0, 8.0),
        "sum_b": (4.0, -16.0),
    }
    for mask, i in ((w0, 0), (w1, 1)):
      ep = provider.get_episode_metrics(mask)
      for k, v in expected_ep2.items():
        torch.testing.assert_close(ep[k], torch.tensor(v[i]))

  def test_read_idempotency_and_buffer_isolation(self) -> None:
    provider = functional_metric.FunctionalMetricProvider(
        terms={
            "mean_m": config.Metric("x", reduce="mean"),
            "sum_m": config.Metric("x", reduce="sum"),
            "max_m": config.Metric("x", reduce="max"),
            "last_m": config.Metric("x", reduce="last"),
        },
        nworld=2,
        xpu_device="cpu",
    )

    provider.update_metrics({"x": torch.tensor([4.0, 8.0])})
    provider.update_metrics({"x": torch.tensor([6.0, 2.0])})
    torch.testing.assert_close(
        provider._step_counts, torch.tensor([2, 2], dtype=torch.long)
    )
    both = torch.tensor([True, True])

    # Calling get_episode_metrics repeatedly is strictly read-only/idempotent
    # and never increments _step_counts or mutates internal buffers.
    for _ in range(5):
      ep_view = provider.get_episode_metrics(both)
      torch.testing.assert_close(ep_view["mean_m"], torch.tensor(5.0))
      torch.testing.assert_close(ep_view["sum_m"], torch.tensor(10.0))
      torch.testing.assert_close(ep_view["max_m"], torch.tensor(7.0))
      torch.testing.assert_close(ep_view["last_m"], torch.tensor(4.0))
      for tensor in ep_view.values():
        tensor.fill_(-999.0)
      ep_view["injected"] = torch.tensor(1.0)

    torch.testing.assert_close(
        provider._step_counts, torch.tensor([2, 2], dtype=torch.long)
    )

    # Next step still accumulates cleanly from unmodified buffers.
    provider.update_metrics({"x": torch.tensor([2.0, 8.0])})
    ep_after = provider.get_episode_metrics(both)
    torch.testing.assert_close(ep_after["mean_m"], torch.tensor(5.0))
    torch.testing.assert_close(ep_after["sum_m"], torch.tensor(15.0))
    torch.testing.assert_close(ep_after["max_m"], torch.tensor(7.0))
    torch.testing.assert_close(ep_after["last_m"], torch.tensor(5.0))

  def test_nan_and_inf_sanitized_without_mutating_features(self) -> None:
    provider = functional_metric.FunctionalMetricProvider(
        terms={
            "mean_m": config.Metric("x", reduce="mean"),
            "sum_m": config.Metric("x", reduce="sum"),
            "max_m": config.Metric("x", reduce="max"),
            "last_m": config.Metric("x", reduce="last"),
        },
        nworld=4,
        xpu_device="cpu",
    )
    raw = torch.tensor([float("nan"), float("inf"), float("-inf"), 4.0])
    provider.update_metrics({"x": raw})
    self.assertTrue(torch.isnan(raw[0]))
    self.assertTrue(torch.isinf(raw[1]))
    self.assertTrue(torch.isneginf(raw[2]))
    ep = provider.get_episode_metrics(torch.tensor([True, True, True, True]))
    for val in ep.values():
      torch.testing.assert_close(val, torch.tensor(1.0))

  def test_shape_validation(self) -> None:
    provider = functional_metric.FunctionalMetricProvider(
        terms={"m": config.Metric("x", reduce="mean")},
        nworld=2,
        xpu_device="cpu",
    )
    with self.assertRaises(ValueError):
      provider.get_episode_metrics(torch.tensor([True, False, True]))
    with self.assertRaises(ValueError):
      provider.update_metrics({"x": torch.zeros((2, 1))})
    with self.assertRaises(ValueError):
      provider.update_metrics({"x": torch.tensor(7.0)})
    with self.assertRaises(ValueError):
      provider.update_metrics({"x": torch.zeros((2, 3))})
    with self.assertRaises(ValueError):
      provider.reset(torch.tensor([True]))
    # Failed update_metrics calls must not increment _step_counts.
    provider.update_metrics({"x": torch.tensor([4.0, 2.0])})
    torch.testing.assert_close(
        provider.get_episode_metrics(torch.tensor([True, True]))["m"],
        torch.tensor(3.0),
    )

  def test_invalid_reduce_mode_raises(self) -> None:
    with self.assertRaises(ValueError):
      functional_metric.FunctionalMetricProvider(
          terms={
              "m": config.Metric(
                  "x",
                  reduce="median",  # pyrefly: ignore[bad-argument-type]
              ),
          },
      )

  def test_cuda_zero_sync(self) -> None:
    if not torch.cuda.is_available():
      self.skipTest("CUDA is not available")
    provider = functional_metric.FunctionalMetricProvider(
        terms={
            "mean_m": config.Metric("x", reduce="mean"),
            "sum_m": config.Metric("x", reduce="sum"),
            "max_m": config.Metric("x", reduce="max"),
            "last_m": config.Metric("x", reduce="last"),
        },
        nworld=2,
        xpu_device="cuda:0",
    )
    x = torch.tensor([2.0, 6.0], device="cuda:0")
    mask = torch.tensor([True, False], device="cuda:0")
    prev_mode = torch.cuda.get_sync_debug_mode()
    self.addCleanup(torch.cuda.set_sync_debug_mode, prev_mode)
    torch.cuda.set_sync_debug_mode("error")
    provider.update_metrics({"x": x})
    ep = provider.get_episode_metrics(mask)
    provider.reset(mask)
    torch.cuda.set_sync_debug_mode(prev_mode)
    torch.testing.assert_close(ep["mean_m"], torch.tensor(2.0, device="cuda:0"))


if __name__ == "__main__":
  absltest.main()
