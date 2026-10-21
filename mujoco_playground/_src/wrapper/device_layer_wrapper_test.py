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

"""Tests for NumpyDeviceLayerWrapper."""

from unittest import mock

from absl.testing import absltest
from dm_env import specs
from mujoco_playground._src.wrapper import device_layer_wrapper
import numpy as np
import torch


class FakeTorchDeviceLayer:
  """A device layer whose float measurements are wider than its spec.

  Mirrors the `cpp` backend, which steps in float64 while entities advertise
  float32; on `warp` the two already agree.
  """

  def __init__(self, measurement_dtype: torch.dtype):
    self._dtype = measurement_dtype

  def measurements_spec(self):
    return {
        "joint": specs.Array(shape=(), dtype=np.float32),
        "camera": specs.Array(shape=(2,), dtype=np.uint8),
    }

  def commands_spec(self):
    return {"goal": specs.Array(shape=(), dtype=np.float32)}

  def _measurements(self):
    return {
        "joint": torch.tensor(0.25, dtype=self._dtype),
        "camera": torch.zeros(2, dtype=torch.uint8),
    }

  def begin_stepping(self):
    return self._measurements()

  def step(self, commands):
    del commands
    return self._measurements()


class NumpyDeviceLayerWrapperTest(absltest.TestCase):

  def test_measurements_match_the_advertised_spec(self):
    for name, dtype in (
        ("cpp-like", torch.float64),
        ("warp-like", torch.float32),
    ):
      with self.subTest(name):
        fake_layer: object = FakeTorchDeviceLayer(dtype)
        layer = device_layer_wrapper.NumpyDeviceLayerWrapper(
            fake_layer  # pyrefly: ignore[bad-argument-type]
        )
        spec = layer.measurements_spec()

        for measurements in (layer.begin_stepping(), layer.step({"goal": 0.0})):
          for key, value in measurements.items():
            spec[key].validate(value)

  def test_conforming_preserves_the_value(self):
    layer = device_layer_wrapper.NumpyDeviceLayerWrapper(
        FakeTorchDeviceLayer(torch.float64)  # pyrefly: ignore[bad-argument-type]
    )

    np.testing.assert_allclose(layer.begin_stepping()["joint"], 0.25)

  def test_non_float_measurements_pass_through(self):
    layer = device_layer_wrapper.NumpyDeviceLayerWrapper(
        FakeTorchDeviceLayer(torch.float64)  # pyrefly: ignore[bad-argument-type]
    )

    camera = layer.begin_stepping()["camera"]

    self.assertEqual(camera.dtype, np.uint8)


class TorchDeviceLayerWrapperTest(absltest.TestCase):

  def test_sim_property_returns_none_when_unprovided(self):
    fake_layer = mock.MagicMock()
    wrapped = device_layer_wrapper.TorchDeviceLayerWrapper(
        fake_layer, control_timestep=0.02
    )

    self.assertIsNone(wrapped.sim)
    self.assertIsNone(wrapped.clock)

  def test_sim_property_returns_simulator_when_provided(self):
    fake_layer = mock.MagicMock()
    fake_sim = mock.MagicMock()
    wrapped = device_layer_wrapper.TorchDeviceLayerWrapper(
        fake_layer, sim=fake_sim, control_timestep=0.02
    )

    self.assertIs(wrapped.sim, fake_sim)
    self.assertIs(wrapped.clock, fake_sim.clock)

  def test_get_measurements_returns_cached_measurements(self):
    fake_layer = mock.MagicMock()
    fake_layer.begin_stepping.return_value = {"joint": np.array([0.5])}
    fake_layer.step.return_value = {"joint": np.array([1.0])}
    wrapped = device_layer_wrapper.TorchDeviceLayerWrapper(
        fake_layer, control_timestep=0.02
    )

    # Initial begin_stepping caches measurements.
    _ = wrapped.begin_stepping()
    fake_layer._get_measurements.reset_mock()
    m0 = wrapped.get_measurements()
    self.assertEqual(m0["joint"].item(), 0.5)
    fake_layer._get_measurements.assert_not_called()

    # step() updates cached measurements.
    _ = wrapped.step({"cmd": torch.tensor([0.0])})
    m1 = wrapped.get_measurements()
    self.assertEqual(m1["joint"].item(), 1.0)
    fake_layer._get_measurements.assert_not_called()


if __name__ == "__main__":
  absltest.main()
