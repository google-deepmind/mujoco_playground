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
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
import numpy as np
from reaf.simulation.torch._src.common import processors as processors_module
from reaf.simulation.torch._src.core import device as device_module
from reaf.simulation.torch._src.core import device_layer
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import substep_commands_processor as substep_commands_processor_module
from reaf.simulation.torch._src.core import substep_measurements_processor as substep_measurements_processor_module
from reaf.simulation.torch._src.core import types
import torch

_TEST_XML = """
    <mujoco model="test">
      <compiler angle="radian" />

      <worldbody>
        <body name="box" pos="0 0 0">
          <inertial pos="0 0 0" mass="0.5" diaginertia="1 1 1"/>
          <freejoint name="box_free_joint"/>
        </body>

        <body name="arm_base" pos="0 0 0">
          <inertial pos="0 0 0" mass="0.5" diaginertia="1 1 1"/>
          <body name="arm_link1" pos="0 0 0.05">
            <inertial pos="0 0 0" mass="0.1" diaginertia="1 1 1"/>
            <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0" range="-90 90"/>
            <body name="arm_link2" pos="0.3 0 0">
              <inertial pos="0 0 0" mass="0.1" diaginertia="1 1 1"/>
              <joint name="joint2" type="hinge" axis="0 1 0" pos="0 0 0" range="-90 90"/>
            </body>
          </body>
        </body>
      </worldbody>

      <actuator>
        <intvelocity name="actuator1" joint="joint1" inheritrange="1"/>
        <intvelocity name="actuator2" joint="joint2" inheritrange="1"/>
      </actuator>
    </mujoco>
"""


class CounterPassthroughDevice(device_module.Device):
  """Device for counting the DeviceLayer substeps."""

  def __init__(self, name: str = "counter_device"):
    self._name = name
    self._substep_commands_counter = 0
    self._substep_measurements_counter = 0

  @property
  def name(self) -> str:
    return self._name

  def substep_commands_spec(self) -> Mapping[str, specs.Array]:
    return {}

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return {}

  def set_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      consumed_substep_commands: Mapping[str, torch.Tensor],
  ) -> None:
    del simulator
    self._substep_commands_counter += 1

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    del simulator
    self._substep_measurements_counter += 1
    return {}

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: Any = None,
  ) -> None:
    del simulator, env_mask
    self._substep_commands_counter = 0
    self._substep_measurements_counter = 0

  def get_substep_commands_counter(self) -> int:
    return int(self._substep_commands_counter)

  def get_substep_measurements_counter(self) -> int:
    return int(self._substep_measurements_counter)


class CounterPassthroughSubstepMeasurementsProcessor(
    substep_measurements_processor_module.SubstepMeasurementsProcessor
):
  """Processor for counting the DeviceLayer substeps."""

  def __init__(self, name: str = "counter_substep_measurements_processor"):
    self._name = name
    self._counter = torch.tensor(0, dtype=torch.int32)

  @property
  def name(self) -> str:
    return self._name

  def consumed_substep_measurements_keys(self) -> set[str]:
    return set()

  def produced_substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return {}

  def process_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
      substep_measurements: MutableMapping[str, torch.Tensor],
  ) -> None:
    del simulator, substep_measurements
    self._counter += 1

  def reset(self, env_mask: Any = None) -> None:
    self._counter = torch.zeros_like(self._counter)

  def get_process_substep_counter(self) -> int:
    return int(self._counter)


class CounterPassthroughSubstepCommandsProcessor(
    substep_commands_processor_module.SubstepCommandsProcessor
):
  """Processor for counting the DeviceLayer substeps."""

  def __init__(self, name: str = "counter_substep_commands_processor"):
    self._name = name
    self._counter = torch.tensor(0, dtype=torch.int32)

  @property
  def name(self) -> str:
    return self._name

  def consumed_substep_commands_spec(self) -> Mapping[str, specs.Array]:
    return {}

  def produced_substep_commands_keys(self) -> set[str]:
    return set()

  def process_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      substep_commands: MutableMapping[str, torch.Tensor],
  ) -> None:
    del simulator, substep_commands
    self._counter += 1

  def reset(self, env_mask: Any = None) -> None:
    self._counter = torch.zeros_like(self._counter)

  def get_process_substep_counter(self) -> int:
    return int(self._counter)


_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_SPEC = specs.Array(shape=(1,), dtype=np.float32)


def _nworld(impl: types.Impl) -> int:
  return 1 if impl == types.Impl.CPP else 2


def _make_device_layer(
    impl: types.Impl,
    xpu_device: str,
    *,
    n_substeps: int = 1,
    devices: Sequence[device_module.Device] = (),
    processors: Sequence[
        substep_measurements_processor_module.SubstepMeasurementsProcessor
    ] = (),
) -> device_layer.DeviceLayer:
  """Returns a device layer running `n_substeps` substeps of 2 ms per step."""
  sim = simulator_module.Simulator(
      mj_spec=mujoco.MjSpec.from_string(_TEST_XML),
      impl=impl,
      xpu_device=xpu_device,
      nworld=_nworld(impl),
  )
  return device_layer.DeviceLayer(
      simulator=sim,
      control_timestep=0.002 * n_substeps,
      devices=devices,
      substep_measurements_processors=processors,
  )


class _Device(device_module.Device):
  """Device measuring `keys`; subclasses override the measurement methods."""

  def __init__(self, name: str, keys: Sequence[str] = ()):
    self._name = name
    self._keys = tuple(keys) or (name,)

  @property
  def name(self) -> str:
    return self._name

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return {key: _SPEC for key in self._keys}

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask


class _Processor(
    substep_measurements_processor_module.SubstepMeasurementsProcessor
):
  """Processor replacing `consumed` by `produced`; subclasses override hooks."""

  def __init__(self, consumed: str, produced: str):
    self.consumed = consumed
    self.produced = produced

  @property
  def name(self) -> str:
    return self.produced

  def consumed_substep_measurements_keys(self) -> set[str]:
    return {self.consumed}

  def produced_substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return {self.produced: _SPEC}

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    del env_mask


@parameterized.named_parameters(
    dict(
        testcase_name="warp_backend",
        impl=types.Impl.WARP,
        xpu_device=_DEVICE,
    ),
    dict(
        testcase_name="cpp_backend",
        impl=types.Impl.CPP,
        xpu_device="cpu",
    ),
)
class DeviceLayerTest(parameterized.TestCase):

  def test_initialization_specs_computation(self, impl, xpu_device):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("WARP backend requires CUDA.")

    test_mj_spec = mujoco.MjSpec.from_string(_TEST_XML)
    nworld = 1 if impl == types.Impl.CPP else 2

    spec_a = specs.Array(shape=(1,), dtype=np.float32)
    spec_b = specs.Array(shape=(2,), dtype=np.int32)
    spec_c = specs.Array(shape=(3,), dtype=np.float32)
    spec_d = specs.Array(shape=(4,), dtype=np.bool_)
    spec_e = specs.Array(shape=(1,), dtype=np.float32)
    spec_f = specs.Array(shape=(1,), dtype=np.float32)

    # --- Setup for testing Measurements Spec ---
    mock_device_m1 = mock.create_autospec(device_module.Device, instance=True)
    mock_device_m1.name = "device_m1"
    mock_device_m1.substep_commands_spec.return_value = {}
    mock_device_m1.substep_measurements_spec.return_value = {
        "meas_e1": spec_a,
        "meas_shared": spec_c,
    }

    mock_device_m2 = mock.create_autospec(device_module.Device, instance=True)
    mock_device_m2.name = "device_m2"
    mock_device_m2.substep_commands_spec.return_value = {}
    mock_device_m2.substep_measurements_spec.return_value = {"meas_e2": spec_b}

    mock_m_proc1 = mock.create_autospec(
        substep_measurements_processor_module.SubstepMeasurementsProcessor,
        instance=True,
    )
    mock_m_proc1.name = "mproc1"
    mock_m_proc1.consumed_substep_measurements_keys.return_value = {"meas_e1"}
    mock_m_proc1.produced_substep_measurements_spec.return_value = {
        "meas_mp1": spec_d
    }
    mock_m_proc1.process_substep_measurements.return_value = {
        "meas_mp1": np.zeros_like(spec_d)
    }

    mock_m_proc2 = mock.create_autospec(
        substep_measurements_processor_module.SubstepMeasurementsProcessor,
        instance=True,
    )
    mock_m_proc2.name = "mproc2"
    mock_m_proc2.consumed_substep_measurements_keys.return_value = {
        "meas_shared"
    }
    mock_m_proc2.produced_substep_measurements_spec.return_value = {
        "meas_mp2": spec_e
    }
    mock_m_proc2.process_substep_measurements.return_value = {
        "meas_mp2": np.zeros_like(spec_e)
    }

    expected_measurements_spec = {
        "meas_e2": spec_b.replace(shape=(nworld,) + spec_b.shape),
        "meas_mp1": spec_d.replace(shape=(nworld,) + spec_d.shape),
        "meas_mp2": spec_e.replace(shape=(nworld,) + spec_e.shape),
    }

    # --- Setup for testing Commands Spec ---
    mock_device_c1 = mock.create_autospec(device_module.Device, instance=True)
    mock_device_c1.name = "device_c1"
    mock_device_c1.substep_commands_spec.return_value = {
        "cmd_e1": spec_a,
        "cmd_shared_by_devices": spec_f,
    }
    mock_device_c1.substep_measurements_spec.return_value = {}

    mock_device_c2 = mock.create_autospec(device_module.Device, instance=True)
    mock_device_c2.name = "device_c2"
    mock_device_c2.substep_commands_spec.return_value = {"cmd_e2": spec_b}
    mock_device_c2.substep_measurements_spec.return_value = {}

    mock_cp1 = mock.create_autospec(
        substep_commands_processor_module.SubstepCommandsProcessor,
        instance=True,
    )
    mock_cp1.name = "cproc1"
    mock_cp1.consumed_substep_commands_spec.return_value = {
        "cmd_cp1_in": spec_c
    }
    mock_cp1.produced_substep_commands_keys.return_value = {"cmd_e1"}
    mock_cp1.process_substep_commands.return_value = {
        "cmd_e1": np.zeros((1,), dtype=np.float32)
    }

    mock_cp2 = mock.create_autospec(
        substep_commands_processor_module.SubstepCommandsProcessor,
        instance=True,
    )
    mock_cp2.name = "cproc2"
    mock_cp2.consumed_substep_commands_spec.return_value = {
        "cmd_cp2_in": spec_e
    }
    mock_cp2.produced_substep_commands_keys.return_value = {
        "cmd_shared_by_devices"
    }
    mock_cp2.process_substep_commands.return_value = {
        "cmd_shared_by_devices": np.zeros((1,), dtype=np.float32)
    }

    expected_commands_spec = {
        "cmd_e2": spec_b.replace(shape=(nworld,) + spec_b.shape),
        "cmd_cp1_in": spec_c.replace(shape=(nworld,) + spec_c.shape),
        "cmd_cp2_in": spec_e.replace(shape=(nworld,) + spec_e.shape),
    }

    sim = simulator_module.Simulator(
        mj_spec=test_mj_spec,
        impl=impl,
        xpu_device=xpu_device,
        nworld=nworld,
    )

    # --- Test Measurements Spec Computation ---
    dl_meas = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.002,
        substep_commands_processors=[],
        devices=[mock_device_m1, mock_device_m2],
        substep_measurements_processors=[mock_m_proc1, mock_m_proc2],
    )
    self.assertEqual(dl_meas.measurements_spec(), expected_measurements_spec)

    # --- Test Commands Spec Computation ---
    dl_cmd = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.002,
        substep_commands_processors=[mock_cp1, mock_cp2],
        devices=[mock_device_c1, mock_device_c2],
        substep_measurements_processors=[],
    )
    self.assertEqual(dl_cmd.commands_spec(), expected_commands_spec)

  def test_device_returns_correct_measurements(self, impl, xpu_device):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("WARP backend requires CUDA.")

    test_mj_spec = mujoco.MjSpec.from_string(_TEST_XML)
    nworld = 1 if impl == types.Impl.CPP else 2

    spec_m1 = specs.Array(shape=(1,), dtype=np.float32)
    spec_m2 = specs.Array(shape=(2,), dtype=np.int32)
    spec_m_shared = specs.Array(shape=(1,), dtype=np.float32)
    spec_p1_out = specs.Array(shape=(3,), dtype=np.bool_)
    spec_p2_out = specs.Array(shape=(4,), dtype=np.float32)
    spec_m3_passthrough = specs.Array(shape=(1,), dtype=np.float32)

    val_m1 = torch.tensor([1.5], dtype=torch.float32, device=xpu_device)
    val_m2 = torch.tensor([-10, -20], dtype=torch.int32, device=xpu_device)
    val_m_shared = torch.tensor([0.5], dtype=torch.float32, device=xpu_device)
    val_p1_out = torch.tensor(
        [False, True, False], dtype=torch.bool, device=xpu_device
    )
    val_p2_out = torch.tensor(
        [-0.1, -0.2, -0.3, -0.4], dtype=torch.float32, device=xpu_device
    )
    val_m3_passthrough = torch.tensor(
        [100.0], dtype=torch.float32, device=xpu_device
    )

    mock_device1 = mock.create_autospec(device_module.Device, instance=True)
    mock_device1.name = "device1_meas_test"
    mock_device1.substep_commands_spec.return_value = {}
    mock_device1.substep_measurements_spec.return_value = {
        "e1_meas": spec_m1,
        "shared_meas": spec_m_shared,
    }
    mock_device1.get_substep_measurements.return_value = {
        "e1_meas": val_m1,
        "shared_meas": val_m_shared,
    }

    mock_device2 = mock.create_autospec(device_module.Device, instance=True)
    mock_device2.name = "device2_meas_test"
    mock_device2.substep_commands_spec.return_value = {}
    mock_device2.substep_measurements_spec.return_value = {"e2_meas": spec_m2}
    mock_device2.get_substep_measurements.return_value = {"e2_meas": val_m2}

    mock_device3_passthrough = mock.create_autospec(
        device_module.Device, instance=True
    )
    mock_device3_passthrough.name = "device3_passthrough_test"
    mock_device3_passthrough.substep_commands_spec.return_value = {}
    mock_device3_passthrough.substep_measurements_spec.return_value = {
        "e3_passthrough": spec_m3_passthrough
    }
    mock_device3_passthrough.get_substep_measurements.return_value = {
        "e3_passthrough": val_m3_passthrough
    }

    mock_proc1 = mock.create_autospec(
        substep_measurements_processor_module.SubstepMeasurementsProcessor,
        instance=True,
    )
    mock_proc1.name = "proc1_meas_test"
    mock_proc1.consumed_substep_measurements_keys.return_value = {"e1_meas"}
    mock_proc1.produced_substep_measurements_spec.return_value = {
        "proc1_output": spec_p1_out
    }
    recorded_meas_proc1 = {}

    def proc1_side_effect(sim, meas):
      del sim
      recorded_meas_proc1.update(meas)
      meas.pop("e1_meas")
      meas["proc1_output"] = val_p1_out

    mock_proc1.process_substep_measurements.side_effect = proc1_side_effect
    expected_consumed_meas_proc1 = {"e1_meas": val_m1}

    mock_proc2 = mock.create_autospec(
        substep_measurements_processor_module.SubstepMeasurementsProcessor,
        instance=True,
    )
    mock_proc2.name = "proc2_meas_test"
    mock_proc2.consumed_substep_measurements_keys.return_value = {"shared_meas"}
    mock_proc2.produced_substep_measurements_spec.return_value = {
        "proc2_output": spec_p2_out
    }
    recorded_meas_proc2 = {}

    def proc2_side_effect(sim, meas):
      del sim
      recorded_meas_proc2.update(meas)
      meas.pop("shared_meas")
      meas["proc2_output"] = val_p2_out

    mock_proc2.process_substep_measurements.side_effect = proc2_side_effect
    expected_consumed_meas_proc2 = {"shared_meas": val_m_shared}

    expected_measurements = {
        "e2_meas": val_m2,
        "e3_passthrough": val_m3_passthrough,
        "proc1_output": val_p1_out,
        "proc2_output": val_p2_out,
    }

    sim = simulator_module.Simulator(
        mj_spec=test_mj_spec,
        impl=impl,
        xpu_device=xpu_device,
        nworld=nworld,
    )
    dl = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.002,
        substep_commands_processors=[],
        devices=[mock_device1, mock_device2, mock_device3_passthrough],
        substep_measurements_processors=[mock_proc1, mock_proc2],
    )

    dl.step({})
    actual_measurements = dl.get_measurements()

    mock_device1.get_substep_measurements.assert_called()
    mock_device2.get_substep_measurements.assert_called()
    mock_device3_passthrough.get_substep_measurements.assert_called()

    # Verify calls to mock_proc1.process_substep_measurements
    mock_proc1.process_substep_measurements.assert_called()
    for key, expected_arr in expected_consumed_meas_proc1.items():
      self.assertIn(key, recorded_meas_proc1)
      actual_arr = recorded_meas_proc1[key]
      self.assertIsInstance(actual_arr, torch.Tensor)
      self.assertTrue(torch.equal(actual_arr, expected_arr))
      self.assertEqual(actual_arr.dtype, expected_arr.dtype)

    # Verify calls to mock_proc2.process_substep_measurements
    mock_proc2.process_substep_measurements.assert_called()
    for key, expected_arr in expected_consumed_meas_proc2.items():
      self.assertIn(key, recorded_meas_proc2)
      actual_arr = recorded_meas_proc2[key]
      self.assertIsInstance(actual_arr, torch.Tensor)
      self.assertTrue(torch.equal(actual_arr, expected_arr))
      self.assertEqual(actual_arr.dtype, expected_arr.dtype)

    # Verify final measurements
    self.assertEqual(
        set(actual_measurements.keys()),
        set(expected_measurements.keys()),
    )
    for key, expected_arr in expected_measurements.items():
      self.assertIn(key, actual_measurements)
      actual_arr = actual_measurements[key]
      self.assertIsInstance(actual_arr, torch.Tensor)
      self.assertTrue(torch.equal(actual_arr, expected_arr))
      self.assertEqual(actual_arr.dtype, expected_arr.dtype)

  def test_device_propagates_commands(self, impl, xpu_device):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("WARP backend requires CUDA.")

    test_mj_spec = mujoco.MjSpec.from_string(_TEST_XML)
    nworld = 1 if impl == types.Impl.CPP else 2

    spec_cmd_dev_in_cp1 = specs.Array(shape=(1,), dtype=np.float32)
    spec_cmd_cp1_out_cp2_in = specs.Array(shape=(2,), dtype=np.int32)
    spec_cmd_cp2_out_e1_in = specs.Array(shape=(3,), dtype=np.bool_)
    spec_cmd_dev_in_e2 = specs.Array(shape=(1,), dtype=np.float32)

    val_dev_in_cp1 = torch.tensor(
        [10.0], dtype=torch.float32, device=xpu_device
    )
    val_cp1_out_cp2_in = torch.tensor(
        [1, 2], dtype=torch.int32, device=xpu_device
    )
    val_cp2_out_e1_in = torch.tensor(
        [True, False, True], dtype=torch.bool, device=xpu_device
    )
    val_dev_in_e2 = torch.tensor([20.0], dtype=torch.float32, device=xpu_device)

    mock_cp1 = mock.create_autospec(
        substep_commands_processor_module.SubstepCommandsProcessor,
        instance=True,
    )
    mock_cp1.name = "cp1_cmd_test"
    mock_cp1.consumed_substep_commands_spec.return_value = {
        "cmd_from_device_for_cp1": spec_cmd_dev_in_cp1
    }
    mock_cp1.produced_substep_commands_keys.return_value = {
        "cmd_cp1_produces_for_cp2"
    }
    recorded_cmds_cp1 = {}

    def cp1_side_effect(sim, cmds):
      del sim
      recorded_cmds_cp1.update(cmds)
      cmds.pop("cmd_from_device_for_cp1")
      cmds["cmd_cp1_produces_for_cp2"] = val_cp1_out_cp2_in

    mock_cp1.process_substep_commands.side_effect = cp1_side_effect
    expected_consumed_cmds_cp1 = {"cmd_from_device_for_cp1": val_dev_in_cp1}

    mock_cp2 = mock.create_autospec(
        substep_commands_processor_module.SubstepCommandsProcessor,
        instance=True,
    )
    mock_cp2.name = "cp2_cmd_test"
    mock_cp2.consumed_substep_commands_spec.return_value = {
        "cmd_cp1_produces_for_cp2": spec_cmd_cp1_out_cp2_in
    }
    mock_cp2.produced_substep_commands_keys.return_value = {
        "cmd_cp2_produces_for_e1"
    }
    recorded_cmds_cp2 = {}

    def cp2_side_effect(sim, cmds):
      del sim
      recorded_cmds_cp2.update(cmds)
      cmds.pop("cmd_cp1_produces_for_cp2")
      cmds["cmd_cp2_produces_for_e1"] = val_cp2_out_e1_in

    mock_cp2.process_substep_commands.side_effect = cp2_side_effect
    expected_consumed_cmds_cp2 = {
        "cmd_cp1_produces_for_cp2": val_cp1_out_cp2_in
    }

    mock_device1 = mock.create_autospec(device_module.Device, instance=True)
    mock_device1.name = "device1_cmd_test"
    mock_device1.substep_commands_spec.return_value = {
        "cmd_cp2_produces_for_e1": spec_cmd_cp2_out_e1_in
    }
    mock_device1.substep_measurements_spec.return_value = {}
    expected_commands_for_device1 = {
        "cmd_cp2_produces_for_e1": val_cp2_out_e1_in
    }

    mock_device2 = mock.create_autospec(device_module.Device, instance=True)
    mock_device2.name = "device2_cmd_test"
    mock_device2.substep_commands_spec.return_value = {
        "cmd_direct_from_device_for_e2": spec_cmd_dev_in_e2
    }
    mock_device2.substep_measurements_spec.return_value = {}
    expected_commands_for_device2 = {
        "cmd_direct_from_device_for_e2": val_dev_in_e2
    }

    sim = simulator_module.Simulator(
        mj_spec=test_mj_spec,
        impl=impl,
        xpu_device=xpu_device,
        nworld=nworld,
    )
    dl = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.002,
        substep_commands_processors=[mock_cp1, mock_cp2],
        devices=[mock_device1, mock_device2],
        substep_measurements_processors=[],
    )

    commands_to_set = {
        "cmd_from_device_for_cp1": val_dev_in_cp1,
        "cmd_direct_from_device_for_e2": val_dev_in_e2,
    }

    dl._propagate_commands(commands_to_set)

    # Verify calls to mock_cp1.process_substep_commands
    mock_cp1.process_substep_commands.assert_called_once()
    for key, expected_arr in expected_consumed_cmds_cp1.items():
      self.assertIn(key, recorded_cmds_cp1)
      actual_arr = recorded_cmds_cp1[key]
      self.assertIsInstance(actual_arr, torch.Tensor)
      self.assertTrue(torch.equal(actual_arr, expected_arr))
      self.assertEqual(actual_arr.dtype, expected_arr.dtype)

    # Verify calls to mock_cp2.process_substep_commands
    mock_cp2.process_substep_commands.assert_called_once()
    for key, expected_arr in expected_consumed_cmds_cp2.items():
      self.assertIn(key, recorded_cmds_cp2)
      actual_arr = recorded_cmds_cp2[key]
      self.assertIsInstance(actual_arr, torch.Tensor)
      self.assertTrue(torch.equal(actual_arr, expected_arr))
      self.assertEqual(actual_arr.dtype, expected_arr.dtype)

    # Verify calls to mock_device1.set_substep_commands
    mock_device1.set_substep_commands.assert_called_once()
    args, _ = mock_device1.set_substep_commands.call_args
    actual_consumed_cmds_e1 = args[1]
    self.assertIsInstance(actual_consumed_cmds_e1, Mapping)
    for key, expected_arr in expected_commands_for_device1.items():
      self.assertIn(key, actual_consumed_cmds_e1)
      actual_arr = actual_consumed_cmds_e1[key]
      self.assertIsInstance(actual_arr, torch.Tensor)
      self.assertTrue(
          torch.equal(actual_arr, expected_arr),
          f"Array mismatch for device1 command '{key}'",
      )
      self.assertEqual(actual_arr.dtype, expected_arr.dtype)

    # Verify calls to mock_device2.set_substep_commands
    mock_device2.set_substep_commands.assert_called_once()
    args, _ = mock_device2.set_substep_commands.call_args
    actual_consumed_cmds_e2 = args[1]
    self.assertIsInstance(actual_consumed_cmds_e2, Mapping)
    for key, expected_arr in expected_commands_for_device2.items():
      self.assertIn(key, actual_consumed_cmds_e2)
      actual_arr = actual_consumed_cmds_e2[key]
      self.assertIsInstance(actual_arr, torch.Tensor)
      self.assertTrue(
          torch.equal(actual_arr, expected_arr),
          f"Array mismatch for device2 command '{key}'",
      )
      self.assertEqual(actual_arr.dtype, expected_arr.dtype)

  def test_device_graph_capture(self, impl, xpu_device):
    if impl != types.Impl.WARP:
      self.skipTest("Graph capture is only supported for the WARP backend.")
    if not torch.cuda.is_available() or "cpu" in xpu_device:
      self.skipTest("Graph capture requires CUDA.")

    test_mj_spec = mujoco.MjSpec.from_string(_TEST_XML)
    dev = CounterPassthroughDevice()
    cp = CounterPassthroughSubstepCommandsProcessor()
    mp = CounterPassthroughSubstepMeasurementsProcessor()

    sim = simulator_module.Simulator(
        mj_spec=test_mj_spec,
        nworld=2,
        xpu_device=xpu_device,
        impl=types.Impl.WARP,
    )
    dl = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.002,
        substep_commands_processors=[cp],
        devices=[dev],
        substep_measurements_processors=[mp],
    )

    dl.sim.forward()

    # Ensure CUDA graphs were captured.
    self.assertSameElements(
        tuple(dl.sim._graph_captures.keys()), ("forward", "step", "reset_data")
    )
    old_graphs = list(dl.sim._graph_captures.values())
    dl.sim.forward()
    self.assertSameElements(old_graphs, list(dl.sim._graph_captures.values()))

  def test_mjwarp_model_tiling_invalidates_graphs(self, impl, xpu_device):
    if impl != types.Impl.WARP:
      self.skipTest("Graph capture is only supported for the WARP backend.")
    if not torch.cuda.is_available() or "cpu" in xpu_device:
      self.skipTest("Graph capture requires CUDA.")

    spec = mujoco.MjSpec.from_string(_TEST_XML)
    sim = simulator_module.Simulator(
        mj_spec=spec,
        nworld=2,
        xpu_device=xpu_device,
        impl=types.Impl.WARP,
    )
    dl = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.002,
    )

    # Capture graphs.
    dl.sim.forward()
    expected_graphs = {"forward", "step", "reset_data"}
    self.assertEqual(set(dl.sim._graph_captures.keys()), expected_graphs)
    original_graphs = dict(dl.sim._graph_captures)

    _ = dl.sim.model.actuator_gear  # Trigger tiling of stride 0 field.
    self.assertFalse(
        dl.sim._graph_captures,
        "Graph captures cache should be cleared after tiling.",
    )

    # Check that graph is recaptured and different than original graphs.
    dl.sim.forward()
    self.assertEqual(set(dl.sim._graph_captures.keys()), expected_graphs)
    self.assertNotEqual(
        dl.sim._graph_captures["forward"],
        original_graphs["forward"],
        "New graph capture should be different than original capture.",
    )

  def test_device_step_calls_devices_and_processors(self, impl, xpu_device):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("WARP backend requires CUDA.")

    test_mj_spec = mujoco.MjSpec.from_string(_TEST_XML)
    dev = CounterPassthroughDevice()
    cp = CounterPassthroughSubstepCommandsProcessor()
    mp = CounterPassthroughSubstepMeasurementsProcessor()

    sim = simulator_module.Simulator(
        mj_spec=test_mj_spec,
        nworld=1 if impl == types.Impl.CPP else 2,
        xpu_device=xpu_device,
        impl=impl,
    )
    dl = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.01,
        substep_commands_processors=[cp],
        devices=[dev],
        substep_measurements_processors=[mp],
    )

    # On begin_stepping, components are reset and initial measurements taken.
    dl.begin_stepping()
    self.assertEqual(cp.get_process_substep_counter(), 0)
    self.assertEqual(dev.get_substep_commands_counter(), 0)
    self.assertEqual(dev.get_substep_measurements_counter(), 1)
    self.assertEqual(mp.get_process_substep_counter(), 1)

    # On forward, nothing should increase.
    dl.sim.forward()
    self.assertEqual(cp.get_process_substep_counter(), 0)
    self.assertEqual(dev.get_substep_commands_counter(), 0)
    self.assertEqual(dev.get_substep_measurements_counter(), 1)
    self.assertEqual(mp.get_process_substep_counter(), 1)

    # On step({}), with control_timestep 0.01 and sim timestep 0.002, 5 substeps occur.
    initial_time = dl.sim.data.time[0].clone()
    dl.step({})
    new_time = dl.sim.data.time[0]
    expected_time = initial_time + dl.sim.model.opt.timestep[0] * 5
    torch.testing.assert_close(
        new_time.to(torch.float32), expected_time.to(torch.float32)
    )
    self.assertEqual(cp.get_process_substep_counter(), 5)
    self.assertEqual(dev.get_substep_commands_counter(), 5)
    self.assertEqual(dev.get_substep_measurements_counter(), 6)
    self.assertEqual(mp.get_process_substep_counter(), 6)

    # On reset, simulation time resets to zero.
    dl.reset()
    torch.testing.assert_close(
        dl.sim.data.time[0].to(torch.float32),
        torch.tensor(0.0, device=dl.xpu_device, dtype=torch.float32),
    )

  def test_raises_value_error_for_invalid_timestep(self, impl, xpu_device):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("WARP backend requires CUDA.")

    test_mj_spec = mujoco.MjSpec.from_string(_TEST_XML)
    sim = simulator_module.Simulator(
        mj_spec=test_mj_spec,
        impl=impl,
        xpu_device=xpu_device,
    )
    # 0.021 is not a multiple of default physics timestep 0.002
    invalid_control_timestep = 0.021
    with self.assertRaises(ValueError):
      device_layer.DeviceLayer(
          simulator=sim,
          control_timestep=invalid_control_timestep,
      )

  def test_get_device_and_has_device(self, impl, xpu_device):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("WARP backend requires CUDA.")

    test_mj_spec = mujoco.MjSpec.from_string(_TEST_XML)
    dev1 = CounterPassthroughDevice(name="dummy1")
    dev2 = CounterPassthroughDevice(name="dummy2")

    sim = simulator_module.Simulator(
        mj_spec=test_mj_spec,
        impl=impl,
        xpu_device=xpu_device,
    )
    dl = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.002,
        devices=[dev1, dev2],
    )

    self.assertTrue(dl.has_device("dummy1"))
    self.assertTrue(dl.has_device("dummy2"))
    self.assertFalse(dl.has_device("non_existent"))

    self.assertEqual(dl.get_device("dummy1"), dev1)
    self.assertEqual(dl.get_device("dummy2"), dev2)

    with self.assertRaises(ValueError):
      dl.get_device("non_existent")

  def test_call_graph_lifecycle_invariants(self, impl, xpu_device):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("WARP backend requires CUDA.")

    trace: list[str] = []

    class TracingDevice(device_module.Device):

      def __init__(self, name: str = "tracing_device"):
        self._name = name

      @property
      def name(self) -> str:
        return self._name

      def substep_commands_spec(self) -> Mapping[str, specs.Array]:
        return {"dev_cmd": specs.Array(shape=(1,), dtype=np.float32)}

      def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
        return {"dev_sub_meas": specs.Array(shape=(1,), dtype=np.float32)}

      def set_substep_commands(
          self,
          simulator: simulator_module.Simulator,
          consumed_substep_commands: Mapping[str, torch.Tensor],
      ) -> None:
        del simulator, consumed_substep_commands
        trace.append("dev.set_substep_commands")

      def get_substep_measurements(
          self,
          simulator: simulator_module.Simulator,
      ) -> Mapping[str, torch.Tensor]:
        del simulator
        trace.append("dev.get_substep_measurements")
        return {
            "dev_sub_meas": torch.zeros(
                (1,), dtype=torch.float32, device=xpu_device
            )
        }

      def reset(
          self,
          simulator: simulator_module.Simulator,
          env_mask: Any = None,
      ) -> None:
        del simulator, env_mask
        trace.append("dev.reset")

    class TracingCommandsProcessor(
        substep_commands_processor_module.SubstepCommandsProcessor
    ):

      @property
      def name(self) -> str:
        return "tracing_cp"

      def consumed_substep_commands_spec(self) -> Mapping[str, specs.Array]:
        return {"raw_cmd": specs.Array(shape=(1,), dtype=np.float32)}

      def produced_substep_commands_keys(self) -> set[str]:
        return {"dev_cmd"}

      def process_substep_commands(
          self,
          simulator: simulator_module.Simulator,
          substep_commands: MutableMapping[str, torch.Tensor],
      ) -> None:
        del simulator
        trace.append("cp.process_substep_commands")
        if "raw_cmd" in substep_commands:
          substep_commands["dev_cmd"] = substep_commands.pop("raw_cmd")

      def reset(self, env_mask: Any = None) -> None:
        trace.append("cp.reset")

    class TracingMeasurementsProcessor(
        substep_measurements_processor_module.SubstepMeasurementsProcessor
    ):

      @property
      def name(self) -> str:
        return "tracing_mp"

      def consumed_substep_measurements_keys(self) -> set[str]:
        return {"dev_sub_meas"}

      def produced_substep_measurements_spec(self) -> Mapping[str, specs.Array]:
        return {"proc_meas": specs.Array(shape=(1,), dtype=np.float32)}

      def process_substep_measurements(
          self,
          simulator: simulator_module.Simulator,
          substep_measurements: MutableMapping[str, torch.Tensor],
      ) -> None:
        del simulator
        trace.append("mp.process_substep_measurements")
        if "dev_sub_meas" in substep_measurements:
          substep_measurements["proc_meas"] = substep_measurements.pop(
              "dev_sub_meas"
          )

      def reset(self, env_mask: Any = None) -> None:
        trace.append("mp.reset")

    dev = TracingDevice(name="tracing_dev")
    cp = TracingCommandsProcessor()
    mp = TracingMeasurementsProcessor()

    test_mj_spec = mujoco.MjSpec.from_string(_TEST_XML)
    sim = simulator_module.Simulator(
        mj_spec=test_mj_spec,
        impl=impl,
        xpu_device=xpu_device,
        nworld=1 if impl == types.Impl.CPP else 2,
    )

    # control_timestep = 0.006, sim_timestep = 0.002 -> 3 substeps
    dl = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.006,
        substep_commands_processors=[cp],
        devices=[dev],
        substep_measurements_processors=[mp],
    )

    # --- Phase 1: begin_stepping ---
    trace.clear()
    dl.begin_stepping()
    expected_begin_trace = [
        "dev.get_substep_measurements",
        "mp.process_substep_measurements",
    ]
    self.assertEqual(trace, expected_begin_trace)

    # --- Phase 2: step(commands) ---
    trace.clear()
    cmd_dict = {
        "raw_cmd": torch.tensor([1.0], device=xpu_device),
    }
    dl.step(cmd_dict)
    expected_step_trace = [
        # Substep 0
        "cp.process_substep_commands",
        "dev.set_substep_commands",
        "dev.get_substep_measurements",
        "mp.process_substep_measurements",
        # Substep 1
        "cp.process_substep_commands",
        "dev.set_substep_commands",
        "dev.get_substep_measurements",
        "mp.process_substep_measurements",
        # Substep 2
        "cp.process_substep_commands",
        "dev.set_substep_commands",
        "dev.get_substep_measurements",
        "mp.process_substep_measurements",
    ]
    self.assertEqual(trace, expected_step_trace)

    # --- Phase 3: reset ---
    trace.clear()
    dl.reset()
    expected_reset_trace = [
        "dev.reset",
        "cp.reset",
        "mp.reset",
    ]
    self.assertEqual(trace, expected_reset_trace)

  def test_numerical_trajectory_equivalence_and_dataflow_invariants(
      self, impl, xpu_device
  ):
    if impl == types.Impl.WARP and not torch.cuda.is_available():
      self.skipTest("WARP backend requires CUDA.")

    nworld = 1 if impl == types.Impl.CPP else 2

    class ActuatedArmDevice(device_module.Device):

      def __init__(self, name: str = "actuated_arm"):
        self._name = name

      @property
      def name(self) -> str:
        return self._name

      def reset(
          self,
          simulator: simulator_module.Simulator,
          env_mask: Any = None,
      ) -> None:
        del simulator, env_mask

      def substep_commands_spec(self) -> Mapping[str, specs.Array]:
        return {"arm_ctrl": specs.Array(shape=(2,), dtype=np.float32)}

      def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
        return {"raw_qpos": specs.Array(shape=(2,), dtype=np.float32)}

      def set_substep_commands(
          self,
          simulator: simulator_module.Simulator,
          consumed_substep_commands: Mapping[str, torch.Tensor],
      ) -> None:
        if "arm_ctrl" in consumed_substep_commands:
          ctrl_val = consumed_substep_commands["arm_ctrl"]
          simulator.data.ctrl[:, :2] = ctrl_val

      def get_substep_measurements(
          self,
          simulator: simulator_module.Simulator,
      ) -> Mapping[str, torch.Tensor]:
        # Indices 7 and 8 are joint1 and joint2 qpos (freejoint has 7)
        return {"raw_qpos": simulator.data.qpos[:, 7:9].clone()}

    class HalfScaleCommandsProcessor(
        substep_commands_processor_module.SubstepCommandsProcessor
    ):

      @property
      def name(self) -> str:
        return "scale_cp"

      def consumed_substep_commands_spec(self) -> Mapping[str, specs.Array]:
        return {"target_ctrl": specs.Array(shape=(2,), dtype=np.float32)}

      def produced_substep_commands_keys(self) -> set[str]:
        return {"arm_ctrl"}

      def process_substep_commands(
          self,
          simulator: simulator_module.Simulator,
          substep_commands: MutableMapping[str, torch.Tensor],
      ) -> None:
        if "target_ctrl" in substep_commands:
          val = substep_commands.pop("target_ctrl")
          substep_commands["arm_ctrl"] = val * 0.5

      def reset(self, env_mask: Any = None) -> None:
        pass

    class DegreeMeasurementsProcessor(
        substep_measurements_processor_module.SubstepMeasurementsProcessor
    ):

      @property
      def name(self) -> str:
        return "deg_mp"

      def consumed_substep_measurements_keys(self) -> set[str]:
        return {"raw_qpos"}

      def produced_substep_measurements_spec(self) -> Mapping[str, specs.Array]:
        return {"deg_qpos": specs.Array(shape=(2,), dtype=np.float32)}

      def process_substep_measurements(
          self,
          simulator: simulator_module.Simulator,
          substep_measurements: MutableMapping[str, torch.Tensor],
      ) -> None:
        del simulator
        if "raw_qpos" in substep_measurements:
          raw = substep_measurements.pop("raw_qpos")
          substep_measurements["deg_qpos"] = raw * 57.2957795

      def reset(self, env_mask: Any = None) -> None:
        pass

    test_mj_spec = mujoco.MjSpec.from_string(_TEST_XML)
    sim = simulator_module.Simulator(
        mj_spec=test_mj_spec,
        impl=impl,
        xpu_device=xpu_device,
        nworld=nworld,
    )
    dl = device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.01,  # 5 substeps
        substep_commands_processors=[HalfScaleCommandsProcessor()],
        devices=[ActuatedArmDevice(name="arm")],
        substep_measurements_processors=[DegreeMeasurementsProcessor()],
    )

    # 1. Verification of caller dictionary immutability
    original_target = torch.tensor(
        [[1.5, -1.0]] * nworld, dtype=torch.float32, device=xpu_device
    )
    original_mode = torch.tensor(
        [[1]] * nworld, dtype=torch.int32, device=xpu_device
    )
    caller_commands = {
        "target_ctrl": original_target.clone(),
        "mode": original_mode.clone(),
    }

    dl.begin_stepping()
    first_meas = dl.step(caller_commands)

    # Assert caller's dict and tensors were not mutated by the step() call
    self.assertIn("target_ctrl", caller_commands)
    self.assertIn("mode", caller_commands)
    self.assertTrue(
        torch.equal(caller_commands["target_ctrl"], original_target)
    )
    self.assertTrue(torch.equal(caller_commands["mode"], original_mode))

    # Assert intermediate keys were isolated and returned measurements match spec
    self.assertIn("deg_qpos", first_meas)
    self.assertNotIn("raw_qpos", first_meas)
    self.assertNotIn("target_ctrl", first_meas)
    self.assertNotIn("arm_ctrl", first_meas)

    # 2. Run rollout 1 for 20 steps and record trajectory
    rollout1_qpos = []
    rollout1_deg_meas = []
    dl.reset()
    dl.begin_stepping()

    for step_idx in range(20):
      action = torch.sin(
          torch.tensor(
              [[step_idx * 0.2, -step_idx * 0.1]] * nworld, device=xpu_device
          )
      )
      meas = dl.step({"target_ctrl": action, "mode": original_mode})
      rollout1_qpos.append(dl.sim.data.qpos[:, 7:9].clone())
      rollout1_deg_meas.append(meas["deg_qpos"].clone())

    # 3. Run rollout 2 from reset with identical actions and verify bit-for-bit replay
    dl.reset()
    dl.begin_stepping()

    for step_idx in range(20):
      action = torch.sin(
          torch.tensor(
              [[step_idx * 0.2, -step_idx * 0.1]] * nworld, device=xpu_device
          )
      )
      meas = dl.step({"target_ctrl": action, "mode": original_mode})
      qpos_now = dl.sim.data.qpos[:, 7:9]
      deg_meas_now = meas["deg_qpos"]

      # Strict bit-for-bit trajectory replay check
      self.assertTrue(
          torch.equal(qpos_now, rollout1_qpos[step_idx]),
          f"Trajectory discrepancy at step {step_idx}",
      )
      self.assertTrue(
          torch.equal(deg_meas_now, rollout1_deg_meas[step_idx]),
          f"Measurement discrepancy at step {step_idx}",
      )

  def test_split_devices_and_partial_reset(self, impl, xpu_device):
    nworld = _nworld(impl)

    class SubstepCounterDevice(_Device):
      """Counts substeps per world in `on_substep`; reads have no effects."""

      def __init__(self, name: str):
        super().__init__(name)
        self.counts = torch.zeros((nworld, 1), device=xpu_device)
        self.num_on_substep = 0
        self.num_on_step = 0
        self.num_reads = 0
        self.begin_stepping_masks = []

      def on_substep(self, simulator: simulator_module.Simulator) -> None:
        del simulator
        self.num_on_substep += 1
        self.counts += 1.0

      def on_step(self, simulator: simulator_module.Simulator) -> None:
        del simulator
        self.num_on_step += 1

      def on_begin_stepping(
          self,
          simulator: simulator_module.Simulator,
          env_mask: torch.Tensor | None = None,
      ) -> None:
        del simulator
        self.begin_stepping_masks.append(env_mask)

      def get_measurements(
          self, simulator: simulator_module.Simulator
      ) -> Mapping[str, torch.Tensor]:
        del simulator
        self.num_reads += 1
        return {self.name: self.counts.clone()}

      def reset(
          self,
          simulator: simulator_module.Simulator,
          env_mask: torch.Tensor | None = None,
      ) -> None:
        del simulator
        self.counts[slice(None) if env_mask is None else env_mask] = 0.0

    class SumProcessor(_Processor):
      """Sums `a` over substeps into `a_sum`."""

      def __init__(self):
        super().__init__("a", "a_sum")
        self.sums = torch.zeros((nworld, 1), device=xpu_device)
        self.begin_stepping_calls = []
        self.num_process = 0

      def on_substep(
          self,
          simulator: simulator_module.Simulator,
          substep_measurements: Mapping[str, torch.Tensor],
      ) -> None:
        del simulator
        self.sums += substep_measurements["a"]

      def on_begin_stepping(
          self,
          simulator: simulator_module.Simulator,
          measurements: Mapping[str, torch.Tensor],
          env_mask: torch.Tensor | None = None,
      ) -> None:
        del simulator
        self.begin_stepping_calls.append((measurements["a"].clone(), env_mask))

      def process_measurements(
          self,
          simulator: simulator_module.Simulator,
          measurements: MutableMapping[str, torch.Tensor],
      ) -> None:
        del simulator
        self.num_process += 1
        measurements.pop("a")
        measurements["a_sum"] = self.sums.clone()

      def reset(self, env_mask: torch.Tensor | None = None) -> None:
        self.sums[slice(None) if env_mask is None else env_mask] = 0.0

    dev_a = SubstepCounterDevice("a")
    dev_b = SubstepCounterDevice("b")
    sum_processor = SumProcessor()
    dl = _make_device_layer(
        impl,
        xpu_device,
        n_substeps=3,
        devices=[dev_a, dev_b],
        processors=[sum_processor],
    )

    def full(value: float) -> torch.Tensor:
      return torch.full((nworld, 1), value, device=xpu_device)

    meas = dl.begin_stepping()
    torch.testing.assert_close(meas["a_sum"], full(0.0))
    torch.testing.assert_close(meas["b"], full(0.0))

    meas = dl.step({})
    torch.testing.assert_close(meas["a_sum"], full(1.0 + 2.0 + 3.0))
    torch.testing.assert_close(meas["b"], full(3.0))
    self.assertEqual(dev_a.num_on_substep, 3)
    self.assertEqual(dev_b.num_on_substep, 3)
    self.assertEqual(dev_a.num_reads, 1 + 3)
    self.assertEqual(dev_b.num_reads, 1 + 3)
    self.assertEqual(dev_a.num_on_step, 1)
    self.assertEqual(dev_b.num_on_step, 1)
    self.assertEqual(sum_processor.num_process, 1 + 3)

    # Partial reset: only the `env_mask` rows are refreshed and no per-substep
    # state is advanced for the other worlds.
    env_mask = torch.tensor([True, False][:nworld], device=xpu_device)
    mask = env_mask.view(-1, 1)
    dl.reset(env_mask)
    meas = dl.begin_stepping(env_mask)
    torch.testing.assert_close(
        meas["a_sum"], torch.where(mask, full(0.0), full(6.0))
    )
    torch.testing.assert_close(
        meas["b"], torch.where(mask, full(0.0), full(3.0))
    )
    self.assertEqual(dev_a.num_on_substep, 3)
    self.assertEqual(dev_a.num_on_step, 1)  # Not called after resets.
    # `on_begin_stepping` hooks get the reset worlds and reset-state readings.
    self.assertIs(dev_b.begin_stepping_masks[-1], env_mask)
    reset_reading, processor_mask = sum_processor.begin_stepping_calls[-1]
    self.assertIs(processor_mask, env_mask)
    torch.testing.assert_close(
        reset_reading, torch.where(mask, full(0.0), full(3.0))
    )

    # All-False mask (e.g. `AutoResetMode.NO_SYNC`) leaves every world as is.
    no_reset_mask = torch.zeros(nworld, dtype=torch.bool, device=xpu_device)
    dl.reset(no_reset_mask)
    meas = dl.begin_stepping(no_reset_mask)
    torch.testing.assert_close(
        meas["a_sum"], torch.where(mask, full(0.0), full(6.0))
    )

    # Non-reset worlds continue from their state (4 + 5 + 6); reset worlds
    # restart (1 + 2 + 3).
    meas = dl.step({})
    torch.testing.assert_close(
        meas["a_sum"], torch.where(mask, full(6.0), full(6.0 + 15.0))
    )
    torch.testing.assert_close(
        meas["b"], torch.where(mask, full(3.0), full(6.0))
    )

  def test_processor_chain_runs_every_substep(self, impl, xpu_device):
    nworld = _nworld(impl)
    trace: list[str] = []

    class ConstantDevice(_Device):

      def get_measurements(
          self, simulator: simulator_module.Simulator
      ) -> Mapping[str, torch.Tensor]:
        del simulator
        trace.append("x.get_measurements")
        return {"x": torch.zeros((nworld, 1), device=xpu_device)}

    class SubstepIncrementProcessor(_Processor):
      """Replaces `consumed` by `consumed + 1` under the key `produced`."""

      def on_substep(
          self,
          simulator: simulator_module.Simulator,
          substep_measurements: Mapping[str, torch.Tensor],
      ) -> None:
        del simulator
        keys = ",".join(sorted(substep_measurements))
        trace.append(f"{self.produced}.on_substep({keys})")

      def process_measurements(
          self,
          simulator: simulator_module.Simulator,
          measurements: MutableMapping[str, torch.Tensor],
      ) -> None:
        del simulator
        trace.append(f"{self.produced}.process_measurements")
        measurements[self.produced] = measurements.pop(self.consumed) + 1.0

    dl = _make_device_layer(
        impl,
        xpu_device,
        n_substeps=2,
        devices=[ConstantDevice("x")],
        processors=[
            SubstepIncrementProcessor("x", "y"),
            SubstepIncrementProcessor("y", "z"),
        ],
    )
    dl.begin_stepping()

    trace.clear()
    meas = dl.step({})
    self.assertEqual(
        trace,
        [
            "x.get_measurements",
            "y.on_substep(x)",
            "y.process_measurements",
            "z.on_substep(y)",
            "z.process_measurements",
        ]
        * 2,
    )
    torch.testing.assert_close(
        meas["z"], torch.full((nworld, 1), 2.0, device=xpu_device)
    )

  def test_raises_for_deprecated_and_new_methods(self, impl, xpu_device):
    class HalfMigratedDevice(CounterPassthroughDevice):
      """Adds a new hook to a device overriding `get_substep_measurements`."""

      def on_substep(self, simulator: simulator_module.Simulator) -> None:
        del simulator

    class HalfMigratedProcessor(CounterPassthroughSubstepMeasurementsProcessor):
      """Adds a new method to a processor overriding the deprecated one."""

      def process_measurements(
          self,
          simulator: simulator_module.Simulator,
          measurements: MutableMapping[str, torch.Tensor],
      ) -> None:
        del simulator, measurements

    with self.assertRaisesRegex(
        TypeError,
        "HalfMigratedDevice.*`get_substep_measurements\\(\\)`"
        ".*`on_substep\\(\\)`",
    ):
      _make_device_layer(impl, xpu_device, devices=[HalfMigratedDevice()])
    with self.assertRaisesRegex(
        TypeError,
        "HalfMigratedProcessor.*`process_substep_measurements\\(\\)`"
        ".*`process_measurements\\(\\)`",
    ):
      _make_device_layer(impl, xpu_device, processors=[HalfMigratedProcessor()])

  def test_warns_for_legacy_processors(self, impl, xpu_device):
    with self.assertWarnsRegex(
        DeprecationWarning,
        "CounterPassthroughSubstepMeasurementsProcessor.*"
        "process_substep_measurements",
    ):
      _make_device_layer(
          impl,
          xpu_device,
          processors=[CounterPassthroughSubstepMeasurementsProcessor()],
      )

  def test_partial_reset_takes_unbatched_measurements_as_is(
      self, impl, xpu_device
  ):
    nworld = _nworld(impl)

    class ValueDevice(_Device):
      """Returns `value` batched, as a 0-dim tensor, and not batched."""

      def __init__(self):
        super().__init__("value", keys=("batched", "scalar", "table"))
        self.value = 1.0

      def get_measurements(
          self, simulator: simulator_module.Simulator
      ) -> Mapping[str, torch.Tensor]:
        del simulator
        return {
            "batched": torch.full((nworld, 1), self.value, device=xpu_device),
            "scalar": torch.tensor(self.value, device=xpu_device),
            "table": torch.full((nworld + 1,), self.value, device=xpu_device),
        }

    device = ValueDevice()
    dl = _make_device_layer(impl, xpu_device, devices=[device])
    dl.begin_stepping()

    # No world is reset: the batched rows are kept, while measurements that
    # are not batched over worlds are taken as is.
    device.value = 2.0
    env_mask = torch.zeros(nworld, dtype=torch.bool, device=xpu_device)
    dl.reset(env_mask)
    meas = dl.begin_stepping(env_mask)
    torch.testing.assert_close(
        meas["batched"], torch.full((nworld, 1), 1.0, device=xpu_device)
    )
    torch.testing.assert_close(
        meas["scalar"], torch.tensor(2.0, device=xpu_device)
    )
    torch.testing.assert_close(
        meas["table"], torch.full((nworld + 1,), 2.0, device=xpu_device)
    )

  def test_no_op_resets_do_not_advance_delay_buffers(self, impl, xpu_device):
    nworld = _nworld(impl)

    class SubstepCountDevice(_Device):
      """Measures the number of substeps since the start."""

      def __init__(self):
        super().__init__("x")
        self.count = torch.zeros((nworld, 1), device=xpu_device)

      def on_substep(self, simulator: simulator_module.Simulator) -> None:
        del simulator
        self.count = self.count + 1.0

      def get_measurements(
          self, simulator: simulator_module.Simulator
      ) -> Mapping[str, torch.Tensor]:
        del simulator
        return {"x": self.count}

    dl = _make_device_layer(
        impl,
        xpu_device,
        devices=[SubstepCountDevice()],
        processors=[
            processors_module.StochasticSubstepMeasurements(
                keys="x",
                delay=(0.004, 0.004),  # 2 substeps.
                nworld=nworld,
                xpu_device=xpu_device,
                simulation_timestep=0.002,
            )
        ],
    )
    dl.begin_stepping()
    no_reset = torch.zeros(nworld, dtype=torch.bool, device=xpu_device)
    # The reset reading (0) is held until the delay has passed.
    for expected in (0.0, 0.0, 1.0, 2.0):
      meas = dl.step({})
      torch.testing.assert_close(
          meas["x"], torch.full((nworld, 1), expected, device=xpu_device)
      )
      # As under `AutoResetMode.NO_SYNC` when no world terminates.
      dl.reset(no_reset)
      dl.begin_stepping(no_reset)

  def test_update_measurements_reads_and_processes_without_hooks(
      self, impl, xpu_device
  ):
    nworld = _nworld(impl)
    trace: list[str] = []

    class ValueDevice(_Device):

      def __init__(self):
        super().__init__("x")
        self.value = 0.0

      def on_substep(self, simulator: simulator_module.Simulator) -> None:
        del simulator
        trace.append("x.on_substep")

      def get_measurements(
          self, simulator: simulator_module.Simulator
      ) -> Mapping[str, torch.Tensor]:
        del simulator
        return {"x": torch.full((nworld, 1), self.value, device=xpu_device)}

    class IncrementProcessor(_Processor):

      def on_substep(
          self,
          simulator: simulator_module.Simulator,
          substep_measurements: Mapping[str, torch.Tensor],
      ) -> None:
        del simulator, substep_measurements
        trace.append("y.on_substep")

      def process_measurements(
          self,
          simulator: simulator_module.Simulator,
          measurements: MutableMapping[str, torch.Tensor],
      ) -> None:
        del simulator
        measurements["y"] = measurements.pop("x") + 1.0

    device = ValueDevice()
    dl = _make_device_layer(
        impl,
        xpu_device,
        devices=[device],
        processors=[IncrementProcessor("x", "y")],
    )
    dl.begin_stepping()

    # E.g. after a state change without stepping.
    device.value = 5.0
    trace.clear()
    dl.update_measurements()
    self.assertEmpty(trace)
    torch.testing.assert_close(
        dl.get_measurements()["y"],
        torch.full((nworld, 1), 6.0, device=xpu_device),
    )

  def test_output_buffers_isolate_sim_views_and_preserve_data_ptr(
      self, impl, xpu_device
  ):
    nworld = _nworld(impl)

    class SimSliceDevice(_Device):

      def get_measurements(
          self, simulator: simulator_module.Simulator
      ) -> Mapping[str, torch.Tensor]:
        # Returns a strided (non-contiguous) view directly into `sim.data.qpos`.
        return {"qpos_even": simulator.data.qpos[:, ::2]}

      def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
        return {"qpos_even": specs.Array((5,), np.float32)}

    dl = _make_device_layer(impl, xpu_device, devices=[SimSliceDevice("sim")])
    dl.sim.data.qpos[:] = 3.0
    meas = dl.begin_stepping()
    buf = meas["qpos_even"]
    initial_ptr = buf.data_ptr()

    self.assertTrue(buf.is_contiguous())
    self.assertNotEqual(
        buf.untyped_storage().data_ptr(),
        dl.sim.data.qpos.untyped_storage().data_ptr(),
    )

    # Mutating `sim.data.qpos` directly + calling `forward()` must not alter
    # the caller's held output buffer prior to `begin_stepping`.
    dl.sim.data.qpos[:] = 99.0
    dl.forward()
    torch.testing.assert_close(
        buf, torch.full_like(buf, 3.0), rtol=0.0, atol=0.0
    )

    mask = torch.zeros(nworld, dtype=torch.bool, device=xpu_device)
    mask[0] = True
    dl.reset(mask)
    dl.forward()
    meas = dl.begin_stepping(mask)
    self.assertEqual(meas["qpos_even"].data_ptr(), initial_ptr)
    if nworld > 1:
      torch.testing.assert_close(
          meas["qpos_even"][1:],
          torch.full_like(meas["qpos_even"][1:], 3.0),
      )

    # `step()` and `update_measurements()` write into the same buffers.
    self.assertEqual(dl.step({})["qpos_even"].data_ptr(), initial_ptr)
    dl.update_measurements()
    self.assertEqual(dl.get_measurements()["qpos_even"].data_ptr(), initial_ptr)

  def test_output_buffers_raise_for_undeclared_or_missing_keys(
      self, impl, xpu_device
  ):
    nworld = _nworld(impl)

    class KeysDevice(_Device):

      def __init__(self):
        super().__init__("x", keys=("x", "y"))
        self.keys: tuple[str, ...] = ("x", "y")

      def get_measurements(
          self, simulator: simulator_module.Simulator
      ) -> Mapping[str, torch.Tensor]:
        del simulator
        return {
            k: torch.zeros((nworld, 1), device=xpu_device) for k in self.keys
        }

    # A key outside the specs raises on its first (unmasked) commit.
    dev = KeysDevice()
    dev.keys = ("x", "y", "undeclared")
    dl = _make_device_layer(impl, xpu_device, devices=[dev])
    with self.assertRaisesRegex(KeyError, "'undeclared' is not in the spec"):
      dl.begin_stepping()

    # Omitting a previously emitted key raises instead of serving stale values.
    dev = KeysDevice()
    dl = _make_device_layer(impl, xpu_device, devices=[dev])
    dl.begin_stepping()
    dev.keys = ("x",)
    with self.assertRaisesRegex(KeyError, "Missing measurement keys"):
      dl.step({})

  def test_on_step_zero_order_hold_on_final_substep(self, impl, xpu_device):
    nworld = _nworld(impl)
    seen_by_substep_proc: list[float] = []

    class StepHeldDevice(_Device):

      def __init__(self):
        super().__init__("x")
        self.step_val = 0.0

      def on_step(self, simulator: simulator_module.Simulator) -> None:
        self.step_val += 1.0

      def get_measurements(
          self, simulator: simulator_module.Simulator
      ) -> Mapping[str, torch.Tensor]:
        return {"x": torch.full((nworld, 1), self.step_val, device=xpu_device)}

    class RecordingSubstepProcessor(_Processor):

      def on_substep(
          self,
          simulator: simulator_module.Simulator,
          substep_measurements: Mapping[str, torch.Tensor],
      ) -> None:
        seen_by_substep_proc.append(float(substep_measurements["x"][0, 0]))

    dl = _make_device_layer(
        impl,
        xpu_device,
        n_substeps=3,
        devices=[StepHeldDevice()],
        processors=[RecordingSubstepProcessor("x", "x")],
    )
    dl.begin_stepping()
    meas = dl.step({})
    # Substeps 0 and 1 hold `0.0` (zero-order hold); final substep 2 runs
    # `on_step` before `get_measurements` and `proc.on_substep`, yielding `1.0`.
    self.assertEqual(seen_by_substep_proc, [0.0, 0.0, 1.0])
    torch.testing.assert_close(
        meas["x"], torch.full((nworld, 1), 1.0, device=xpu_device)
    )


if __name__ == "__main__":
  absltest.main()
