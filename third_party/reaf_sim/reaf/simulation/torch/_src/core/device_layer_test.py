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
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
import numpy as np
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


if __name__ == "__main__":
  absltest.main()
