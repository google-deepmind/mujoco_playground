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

"""Tests for the macOS `gs_usb` CAN transport shim."""

import struct
import sys
import time
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.real import gs_usb_darwin

_FRAME_STRUCT = struct.Struct("<IIBB2x8sI")
_NONE_ECHO_ID = 0xFFFFFFFF


def _pack_frame(
    can_id: int = 0x11,
    data: bytes = b"\x01",
    echo_id: int = _NONE_ECHO_ID,
) -> bytes:
  return _FRAME_STRUCT.pack(
      echo_id, can_id, len(data), 0, data.ljust(8, b"\x00"), 0
  )


def _fake_darwin_usb_can_stack(
    rx_packets: list[bytes],
) -> tuple[dict[str, Any], Any, Any, Any, Any]:
  """Builds stub `can`, `gs_usb`, and `usb` modules matching `python-can==4.5.0`."""

  class FakeCanInitializationError(Exception):
    pass

  class FakeUSBError(OSError):
    pass

  class FakeUSBTimeoutError(FakeUSBError):
    pass

  class FakeNoBackendError(ValueError):
    pass

  class FakeBusABC:
    _is_shutdown = True

    def __init__(self, channel: Any = None, **kwargs: Any) -> None:
      self.init_channel = channel
      self.init_kwargs = kwargs
      self.stopped_periodic_tasks = 0
      self._is_shutdown = False

    def stop_all_periodic_tasks(self) -> None:
      self.stopped_periodic_tasks += 1

    def shutdown(self) -> None:
      if self._is_shutdown:
        return
      self._is_shutdown = True
      self.stop_all_periodic_tasks()

  class FakeGsUsbBus(FakeBusABC):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
      raise AssertionError(
          "GsUsbBus.__init__ calls GsUsb.start() which resets USB on macOS."
      )

    def shutdown(self) -> None:
      raise AssertionError(
          "GsUsbBus.shutdown calls GsUsb.start() which resets USB on macOS."
      )

  class FakeGsUsbFrame:
    GS_USB_NONE_ECHO_ID = _NONE_ECHO_ID

    def __sizeof__(self, hw_timestamps: bool = True) -> int:
      return _FRAME_STRUCT.size if hw_timestamps else 20

    @staticmethod
    def unpack_into(frame: Any, data: bytes, hw_timestamps: bool) -> None:
      del hw_timestamps
      echo_id, can_id, dlc, _, raw, ts = _FRAME_STRUCT.unpack(data)
      frame.echo_id = echo_id
      frame.arbitration_id = can_id & 0x1FFFFFFF
      frame.is_extended_id = bool(can_id & 0x80000000)
      frame.is_remote_frame = bool(can_id & 0x40000000)
      frame.is_error_frame = bool(can_id & 0x20000000)
      frame.can_dlc = dlc
      frame.data = raw
      frame.timestamp = ts / 1e6

  fake_can_interfaces: Any = types.ModuleType("can.interfaces")
  fake_can_interfaces.BACKENDS = {
      "socketcan": ("can.interfaces.socketcan", "SocketcanBus")
  }
  fake_can_gs_usb: Any = types.ModuleType("can.interfaces.gs_usb")
  fake_can_gs_usb.GsUsbBus = FakeGsUsbBus
  fake_can_interfaces.gs_usb = fake_can_gs_usb

  fake_can: Any = types.ModuleType("can")
  fake_can.BusABC = FakeBusABC
  fake_can.CanInitializationError = FakeCanInitializationError
  fake_can.Message = types.SimpleNamespace
  fake_can.BitTiming = types.SimpleNamespace(
      from_sample_point=lambda **_: types.SimpleNamespace(
          tseg1=6, tseg2=1, sjw=1, brp=6
      )
  )
  fake_can.interfaces = fake_can_interfaces

  drained = False

  def _usb_read(ep: int, size: int, timeout: int) -> bytes:
    nonlocal drained
    assert ep == 0x81 and size == 64
    if timeout == 1:
      if not drained:
        drained = True
        return b"\x00" * 24
      raise FakeUSBTimeoutError("drained")
    if rx_packets:
      return rx_packets.pop(0)
    time.sleep(0.002)
    raise FakeUSBTimeoutError("idle")

  usb_dev = types.SimpleNamespace(
      bus=1,
      address=1,
      default_timeout=1000,
      ctrl_transfer=mock.MagicMock(),
      read=mock.MagicMock(side_effect=_usb_read),
  )
  fake_adapter = types.SimpleNamespace(
      gs_usb=usb_dev,
      device_capability=types.SimpleNamespace(feature=16, fclk_can=48_000_000),
      device_flags=0,
      set_timing=mock.MagicMock(),
      start=mock.MagicMock(
          side_effect=AssertionError("GsUsb.start() must not be called")
      ),
      stop=mock.MagicMock(),
  )
  fake_gs_usb_mod: Any = types.ModuleType("gs_usb.gs_usb")
  fake_gs_usb_mod.GsUsb = types.SimpleNamespace(scan=lambda: [fake_adapter])
  fake_gs_usb_mod.GS_CAN_MODE_HW_TIMESTAMP = 16
  fake_gs_usb_mod.GS_CAN_MODE_START = 1
  fake_gs_usb_mod.DeviceMode = lambda mode, flags: types.SimpleNamespace(
      pack=lambda: bytes([mode, flags])
  )
  fake_gs_usb_frame: Any = types.ModuleType("gs_usb.gs_usb_frame")
  fake_gs_usb_frame.GsUsbFrame = FakeGsUsbFrame
  fake_gs_usb_frame.GS_USB_NONE_ECHO_ID = _NONE_ECHO_ID
  fake_gs_usb_pkg: Any = types.ModuleType("gs_usb")
  fake_gs_usb_pkg.gs_usb = fake_gs_usb_mod
  fake_gs_usb_pkg.gs_usb_frame = fake_gs_usb_frame

  fake_libusb1: Any = types.ModuleType("usb.backend.libusb1")
  fake_libusb1.get_backend = mock.MagicMock(return_value=object())
  fake_usb_backend: Any = types.ModuleType("usb.backend")
  fake_usb_backend.libusb1 = fake_libusb1
  fake_usb_core: Any = types.ModuleType("usb.core")
  fake_usb_core.USBError = FakeUSBError
  fake_usb_core.USBTimeoutError = FakeUSBTimeoutError
  fake_usb_core.NoBackendError = FakeNoBackendError
  fake_usb_util: Any = types.ModuleType("usb.util")
  fake_usb_util.claim_interface = mock.MagicMock()
  fake_usb_util.dispose_resources = mock.MagicMock()
  fake_usb_pkg: Any = types.ModuleType("usb")
  fake_usb_pkg.backend = fake_usb_backend
  fake_usb_pkg.core = fake_usb_core
  fake_usb_pkg.util = fake_usb_util

  modules = {
      "can": fake_can,
      "can.interfaces": fake_can_interfaces,
      "can.interfaces.gs_usb": fake_can_gs_usb,
      "gs_usb": fake_gs_usb_pkg,
      "gs_usb.gs_usb": fake_gs_usb_mod,
      "gs_usb.gs_usb_frame": fake_gs_usb_frame,
      "usb": fake_usb_pkg,
      "usb.backend": fake_usb_backend,
      "usb.backend.libusb1": fake_libusb1,
      "usb.core": fake_usb_core,
      "usb.util": fake_usb_util,
  }
  return modules, fake_can_interfaces, fake_adapter, usb_dev, fake_usb_util


class GsUsbDarwinTest(absltest.TestCase):

  def test_parse_can_index(self) -> None:
    self.assertEqual(gs_usb_darwin.parse_can_index("can0"), 0)
    self.assertEqual(gs_usb_darwin.parse_can_index("can2"), 2)
    for invalid in ("vcan0", "can\u0660", "can", "can-1"):
      with self.assertRaisesRegex(
          ValueError, "Expected CAN channel like 'can0'"
      ):
        gs_usb_darwin.parse_can_index(invalid)

  def test_check_can_up_and_adapter_sorting(self) -> None:
    modules, _, low_adapter, _, _ = _fake_darwin_usb_can_stack([])
    high_adapter = types.SimpleNamespace(
        gs_usb=types.SimpleNamespace(
            bus=2,
            address=5,
            default_timeout=1000,
            ctrl_transfer=mock.MagicMock(),
            read=low_adapter.gs_usb.read,
        ),
        device_capability=low_adapter.device_capability,
        device_flags=0,
        set_timing=mock.MagicMock(),
        stop=mock.MagicMock(),
    )

    with mock.patch.dict(sys.modules, modules):
      modules["usb.backend.libusb1"].get_backend.return_value = None
      with self.assertRaisesRegex(RuntimeError, "brew install libusb"):
        gs_usb_darwin.check_can_up("can0")
      with self.assertRaisesRegex(
          modules["can"].CanInitializationError, "brew install libusb"
      ):
        gs_usb_darwin.open_bus("can0")

      modules["usb.backend.libusb1"].get_backend.return_value = object()
      modules["gs_usb.gs_usb"].GsUsb.scan = lambda: [high_adapter, low_adapter]
      gs_usb_darwin.check_can_up("can0")
      gs_usb_darwin.check_can_up("can1")
      with self.assertRaisesRegex(
          RuntimeError, "No USB-CAN adapter for 'can2'"
      ):
        gs_usb_darwin.check_can_up("can2")

      bus = gs_usb_darwin.open_bus("can0", bitrate=constant.CAN_BITRATE)
      self.assertIs(bus.gs_usb, low_adapter)
      bus.shutdown()

  def test_open_bus_coalesced_rx_filters_echoes_and_patches_clean_error(
      self,
  ) -> None:
    coalesced = _pack_frame(echo_id=0) + _pack_frame(can_id=0x20000001)
    valid = _pack_frame(can_id=0x14, data=b"\xab\xcd")
    modules, fake_can_interfaces, fake_adapter, usb_dev, fake_usb_util = (
        _fake_darwin_usb_can_stack([coalesced, valid])
    )
    events: list[str] = []

    class FakeMotorIface:

      def clean_error(self, motor_id: int) -> None:
        events.append(f"clean:{motor_id}")

      def _drain_bus(self, timeout_s: float, idle_count: int) -> None:
        events.append(f"drain:{timeout_s}:{idle_count}")

    modules["i2rt.motor_drivers.dm_driver"] = types.SimpleNamespace(
        DMSingleMotorCanInterface=FakeMotorIface
    )

    with mock.patch.dict(sys.modules, modules):
      gs_usb_darwin.register_socketcan_backend()
      bus = gs_usb_darwin.open_bus("can0", bitrate=constant.CAN_BITRATE)
      msg, _ = bus._recv_internal(0.1)
      none_timeout, _ = bus._recv_internal(0.0)
      bus.shutdown()
      bus.shutdown()
      none_closed, _ = bus._recv_internal(None)
      FakeMotorIface().clean_error(motor_id=4)

    self.assertEqual(
        fake_can_interfaces.BACKENDS["socketcan"],
        (gs_usb_darwin.__name__, "open_bus"),
    )
    self.assertEqual(bus.channel_info, "can0")
    self.assertEqual(usb_dev.default_timeout, 50)
    self.assertEqual(fake_adapter.device_flags, 16)
    fake_adapter.start.assert_not_called()
    fake_usb_util.claim_interface.assert_called_once_with(usb_dev, 0)
    fake_adapter.set_timing.assert_called_once_with(1, 5, 1, 1, 6)
    usb_dev.ctrl_transfer.assert_called_once_with(0x41, 2, 0, 0, b"\x01\x10")
    self.assertEqual(bytes(msg.data), b"\xab\xcd")
    self.assertEqual(msg.arbitration_id, 0x14)
    self.assertIsNone(none_timeout)
    self.assertIsNone(none_closed)
    self.assertEqual(events, ["clean:4", "drain:0.05:5"])
    self.assertEqual(bus.stopped_periodic_tasks, 1)
    self.assertEqual(fake_adapter.stop.call_count, 2)
    fake_usb_util.dispose_resources.assert_called_once_with(usb_dev)

  def test_init_failure_translates_usb_error_and_disposes_resources(
      self,
  ) -> None:
    modules, _, fake_adapter, usb_dev, fake_usb_util = (
        _fake_darwin_usb_can_stack([])
    )
    fake_adapter.set_timing.side_effect = modules["usb.core"].USBError("busy")

    with mock.patch.dict(sys.modules, modules):
      with self.assertRaises(modules["can"].CanInitializationError):
        gs_usb_darwin.open_bus("can0", bitrate=constant.CAN_BITRATE)

    fake_usb_util.claim_interface.assert_called_once_with(usb_dev, 0)
    fake_usb_util.dispose_resources.assert_called_once_with(usb_dev)


if __name__ == "__main__":
  absltest.main()
