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

"""macOS user-space `gs_usb` CAN transport shim for I2RT (`yam` extra).

`i2rt.robots.get_robot.get_yam_robot` hardcodes `interface="socketcan"` when
opening `can.Bus`. Linux provides `socketcan` in the kernel, whereas macOS
accesses candleLight / `gs_usb` USB-CAN adapters in user space via `libusb`.

`python-can`'s built-in `GsUsbBus` (`python-can==4.5.0`, `gs-usb==0.3.1`) also
requires three macOS workarounds:
1. `GsUsb.start()` calls `pyusb` `reset()`, triggering `USBDeviceReEnumerate`
   on macOS and invalidating the open USB handle (`GsUsbBus.shutdown()` also
   calls `start()`). We send `GS_CAN_MODE_START` (`ctrl_transfer` `0x41, 2`)
   directly without resetting.
2. `gs_usb` loops transmitted frames back on the bulk IN endpoint with
   `is_rx=False`; Linux `socketcan` suppresses local echoes by default, and
   returning them breaks `i2rt`'s request/reply motor loop.
3. `GsUsb.read()` passes a 24-byte buffer (`LIBUSB_ERROR_OVERFLOW` when a TX
   echo and RX reply coalesce into one 64-byte USB packet) and 1 ms synchronous
   reads trigger `AbortPipe()` frame drops on macOS. A background thread reads
   64-byte bulk IN packets into a queue, and `open_bus` drains extra `0xFB`
   replies left unconsumed by `i2rt`'s `clean_error`.
"""

import contextlib
import queue
import sys
import threading
import time
from typing import Any

from mujoco_playground.robots.yam.real import constant

_USB_REQ_HOST_TO_DEV_VENDOR_INTERFACE = 0x41
_GS_USB_BREQ_MODE = 2
_GS_USB_EP_IN = 0x81
_USB_PACKET_BYTES = 64
_USB_READ_TIMEOUT_MS = 25
_USB_TIMEOUT_MS = 50
_MAX_DRAIN_FRAMES = 64


def parse_can_index(channel: str) -> int:
  """Maps a SocketCAN-style channel name (`'can0'`) to a 0-based USB index."""
  if (
      channel.startswith("can")
      and channel[3:].isascii()
      and channel[3:].isdigit()
  ):
    return int(channel[3:])
  raise ValueError(f"Expected CAN channel like 'can0', got {channel!r}.")


def _scan_gs_usb() -> list[Any]:
  """Returns `gs_usb` adapters sorted deterministically by `(bus, address)`."""
  # pylint: disable=g-import-not-at-top
  from gs_usb import gs_usb as gs_usb_mod  # pyrefly: ignore[missing-import]
  from usb.backend import libusb1  # pyrefly: ignore[missing-import]

  # pylint: enable=g-import-not-at-top

  if libusb1.get_backend() is None:
    raise RuntimeError(
        "libusb not found on macOS. Install it with:\n  brew install libusb"
    )
  return sorted(
      gs_usb_mod.GsUsb.scan(),
      key=lambda dev: (dev.gs_usb.bus or 0, dev.gs_usb.address or 0),
  )


def check_can_up(channel: str = constant.CAN_CHANNEL) -> None:
  """Raises with remediation instructions if `libusb` or `gs_usb` is missing."""
  idx = parse_can_index(channel)
  devs = _scan_gs_usb()
  if idx >= len(devs):
    raise RuntimeError(
        f"No USB-CAN adapter for '{channel}' (found {len(devs)} gs_usb"
        " device(s)). Is the adapter plugged in?"
    )


def open_bus(
    channel: str = constant.CAN_CHANNEL,
    bitrate: int = constant.CAN_BITRATE,
    **kwargs: Any,
) -> Any:
  """Opens a `gs_usb` USB-CAN adapter on macOS without USB `reset()` or echoes."""
  # pylint: disable=g-import-not-at-top
  import can  # pyrefly: ignore[missing-import]
  from can.interfaces import gs_usb as gs_usb_bus  # pyrefly: ignore[missing-import]
  from gs_usb import gs_usb as gs_usb_mod  # pyrefly: ignore[missing-import]
  from gs_usb import gs_usb_frame  # pyrefly: ignore[missing-import]
  from usb import core as usb_core  # pyrefly: ignore[missing-import]
  from usb import util as usb_util  # pyrefly: ignore[missing-import]

  # pylint: enable=g-import-not-at-top

  class _MacGsUsbBus(gs_usb_bus.GsUsbBus):
    """`GsUsbBus` subclass avoiding `USBDeviceReEnumerate` and TX echoes."""

    def __init__(self) -> None:
      self._rx_queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
      self._closing = False
      self._rx_thread: threading.Thread | None = None
      idx = parse_can_index(channel)
      try:
        devs = _scan_gs_usb()
      except (RuntimeError, usb_core.USBError, usb_core.NoBackendError) as exc:
        raise can.CanInitializationError(str(exc)) from exc
      if idx >= len(devs):
        raise can.CanInitializationError(f"No gs_usb device for {channel!r}.")
      self.gs_usb = devs[idx]
      self.channel_info = channel
      self.gs_usb.gs_usb.default_timeout = _USB_TIMEOUT_MS
      try:
        usb_util.claim_interface(self.gs_usb.gs_usb, 0)
        self.gs_usb.stop()
        with contextlib.suppress(usb_core.USBError):
          for _ in range(_MAX_DRAIN_FRAMES):
            self.gs_usb.gs_usb.read(_GS_USB_EP_IN, _USB_PACKET_BYTES, 1)
        flags = (
            gs_usb_mod.GS_CAN_MODE_HW_TIMESTAMP
            & self.gs_usb.device_capability.feature
        )
        self.gs_usb.device_flags = flags
        timing = can.BitTiming.from_sample_point(
            f_clock=self.gs_usb.device_capability.fclk_can,
            bitrate=bitrate,
            sample_point=87.5,  # 87.5% of bit period (GsUsbBus default).
        )
        self.gs_usb.set_timing(
            1, timing.tseg1 - 1, timing.tseg2, timing.sjw, timing.brp
        )
        mode = gs_usb_mod.DeviceMode(gs_usb_mod.GS_CAN_MODE_START, flags)
        self.gs_usb.gs_usb.ctrl_transfer(
            _USB_REQ_HOST_TO_DEV_VENDOR_INTERFACE,
            _GS_USB_BREQ_MODE,
            0,
            0,
            mode.pack(),
        )
        can.BusABC.__init__(self, channel=channel, **kwargs)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
      except usb_core.USBError as exc:
        self._close_usb()
        raise can.CanInitializationError(f"gs_usb init failed: {exc}") from exc
      except Exception:
        self._close_usb()
        raise

    def _close_usb(self) -> None:
      with contextlib.suppress(usb_core.USBError):
        self.gs_usb.stop()
      with contextlib.suppress(usb_core.USBError):
        usb_util.dispose_resources(self.gs_usb.gs_usb)

    def _rx_loop(self) -> None:
      hw_ts = bool(
          self.gs_usb.device_flags & gs_usb_mod.GS_CAN_MODE_HW_TIMESTAMP
      )
      size = gs_usb_frame.GsUsbFrame().__sizeof__(hw_ts)
      while not self._closing:
        try:
          raw = bytes(
              self.gs_usb.gs_usb.read(
                  _GS_USB_EP_IN, _USB_PACKET_BYTES, _USB_READ_TIMEOUT_MS
              )
          )
        except usb_core.USBTimeoutError:
          continue
        except usb_core.USBError:
          if not self._closing:
            time.sleep(0.005)
          continue
        for off in range(0, len(raw) - size + 1, size):
          frame = gs_usb_frame.GsUsbFrame()
          gs_usb_frame.GsUsbFrame.unpack_into(
              frame, raw[off : off + size], hw_ts
          )
          if (
              frame.echo_id == gs_usb_frame.GS_USB_NONE_ECHO_ID
              and not frame.is_error_frame
          ):
            self._rx_queue.put(
                can.Message(
                    timestamp=frame.timestamp,
                    arbitration_id=frame.arbitration_id,
                    is_extended_id=frame.is_extended_id,
                    is_remote_frame=frame.is_remote_frame,
                    channel=self.channel_info,
                    dlc=frame.can_dlc,
                    data=bytearray(frame.data)[: frame.can_dlc],
                    is_rx=True,
                )
            )

    def _recv_internal(self, timeout: float | None) -> tuple[Any, bool]:
      if self._closing:
        return None, False
      try:
        wait = None if timeout is None else max(0.0, timeout)
        return self._rx_queue.get(timeout=wait), False
      except queue.Empty:
        return None, False

    def shutdown(self) -> None:
      if self._is_shutdown:
        return
      self._closing = True
      if self._rx_thread is not None:
        self._rx_thread.join(timeout=0.1)
      self._rx_queue.put(None)
      try:
        can.BusABC.shutdown(self)
      finally:
        self._close_usb()

  dm_driver = sys.modules.get("i2rt.motor_drivers.dm_driver")
  if dm_driver is not None:
    cls = dm_driver.DMSingleMotorCanInterface
    if not getattr(cls.clean_error, "_drains_bus", False):
      orig = cls.clean_error

      def _clean_error(self: Any, *args: Any, **kwargs: Any) -> Any:
        res = orig(self, *args, **kwargs)
        self._drain_bus(timeout_s=0.05, idle_count=5)  # pylint: disable=protected-access
        return res

      setattr(_clean_error, "_drains_bus", True)
      cls.clean_error = _clean_error
  return _MacGsUsbBus()


def register_socketcan_backend() -> None:
  """Redirects `python-can`'s `'socketcan'` backend to `open_bus` on macOS."""
  # pylint: disable=g-import-not-at-top
  from can import interfaces as can_interfaces  # pyrefly: ignore[missing-import]

  # pylint: enable=g-import-not-at-top

  can_interfaces.BACKENDS["socketcan"] = (__name__, "open_bus")
