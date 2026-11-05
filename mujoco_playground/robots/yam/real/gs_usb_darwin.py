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
3. Sub-millisecond read timeouts round to `0` ms in `GsUsbBus._recv_internal`,
   so we clamp non-`None` timeouts to at least `1` ms (`0.001` s).
"""

import contextlib
import time
from typing import Any

from mujoco_playground.robots.yam.real import constant

_USB_REQ_HOST_TO_DEV_VENDOR_INTERFACE = 0x41
_GS_USB_BREQ_MODE = 2
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
        flags = (
            gs_usb_mod.GS_CAN_MODE_HW_TIMESTAMP
            & self.gs_usb.device_capability.feature
        )
        self.gs_usb.device_flags = flags
        frame = gs_usb_frame.GsUsbFrame()
        for _ in range(_MAX_DRAIN_FRAMES):
          if not self.gs_usb.read(frame, timeout_ms=1):
            break
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

    def _recv_internal(self, timeout: float | None) -> tuple[Any, bool]:
      deadline = (
          None if timeout is None else time.monotonic() + max(0.001, timeout)
      )
      while True:
        wait = (
            None
            if deadline is None
            else max(0.001, deadline - time.monotonic())
        )
        msg, filtered = super()._recv_internal(wait)
        if msg is None or (msg.is_rx and not msg.is_error_frame):
          return msg, filtered
        if deadline is not None and time.monotonic() >= deadline:
          return None, False

    def shutdown(self) -> None:
      if self._is_shutdown:
        return
      try:
        can.BusABC.shutdown(self)
      finally:
        self._close_usb()

  return _MacGsUsbBus()


def register_socketcan_backend() -> None:
  """Redirects `python-can`'s `'socketcan'` backend to `open_bus` on macOS."""
  # pylint: disable=g-import-not-at-top
  from can import interfaces as can_interfaces  # pyrefly: ignore[missing-import]

  # pylint: enable=g-import-not-at-top

  can_interfaces.BACKENDS["socketcan"] = (__name__, "open_bus")
