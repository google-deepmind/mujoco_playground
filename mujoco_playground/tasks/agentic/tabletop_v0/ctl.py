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

"""Tiny HTTP client and Code-as-Policies SDK for `run.py --agent=bridge`.

ctl obs                                           # print the latest observation
ctl plan 'robot.grasp(0.22, 0.0, 0.015)'          # compile & run a batch plan
ctl code 'obs = robot.move_until_contact(dz=-0.05)' # run closed-loop Python
ctl goto --x=0.22 --y=0.0 --z=0.12 --arc_z=0.16   # move with optional arc_z
ctl move_until_contact --dz=-0.05                 # guarded relative move
ctl place_until_contact --x=0.22 --y=0.0          # guarded contact place
ctl gripper 0.4                                   # jaws to 40% open
ctl close_gripper --verify_grasp                  # fail if nothing held
ctl grasp --x=0.22 --y=0.0 --opening=0.4          # open 40% before descending
ctl done                                          # finish the current episode
ctl reset                                         # start a new episode
"""

import json
import os
import sys
from typing import Any
import urllib.request

from absl import app
from absl import flags

_URL = flags.DEFINE_string("url", "", "Bridge URL (auto-discovered if empty).")
_BRIDGE_DIR = flags.DEFINE_string(
    "bridge_dir", "", "Bridge directory containing bridge_info.json."
)
_X = flags.DEFINE_float("x", None, "Target world X (m).")
_Y = flags.DEFINE_float("y", None, "Target world Y (m).")
_Z = flags.DEFINE_float("z", None, "Target world Z (m).")
# Not `--u`/`--v`: absl.logging already owns `-v` (verbosity).
_CAM_U = flags.DEFINE_float(
    "cam_u", None, "Target wrist-cam pixel/normalized U."
)
_CAM_V = flags.DEFINE_float(
    "cam_v", None, "Target wrist-cam pixel/normalized V."
)
_DX = flags.DEFINE_float("dx", 0.0, "Relative delta X (m).")
_DY = flags.DEFINE_float("dy", 0.0, "Relative delta Y (m).")
_DZ = flags.DEFINE_float("dz", None, "Relative delta Z (m).")
_FORWARD = flags.DEFINE_float("forward", 0.0, "Coaxial tool-frame delta (m).")
_ARC_Z = flags.DEFINE_float(
    "arc_z", None, "Optional parabolic arc apex Z (m) during goto/transit."
)
_SPEED = flags.DEFINE_float("speed", None, "Optional linear speed (m/s).")
_MAX_LAG = flags.DEFINE_float(
    "max_lag", 0.012, "Axial tracking lag threshold (m) for contact detection."
)
_RELIEF = flags.DEFINE_float(
    "relief", 0.002, "Back-off distance (m) along approach axis upon contact."
)
_YAW_DEG = flags.DEFINE_float("yaw_deg", None, "Object long-axis yaw (deg).")
_PITCH_DEG = flags.DEFINE_float("pitch_deg", None, "Approach pitch (deg).")
_ROLL_DEG = flags.DEFINE_float("roll_deg", None, "Jaw roll (deg).")
_AZIMUTH_DEG = flags.DEFINE_float(
    "azimuth_deg", None, "Horizontal approach azimuth (deg)."
)
_CLEARANCE = flags.DEFINE_float(
    "clearance", None, "Optional release height clearance (m) on place."
)
_OPENING = flags.DEFINE_float(
    "opening", None, "Optional jaw opening in [0, 1] for a grasp or place."
)
_VERIFY_GRASP = flags.DEFINE_bool(
    "verify_grasp",
    False,
    "On grasp/gripper: fail unless an object holds the jaws open.",
)
_THRESHOLD = flags.DEFINE_float(
    "threshold",
    None,
    "Optional measured-minus-commanded opening of a verified grasp.",
)


def _resolve_url(url: str = "", bridge_dir: str = "") -> str:
  """Resolves the bridge HTTP URL from flags, bridge_info.json, or default."""
  if url:
    return url.rstrip("/")
  bdir = bridge_dir or os.environ.get(
      "SO101_BRIDGE_DIR",
      os.path.join(os.environ.get("TEST_TMPDIR", "/tmp"), "so101_agent_bridge"),
  )
  info_path = os.path.join(bdir, "bridge_info.json")
  if os.path.exists(info_path):
    try:
      with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
      if "http_url" in info:
        return str(info["http_url"]).rstrip("/")
    except (OSError, json.JSONDecodeError):
      pass
  return "http://127.0.0.1:8081"


class Robot:
  """Stateful HTTP client for closed-loop Code-as-Policies over the bridge."""

  def __init__(self, url: str = "", bridge_dir: str = ""):
    self._url = _resolve_url(url, bridge_dir)

  def _req(self, path: str, body: bytes | None = None) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{self._url}/{path}",
        data=body,
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
      return json.load(resp)

  def obs(self) -> dict[str, Any]:
    return self._req("obs")

  def done(self) -> dict[str, Any]:
    return self._req("done", b"")

  def reset(self) -> dict[str, Any]:
    return self._req("reset", b"")

  def deproject(
      self, u: float, v: float, z_plane: float | None = None
  ) -> tuple[float, float]:
    payload: dict[str, object] = {"u": float(u), "v": float(v)}
    if z_plane is not None:
      payload["z_plane"] = float(z_plane)
    res = self._req("deproject", json.dumps(payload).encode())
    return float(res["x"]), float(res["y"])

  def plan(self, body: str | list[dict[str, object]]) -> dict[str, Any]:
    raw = body if isinstance(body, str) else json.dumps(body)
    return self._req("plan", raw.encode())

  def _op(
      self,
      op: str,
      x: float | None = None,
      y: float | None = None,
      z: float = 0.015,
      *,
      u: float | None = None,
      v: float | None = None,
      **kw: Any,
  ) -> dict[str, Any]:
    item: dict[str, Any] = {"op": op, "z": float(z), **kw}
    if u is not None and v is not None:
      item["u"], item["v"] = float(u), float(v)
    elif x is not None and y is not None:
      item["x"], item["y"] = float(x), float(y)
    elif op not in (
        "move_by",
        "move_until_contact",
        "gripper",
        "home",
        "done",
    ):
      raise ValueError(f"{op} requires (x, y) or (u, v)")
    return self.plan([item])

  def grasp(
      self,
      x: float | None = None,
      y: float | None = None,
      z: float = 0.015,
      **kw: Any,
  ) -> dict[str, Any]:
    return self._op("grasp", x, y, z, **kw)

  def place(
      self,
      x: float | None = None,
      y: float | None = None,
      z: float = 0.075,
      **kw: Any,
  ) -> dict[str, Any]:
    return self._op("place", x, y, z, **kw)

  def place_until_contact(
      self,
      x: float | None = None,
      y: float | None = None,
      z: float = 0.015,
      *,
      u: float | None = None,
      v: float | None = None,
      max_lag: float = 0.012,
      relief: float = 0.002,
      **kw: Any,
  ) -> dict[str, Any]:
    """Transits above (x, y) or (u, v), descends until contact, and releases."""
    return self._op(
        "place_until_contact",
        x,
        y,
        z,
        u=u,
        v=v,
        max_lag=float(max_lag),
        relief=float(relief),
        **kw,
    )

  def goto(
      self,
      x: float | None = None,
      y: float | None = None,
      z: float = 0.12,
      *,
      arc_z: float | None = None,
      **kw: Any,
  ) -> dict[str, Any]:
    if arc_z is not None:
      kw["arc_z"] = float(arc_z)
    return self._op("goto", x, y, z, **kw)

  def look_at(
      self,
      x: float | None = None,
      y: float | None = None,
      z: float = 0.12,
      *,
      pitch: float | None = None,
      pitch_deg: float | None = None,
      **kw: Any,
  ) -> dict[str, Any]:
    """Moves the camera above (x, y) or (u, v) at height z with a top-down pitch."""
    if pitch_deg is not None:
      kw["pitch_deg"] = float(pitch_deg)
    else:
      kw["pitch"] = float(1.57 if pitch is None else pitch)
    return self._op("goto", x, y, z, **kw)

  def move_by(
      self,
      dx: float = 0.0,
      dy: float = 0.0,
      dz: float = 0.0,
      *,
      forward: float = 0.0,
      dyaw_deg: float = 0.0,
      droll_deg: float = 0.0,
      dpitch: float = 0.0,
      dpitch_deg: float = 0.0,
      dazimuth_deg: float = 0.0,
      **kw: Any,
  ) -> dict[str, Any]:
    """Moves the pinch center relative to the current 6-DoF pose."""
    return self.plan([{
        "op": "move_by",
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "forward": float(forward),
        "dyaw_deg": float(dyaw_deg),
        "droll_deg": float(droll_deg),
        "dpitch": float(dpitch),
        "dpitch_deg": float(dpitch_deg),
        "dazimuth_deg": float(dazimuth_deg),
        **kw,
    }])

  def move_until_contact(
      self,
      dx: float = 0.0,
      dy: float = 0.0,
      dz: float = -0.06,
      *,
      forward: float = 0.0,
      speed: float = 0.03,
      max_lag: float = 0.012,
      relief: float = 0.002,
      **kw: Any,
  ) -> dict[str, Any]:
    """Moves relative to the current pose until contact stalls the pinch center."""
    return self.plan([{
        "op": "move_until_contact",
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "forward": float(forward),
        "speed": float(speed),
        "max_lag": float(max_lag),
        "relief": float(relief),
        **kw,
    }])

  def gripper(
      self,
      opening: float,
      *,
      verify_grasp: bool = False,
      threshold: float | None = None,
  ) -> dict[str, Any]:
    """Moves the jaws to `opening` in [0 (closed), 1 (open)].

    With `verify_grasp`, the plan fails (`status="grasp_failed"`) unless the
    measured opening settles more than `threshold` above `opening`.

    Args:
      opening: Target jaw opening.
      verify_grasp: Whether to check that an object stopped the jaws.
      threshold: Optional override of the executor's default threshold.

    Returns:
      The bridge response.
    """
    item: dict[str, Any] = {"op": "gripper", "gripper": float(opening)}
    if verify_grasp:
      item["verify_grasp"] = True
      if threshold is not None:
        item["threshold"] = float(threshold)
    return self.plan([item])

  def open_gripper(self) -> dict[str, Any]:
    return self.gripper(1.0)

  def close_gripper(self, **kw: Any) -> dict[str, Any]:
    return self.gripper(0.0, **kw)

  def sweep(
      self,
      x0: float,
      y0: float,
      x1: float,
      y1: float,
      z: float = 0.02,
  ) -> dict[str, Any]:
    return self.plan(
        f"robot.sweep({float(x0)}, {float(y0)}, {float(x1)}, {float(y1)},"
        f" z={float(z)})"
    )

  def home(self) -> dict[str, Any]:
    return self.plan([{"op": "home"}])


_RemoteRobot = Robot

_COMMANDS = (
    "obs",
    "plan",
    "code",
    "done",
    "reset",
    "goto",
    "look_at",
    "grasp",
    "place",
    "place_until_contact",
    "move_by",
    "move_until_contact",
    "gripper",
    "open_gripper",
    "close_gripper",
    "home",
)


def _extra_kw() -> dict[str, Any]:
  kw: dict[str, Any] = {}
  if _ARC_Z.value is not None:
    kw["arc_z"] = _ARC_Z.value
  if _SPEED.value is not None:
    kw["speed"] = _SPEED.value
  if _YAW_DEG.value is not None:
    kw["yaw_deg"] = _YAW_DEG.value
  if _PITCH_DEG.value is not None:
    kw["pitch_deg"] = _PITCH_DEG.value
  if _ROLL_DEG.value is not None:
    kw["roll_deg"] = _ROLL_DEG.value
  if _AZIMUTH_DEG.value is not None:
    kw["azimuth_deg"] = _AZIMUTH_DEG.value
  if _CLEARANCE.value is not None:
    kw["clearance"] = _CLEARANCE.value
  if _OPENING.value is not None:
    kw["opening"] = _OPENING.value
  if _VERIFY_GRASP.value:
    kw["verify_grasp"] = True
  if _THRESHOLD.value is not None:
    kw["threshold"] = _THRESHOLD.value
  return kw


def main(argv):
  if len(argv) < 2 or argv[1] not in _COMMANDS:
    raise app.UsageError(__doc__)
  cmd = argv[1]
  bot = Robot(_URL.value, _BRIDGE_DIR.value)
  if cmd == "obs":
    out = bot.obs()
  elif cmd == "done":
    out = bot.done()
  elif cmd == "reset":
    out = bot.reset()
  elif cmd == "home":
    out = bot.home()
  elif cmd == "open_gripper":
    out = bot.open_gripper()
  elif cmd in ("gripper", "close_gripper"):
    if cmd == "gripper" and len(argv) < 3:
      raise app.UsageError("Usage: ctl gripper <opening in [0, 1]>")
    opening = float(argv[2]) if cmd == "gripper" else 0.0
    out = bot.gripper(
        opening,
        verify_grasp=_VERIFY_GRASP.value,
        threshold=_THRESHOLD.value,
    )
  elif cmd in ("goto", "look_at", "grasp", "place", "place_until_contact"):
    default_z = {
        "goto": 0.12,
        "look_at": 0.12,
        "grasp": 0.015,
        "place": 0.075,
        "place_until_contact": 0.015,
    }[cmd]
    z = _Z.value if _Z.value is not None else default_z
    kw = _extra_kw()
    if cmd == "place_until_contact":
      kw["max_lag"] = _MAX_LAG.value
      kw["relief"] = _RELIEF.value
    out = getattr(bot, cmd)(
        x=_X.value, y=_Y.value, z=z, u=_CAM_U.value, v=_CAM_V.value, **kw
    )
  elif cmd == "move_by":
    dz = _DZ.value if _DZ.value is not None else 0.0
    out = bot.move_by(
        dx=_DX.value, dy=_DY.value, dz=dz, forward=_FORWARD.value, **_extra_kw()
    )
  elif cmd == "move_until_contact":
    dz = _DZ.value if _DZ.value is not None else -0.06
    speed = _SPEED.value if _SPEED.value is not None else 0.03
    out = bot.move_until_contact(
        dx=_DX.value,
        dy=_DY.value,
        dz=dz,
        forward=_FORWARD.value,
        speed=speed,
        max_lag=_MAX_LAG.value,
        relief=_RELIEF.value,
        **_extra_kw(),
    )
  elif cmd == "plan":
    src = argv[2] if len(argv) > 2 else sys.stdin.read()
    out = bot.plan(src)
  else:
    src = argv[2] if len(argv) > 2 else sys.stdin.read()
    env = {
        "robot": bot,
        "deproject": bot.deproject,
        "grasp": bot.grasp,
        "place": bot.place,
        "place_until_contact": bot.place_until_contact,
        "goto": bot.goto,
        "look_at": bot.look_at,
        "move_by": bot.move_by,
        "move_until_contact": bot.move_until_contact,
        "gripper": bot.gripper,
        "open_gripper": bot.open_gripper,
        "close_gripper": bot.close_gripper,
        "sweep": bot.sweep,
        "home": bot.home,
        "done": bot.done,
        "obs": bot.obs(),
    }
    exec(src, env)  # pylint: disable=exec-used
    out = bot.obs()
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  app.run(main)
