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

"""Primitive plan grammar, smooth 50 Hz plan executor, and agent backends.

A plan is a JSON list of primitives (or Python Code-as-Policies calls on
`robot`). Positions are world meters unless given as wrist-camera pixels
`(u, v)`, which are deprojected onto the plane `z` of the primitive.

  {"op": "grasp",   "x": .., "y": .., "z": .., "yaw_deg": 0, "opening": 1.0}
  {"op": "place",   "x": .., "y": .., "z": .., "yaw_deg": 0, "opening": 1.0}
  {"op": "goto",    "x": .., "y": .., "z": .., "yaw_deg": 0, "pitch": 1.57,
   "gripper": 1.0,  "speed": 0.25}
  {"op": "move_by", "dx": .., "dy": .., "dz": .., "dyaw_deg": 0, "dpitch": 0,
   "gripper": 1.0,  "speed": 0.08}
  {"op": "gripper", "gripper": 0.0, "verify_grasp": true, "threshold": 0.004}
  {"op": "home"}

`yaw`/`yaw_deg` is the world direction of the long axis of whatever is (or
will be) between the jaws; the jaws close across it. `jaw_yaw`/`jaw_yaw_deg`
sets the jaw closing direction directly. Without either, `grasp` centers the
wrist and `place`/`goto` keep the current orientation. `pitch` tilts the
fingers below horizontal (pi/2 points straight down).

`grasp` and `place` expand into rise -> transit at `z + hover` -> vertical
descend -> gripper toggle -> lift. Every segment is a minimum-jerk
point-to-point move; segments before a gripper toggle wait for the arm to
settle. `grasp` opens the jaws to `opening` (0 = closed, 1 = open) before the
descent and `place` opens them to `opening` to release. With `verify_grasp`,
`gripper` and `grasp` abort the plan (`status="grasp_failed"`) unless an object
holds the jaws more than `threshold` above the commanded opening.

Nothing here is specific to one arm or task: robot constants and kinematics come
from an `embodiment.Embodiment`.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import functools
import http.server
import json
import os
import re
import shlex
import subprocess
import threading
import time
from typing import Any
from typing import Protocol

from absl import logging
from etils import epath
import mujoco
from mujoco_playground.tasks.agentic.tabletop_v0 import calibration
from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment as embodiment_lib
import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch

Plan = list[dict[str, Any]]
Embodiment = embodiment_lib.Embodiment

CONTROL_DT = 0.02
PITCH_DOWN = embodiment_lib.PITCH_DOWN
HOVER = 0.06  # Approach height above a grasp/place point (m).
LINEAR_SPEED = 0.30  # Cruise speed of the pinch center (m/s).
ANGULAR_SPEED = 2.6  # Cruise speed of yaw/pitch (rad/s).
MIN_SEGMENT_SECS = 0.16
SETTLE_TOL = 0.005  # Pinch-center tracking error to declare settled (m).
SETTLE_TIMEOUT_SECS = 1.2
GRIPPER_SECS = 0.35  # Minimum dwell while the jaws open or close.
# A grasp is verified when the measured opening stays this far above the
# commanded opening (normalized, ~0.4 mm of jaw aperture on YAM and SO-101).
GRASP_THRESHOLD = 0.004
# Release this far above the requested height: the jaw tips reach 2 cm below
# the pinch center, i.e. 5 mm below the bottom of a held 3 cm cube.
PLACE_CLEARANCE = 0.008
# Nominal half height of the demo objects (3 cm cubes), used to cast pixel
# targets onto the surface they point at.
HALF_HEIGHT = 0.015
GRID_Z = 0.015  # Default height of the labelled table grid in the wrist image.
# Relative jaw yaw is kept this far inside the +-pi/2 wrist band so small
# heading differences between planner and IK never fold it across the edge.
YAW_BAND_HALF = np.pi / 2 - 0.05

OVERLAY_SCALE = 2  # wrist_cam.png is upscaled so grid labels stay legible.


def default_bridge_dir() -> str:
  """Directory for the file side of the bridge (obs.json, wrist_cam.png)."""
  return os.environ.get(
      "SO101_BRIDGE_DIR",
      os.path.join(os.environ.get("TEST_TMPDIR", "/tmp"), "so101_agent_bridge"),
  )


def _item_yaw(item: Mapping[str, Any]) -> float | None:
  """Jaw closing yaw requested by a primitive, or None if unspecified."""
  for key, offset in (("jaw_yaw", 0.0), ("yaw", np.pi / 2)):
    if key in item:
      return float(item[key]) + offset
    if f"{key}_deg" in item:
      return float(np.radians(float(item[f"{key}_deg"]))) + offset
  return None


def _item_pitch(item: Mapping[str, Any], default: float) -> float:
  """Approach pitch (rad below horizontal) from `pitch` or `pitch_deg`."""
  if "pitch" in item:
    return float(item["pitch"])
  if "pitch_deg" in item:
    return float(np.radians(float(item["pitch_deg"])))
  return float(default)


def _item_azimuth(item: Mapping[str, Any]) -> float | None:
  """Horizontal approach azimuth (rad in world XY) or None if radial default."""
  for key in ("azimuth", "approach_yaw"):
    if key in item:
      return float(item[key])
    if f"{key}_deg" in item:
      return float(np.radians(float(item[f"{key}_deg"])))
  return None


def _item_roll(item: Mapping[str, Any]) -> float | None:
  """Jaw closing roll (rad around the approach axis) or None if unspecified."""
  if "roll" in item:
    return float(item["roll"])
  if "roll_deg" in item:
    return float(np.radians(float(item["roll_deg"])))
  return None


# --- Wrist camera geometry (pure FK, identical in sim and real) --------------


Kinematics = calibration.Kinematics
kinematics = calibration.kinematics


def render_wrist_image(
    rgb: torch.Tensor | np.ndarray,
    qpos: np.ndarray,
    emb: Embodiment,
    grid_z: float = GRID_Z,
) -> Image.Image:
  """Renders the wrist image (2x upscaled) with an FK grid and jaw reticle."""
  kin = kinematics(emb)
  arr = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
  arr = np.clip(np.asarray(arr).reshape(-1, *arr.shape[-3:])[0], 0, 255)
  img = Image.fromarray(arr.astype(np.uint8))
  w, h = kin.image_size
  s = OVERLAY_SCALE
  img = img.resize((w * s, h * s), Image.Resampling.BILINEAR)
  draw = ImageDraw.Draw(img)

  def _px(p):
    uv = kin.project(p, qpos)
    return None if uv is None else (uv[0] * s, uv[1] * s)

  def _line(points, color=(170, 210, 255), width=1):
    pts = [_px(p) for p in points]
    pts = [
        p
        for p in pts
        if p is not None and -50 < p[0] < w * s + 50 and -50 < p[1] < h * s + 50
    ]
    if len(pts) >= 2:
      draw.line(pts, fill=color, width=width)

  # 2 cm grid over the workspace footprint at height `grid_z`, labelled every
  # 4 cm.
  x0, y0 = (np.floor(100 * v / 2) * 0.02 for v in emb.workspace_min[:2])
  x1, y1 = (np.ceil(100 * v / 2) * 0.02 for v in emb.workspace_max[:2])
  xs = np.arange(x0, x1 + 1e-3, 0.02)
  ys = np.arange(y0, y1 + 1e-3, 0.02)
  for x in xs:
    _line([(x, y, grid_z) for y in np.linspace(y0, y1, 2 * len(ys))])
  for y in ys:
    _line([(x, y, grid_z) for x in np.linspace(x0, x1, 2 * len(xs))])
  for x in xs:
    for y in ys:
      if round(100 * x) % 4 or round(100 * y) % 4:
        continue
      uv = _px((x, y, grid_z))
      if uv is not None and 4 < uv[0] < w * s - 40 and 4 < uv[1] < h * s - 12:
        text = f"{100 * x:.0f},{100 * y:+.0f}"
        draw.text(
            (uv[0] + 3, uv[1] - 6),
            text,
            fill=(255, 240, 120),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
  pinch, rot = kin.pinch_pose(qpos)
  closing, tangent = rot[:, 0], rot[:, 1]
  # Draw projected 3D jaw brackets at +-22 mm along closing axis and +-10 mm
  # along pad tangent.
  for sign in (-1.0, 1.0):
    jaw_c = pinch + sign * 0.022 * closing
    b0 = jaw_c - 0.010 * tangent
    b1 = jaw_c + 0.010 * tangent
    _line(
        [b0 - sign * 0.005 * closing, b0, b1, b1 - sign * 0.005 * closing],
        color=(0, 255, 255),
        width=2,
    )
  # Draw green long-axis line (the `yaw_deg` object axis between the jaws).
  _line(
      [pinch - 0.018 * tangent, pinch + 0.018 * tangent],
      color=(80, 255, 140),
      width=2,
  )
  uv = _px(pinch)
  if uv is not None:
    px, py = uv
    draw.line([(px - 8, py), (px + 8, py)], fill=(0, 255, 255), width=2)
    draw.line([(px, py - 8), (px, py + 8)], fill=(0, 255, 255), width=2)
  return img


def save_wrist_image(
    rgb: torch.Tensor | np.ndarray,
    path: str,
    qpos: np.ndarray,
    emb: Embodiment,
    grid_z: float = GRID_Z,
) -> str:
  """Saves the wrist image (2x upscaled) with a table grid and a reticle.

  The grid is drawn on the plane z=grid_z every 2 cm; lines on multiples of
  4 cm are labelled with their world `x,y` in centimetres. A cyan reticle and
  3D jaw brackets mark where the pinch center and jaws currently project.
  """
  img = render_wrist_image(rgb, qpos, emb, grid_z=grid_z)
  out = epath.Path(path)
  out.parent.mkdir(parents=True, exist_ok=True)
  img.save(str(out), optimize=True)
  return str(out)


def create_trajectory_strip(
    frames: Sequence[Image.Image],
    labels: Sequence[str] = ("Start", "Mid-motion", "Settled"),
) -> Image.Image:
  """Combines key frames horizontally into a 3-panel labeled trajectory strip."""
  if not frames:
    raise ValueError("frames must not be empty")
  w, h = frames[0].size
  norm_frames = []
  for f in frames:
    if f.size != (w, h):
      norm_frames.append(f.resize((w, h), Image.Resampling.BILINEAR))
    else:
      norm_frames.append(f)
  while len(norm_frames) < 3:
    norm_frames.append(norm_frames[-1])
  strip = Image.new("RGB", (w * 3, h))
  draw = ImageDraw.Draw(strip)
  for i, (f, label) in enumerate(zip(norm_frames[:3], labels[:3])):
    x_offset = i * w
    strip.paste(f, (x_offset, 0))
    if i > 0:
      draw.line([(x_offset, 0), (x_offset, h)], fill=(60, 65, 80), width=2)
    draw.text(
        (x_offset + 12, 10),
        label,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
  return strip


# --- Plan compilation and execution -----------------------------------------


@dataclasses.dataclass
class Segment:
  """One point-to-point move; `pose` is [x, y, z, yaw, pitch, az_off]."""

  pose: np.ndarray
  gripper: float
  settle: bool = False
  verify_grasp: bool = False
  grasp_threshold: float = GRASP_THRESHOLD
  dwell: float = 0.0
  speed: float = LINEAR_SPEED  # Cruise speed of the pinch center (m/s).
  arc_z: float | None = None
  stop_on_contact: bool = False
  contact_lag: float = 0.012
  contact_relief: float = 0.002
  monitor_slip: bool = False
  label: str = ""


def _obs(obs: Mapping[str, Any], key: str) -> np.ndarray:
  val = obs[key]
  if isinstance(val, torch.Tensor):
    val = val.detach().cpu().numpy()
  return np.asarray(val, dtype=np.float64).reshape(-1)


def _default_z_plane(item: Mapping[str, Any], op: str, z: float) -> float:
  """Returns the world Z plane onto which pixel `(u, v)` targets are cast."""
  if "z_plane" in item:
    return float(item["z_plane"])
  if op in ("goto", "look_at"):
    return GRID_Z
  if op in ("place", "place_until_contact"):
    return z - HALF_HEIGHT
  return z


def _xy(
    item: Mapping[str, Any], qpos: np.ndarray, z_plane: float, emb: Embodiment
) -> tuple[float, float]:
  if "u" in item and "v" in item:
    return kinematics(emb).deproject(
        float(item["u"]), float(item["v"]), qpos, z_plane
    )
  return float(item["x"]), float(item["y"])


def resolve_pixels(plan: Plan, qpos: np.ndarray, emb: Embodiment) -> Plan:
  """Replaces pixel `(u, v)` targets by world `(x, y)` using `qpos`.

  Pixels must be cast with the joint state at which the image was captured.
  The arm may still be settling when an image is snapshot, so resolving them
  later against the live joint state (e.g. when the reply arrives) would skew
  the target by the motion in between.

  Args:
    plan: Primitives, some of which may target pixels.
    qpos: Joint positions recorded together with the image.
    emb: The embodiment whose wrist camera took the image.

  Returns:
    The plan with every pixel target converted to world coordinates.
  """
  out: Plan = []
  for item in plan:
    op = item.get("op", "goto")
    u, v = None, None
    if "u" in item and "v" in item:
      u, v = float(item["u"]), float(item["v"])
    elif "pixel" in item and len(item["pixel"]) == 2:
      u, v = float(item["pixel"][0]), float(item["pixel"][1])
    if (
        u is not None
        and v is not None
        and op
        not in (
            "home",
            "move_by",
            "move_until_contact",
            "gripper",
            "done",
        )
    ):
      z = float(item.get("z", HALF_HEIGHT))
      z_plane = _default_z_plane(item, op, z)
      x, y = _xy({"u": u, "v": v}, qpos, z_plane, emb)
      item = {k: v for k, v in item.items() if k not in ("u", "v", "pixel")}
      item |= {"x": x, "y": y, "z": z}
    out.append(item)
  return out


class PlanBuilder:
  """Code-as-Policies helper that records primitives into a `Plan` list."""

  def __init__(self):
    self.plan: Plan = []

  def _add(
      self,
      op: str,
      x: float | None = None,
      y: float | None = None,
      z: float = HALF_HEIGHT,
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
    self.plan.append(item)
    return item

  def grasp(
      self,
      x: float | None = None,
      y: float | None = None,
      z: float = HALF_HEIGHT,
      **kw: Any,
  ) -> dict[str, Any]:
    return self._add("grasp", x, y, z, **kw)

  def place(
      self,
      x: float | None = None,
      y: float | None = None,
      z: float = 0.075,
      **kw: Any,
  ) -> dict[str, Any]:
    return self._add("place", x, y, z, **kw)

  def place_until_contact(
      self,
      x: float | None = None,
      y: float | None = None,
      z: float = HALF_HEIGHT,
      *,
      u: float | None = None,
      v: float | None = None,
      max_lag: float = 0.012,
      relief: float = 0.002,
      **kw: Any,
  ) -> dict[str, Any]:
    """Transits above (x, y) or (u, v), descends until contact, and releases."""
    return self._add(
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
    return self._add("goto", x, y, z, **kw)

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
      kw["pitch"] = float(PITCH_DOWN if pitch is None else pitch)
    return self._add("goto", x, y, z, **kw)

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
    item: dict[str, Any] = {
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
    }
    self.plan.append(item)
    return item

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
    item: dict[str, Any] = {
        "op": "move_until_contact",
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "forward": float(forward),
        "speed": float(speed),
        "max_lag": float(max_lag),
        "relief": float(relief),
        **kw,
    }
    self.plan.append(item)
    return item

  def gripper(
      self,
      opening: float,
      *,
      verify_grasp: bool = False,
      threshold: float = GRASP_THRESHOLD,
  ) -> dict[str, Any]:
    """Moves the jaws to normalized `opening` in [0 (closed), 1 (open)].

    Args:
      opening: Target jaw opening.
      verify_grasp: If true, abort the plan with `status="grasp_failed"` unless
        the measured opening settles more than `threshold` above `opening`,
        i.e. an object stopped the jaws.
      threshold: Minimum measured-minus-commanded opening of a verified grasp.

    Returns:
      The recorded primitive.
    """
    item: dict[str, Any] = {"op": "gripper", "gripper": float(opening)}
    if verify_grasp:
      item |= {"verify_grasp": True, "threshold": float(threshold)}
    self.plan.append(item)
    return item

  def open_gripper(self) -> dict[str, Any]:
    """Opens the gripper jaws in place."""
    return self.gripper(1.0)

  def close_gripper(self, **kw: Any) -> dict[str, Any]:
    """Closes the gripper jaws in place (see `gripper` for `kw`)."""
    return self.gripper(0.0, **kw)

  def sweep(
      self,
      x0: float,
      y0: float,
      x1: float,
      y1: float,
      z: float = 0.02,
  ) -> Plan:
    """Pushes along a line with closed jaws, the broad pad face leading."""
    # Closing axis along the push: the pads' flat faces are the pusher.
    jaw_yaw = float(np.arctan2(y1 - y0, x1 - x0))
    kw = {"jaw_yaw": jaw_yaw, "pitch": PITCH_DOWN}
    self.goto(x0, y0, z + HOVER, gripper=0.0, **kw)
    self.goto(x0, y0, z, gripper=0.0, **kw)
    self.goto(x1, y1, z, gripper=0.0, **kw)
    self.goto(x1, y1, z + HOVER, gripper=1.0, **kw)
    return self.plan

  def home(self) -> dict[str, Any]:
    item = {"op": "home"}
    self.plan.append(item)
    return item

  def done(self) -> dict[str, Any]:
    """Declares that the task is complete after visual verification."""
    item = {"op": "done"}
    self.plan.append(item)
    return item


def _extract_plan_metadata(text: str) -> tuple[str | None, str | None]:
  """Extracts optional `# Plan:` strategy and `# Thought:` reasoning from a plan payload."""
  strategy: str | None = None
  thought: str | None = None
  stripped = text.strip()
  try:
    data = json.loads(stripped)
    if isinstance(data, dict):
      if data.get("strategy") or data.get("plan_summary"):
        strategy = str(data.get("strategy") or data.get("plan_summary")).strip()
      if data.get("thought"):
        thought = str(data["thought"]).strip()
  except (json.JSONDecodeError, TypeError):
    pass
  for line in stripped.splitlines():
    m_plan = re.match(
        r"^\s*#\s*(?:plan|strategy)\s*:\s*(.+)$", line, re.IGNORECASE
    )
    if m_plan:
      strategy = m_plan.group(1).strip()
    m_thought = re.match(
        r"^\s*#\s*(?:thought|observation|reasoning)\s*:\s*(.+)$",
        line,
        re.IGNORECASE,
    )
    if m_thought:
      thought = m_thought.group(1).strip()
  return strategy, thought


def parse_plan(text: str) -> Plan:
  """Parses either a JSON primitive list or a Python Code-as-Policies snippet."""
  stripped = text.strip()
  fenced = re.search(r"```(?:python|json)?\s*\n(.*?)```", stripped, re.DOTALL)
  body = fenced.group(1).strip() if fenced else stripped
  try:
    match = re.search(r"\[.*\]", body, flags=re.DOTALL)
    plan = json.loads(match.group(0) if match else body)
    if isinstance(plan, dict):
      if "code" in plan:
        body = plan["code"]
      elif "plan" in plan and isinstance(plan["plan"], list):
        return list(plan["plan"])
      elif "op" in plan:
        return [plan]
    elif isinstance(plan, list) and all(isinstance(x, dict) for x in plan):
      return list(plan)
  except (json.JSONDecodeError, TypeError):
    pass
  robot = PlanBuilder()
  env = {
      "robot": robot,
      "grasp": robot.grasp,
      "place": robot.place,
      "place_until_contact": robot.place_until_contact,
      "goto": robot.goto,
      "look_at": robot.look_at,
      "move_by": robot.move_by,
      "move_until_contact": robot.move_until_contact,
      "gripper": robot.gripper,
      "open_gripper": robot.open_gripper,
      "close_gripper": robot.close_gripper,
      "sweep": robot.sweep,
      "home": robot.home,
      "done": robot.done,
      "np": np,
  }
  exec(body, env)  # pylint: disable=exec-used
  return robot.plan


class PlanExecutor:
  """Turns primitives into smooth 50 Hz `[x, y, z, gripper, yaw, pitch, az_off]`."""

  def __init__(
      self,
      emb: Embodiment | str,
      dt: float = CONTROL_DT,
      settle_ticks: int = 6,
      speed_scale: float = 1.0,
  ):
    self._emb = embodiment_lib.get(emb)
    self._dt = dt
    self._settle_ticks = settle_ticks
    self._speed_scale = float(np.clip(speed_scale, 0.05, 2.0))
    self._settle_ticks_left = 0
    self._pose = np.array([*self._emb.home(), 0.0], dtype=np.float64)
    self._gripper = 1.0
    self._start_gripper = 1.0
    self._segments: list[Segment] = []
    self._start: np.ndarray | None = None
    self._goal: np.ndarray = self._pose.copy()
    self._u_dir: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    self._prev_ee: np.ndarray | None = None
    self._t = 0.0
    self._duration = 0.0
    self._settle_t = 0.0
    self._az_delta = 0.0
    self._contact_ticks = 0
    self._slip_ticks = 0
    self._grip_settle_ticks = 0
    # (commanded opening, threshold) of the last verified grasp, while held.
    self._hold: tuple[float, float] | None = None
    self.stopped_on_contact: bool = False
    self.contact_pos: list[float] | None = None
    self.grid_z = GRID_Z
    self.status = "idle"

  @property
  def dt(self) -> float:
    return self._dt

  @property
  def embodiment(self) -> Embodiment:
    return self._emb

  @property
  def busy(self) -> bool:
    return (
        bool(self._segments)
        or self._start is not None
        or self._settle_ticks_left > 0
    )

  def _grasp_ok(
      self, obs: Mapping[str, Any], gripper: float, threshold: float
  ) -> bool:
    """True if an object holds the jaws `threshold` above command `gripper`."""
    opening = float(_obs(obs, "gripper_state")[0])
    return opening - self._emb.commanded_opening(gripper) > threshold

  def holding(self, obs: Mapping[str, Any]) -> bool:
    """Whether the jaws are held open past their command (default threshold)."""
    return self._grasp_ok(obs, self._gripper, GRASP_THRESHOLD)

  def sync(self, obs: Mapping[str, Any]) -> None:
    """Adopts the measured pose as the commanded pose (after reset)."""
    ori = _obs(obs, "ee_ori")
    az_off = float(ori[2]) if len(ori) > 2 else 0.0
    self._pose = np.array([*_obs(obs, "ee_pos"), ori[0], ori[1], az_off])
    self._gripper = float(_obs(obs, "gripper_state")[0] > 0.5)
    self._start_gripper = self._gripper
    self._segments, self._start = [], None
    self._settle_ticks_left = 0
    self._contact_ticks = 0
    self._slip_ticks = 0
    self._grip_settle_ticks = 0
    self._hold = None
    self.stopped_on_contact = False
    self.contact_pos = None
    self.grid_z = GRID_Z
    self.status = "idle"

  def load(self, plan: Plan, obs: Mapping[str, Any]) -> None:
    """Compiles and queues a plan; replaces anything still pending."""
    emb = self._emb
    qpos = _obs(obs, "joint_pos")
    if "ee_pos" in obs and not self.busy:
      meas_ee = _obs(obs, "ee_pos")
      if float(np.linalg.norm(meas_ee - self._pose[:3])) > 0.008:
        self._pose[:3] = meas_ee
    segs: list[Segment] = []
    pose = self._pose.copy()
    grip = self._gripper
    z_floor = emb.workspace_min[2]
    self.stopped_on_contact = False
    self.contact_pos = None
    self._contact_ticks = 0
    self._slip_ticks = 0
    self._settle_ticks_left = 0
    for item in plan:
      op = item.get("op", "goto")
      if op == "done":
        continue
      if "grid_z" in item:
        self.grid_z = float(item["grid_z"])
      elif "z_plane" in item:
        self.grid_z = float(item["z_plane"])
      if op == "home":
        self.grid_z = GRID_Z
        pose = np.array([*emb.home(), 0.0], dtype=np.float64)
        segs.append(Segment(pose.copy(), grip, label="home"))
        continue
      if op == "gripper":
        grip = float(item.get("gripper", 1.0))
        segs.append(
            Segment(
                pose.copy(),
                grip,
                dwell=GRIPPER_SECS,
                verify_grasp=bool(item.get("verify_grasp", False)),
                grasp_threshold=float(item.get("threshold", GRASP_THRESHOLD)),
                label="gripper",
            )
        )
        continue
      if op in ("move_by", "move_until_contact"):
        is_guarded = op == "move_until_contact"
        default_dz = (
            -0.06
            if is_guarded
            and not any(
                k in item for k in ("dx", "dy", "dz", "forward", "dforward")
            )
            else 0.0
        )
        forward = float(item.get("forward", item.get("dforward", 0.0)))
        cur_az = emb.heading(pose[0], pose[1]) + pose[5]
        app = emb.approach_vector(pose[0], pose[1], pose[4], cur_az)
        dx = float(item.get("dx", 0.0)) + forward * float(app[0])
        dy = float(item.get("dy", 0.0)) + forward * float(app[1])
        dz = float(item.get("dz", default_dz)) + forward * float(app[2])
        dyaw = (
            float(np.radians(float(item.get("dyaw_deg", 0.0))))
            + float(item.get("dyaw", 0.0))
            + float(np.radians(float(item.get("droll_deg", 0.0))))
            + float(item.get("droll", 0.0))
        )
        dpitch = float(np.radians(float(item.get("dpitch_deg", 0.0)))) + float(
            item.get("dpitch", 0.0)
        )
        daz = float(np.radians(float(item.get("dazimuth_deg", 0.0)))) + float(
            item.get("dazimuth", 0.0)
        )
        nx = pose[0] + dx
        ny = pose[1] + dy
        nz = max(pose[2] + dz, z_floor)
        if abs(pose[5]) > 1e-6 or abs(daz) > 1e-6:
          az_off = embodiment_lib.wrap_pi((cur_az + daz) - emb.heading(nx, ny))
        else:
          az_off = 0.0
        grip = float(item.get("gripper", grip))
        default_speed = 0.03 if is_guarded else emb.approach_speed
        speed = float(item.get("speed", default_speed))
        max_lag = float(item.get("max_lag", 0.012))
        relief = float(item.get("relief", 0.002))
        pose = np.array([
            nx,
            ny,
            nz,
            embodiment_lib.fold(pose[3] + dyaw + daz),
            pose[4] + dpitch,
            az_off,
        ])
        segs.append(
            Segment(
                pose.copy(),
                grip,
                settle=True,
                speed=speed,
                stop_on_contact=is_guarded,
                contact_lag=max_lag,
                contact_relief=relief,
                label=op,
            )
        )
        continue

      z = float(item.get("z", HALF_HEIGHT))
      z_plane = _default_z_plane(item, op, z)
      x, y = _xy(item, qpos, z_plane, emb)
      req_az = _item_azimuth(item)
      if req_az is not None:
        azimuth = req_az
        az_off = embodiment_lib.wrap_pi(azimuth - emb.heading(x, y))
      elif op == "goto" and abs(pose[5]) > 1e-6:
        azimuth = emb.heading(pose[0], pose[1]) + pose[5]
        az_off = embodiment_lib.wrap_pi(azimuth - emb.heading(x, y))
      else:
        azimuth = emb.heading(x, y)
        az_off = 0.0
      roll = _item_roll(item)
      yaw = _item_yaw(item)
      if roll is not None:
        yaw = azimuth + roll
      elif yaw is None:  # Grasps center the wrist; other moves keep the jaws.
        yaw = azimuth + emb.wrist_yaw_offset if op == "grasp" else pose[3]
      yaw = embodiment_lib.fold(yaw)
      if op == "goto":
        pitch = _item_pitch(item, pose[4])
        speed = float(item.get("speed", LINEAR_SPEED))
        arc_z = float(item["arc_z"]) if item.get("arc_z") is not None else None
        pose = np.array([x, y, max(z, z_floor), yaw, pitch, az_off])
        grip = float(item.get("gripper", grip))
        segs.append(
            Segment(
                pose.copy(),
                grip,
                settle=True,
                speed=speed,
                arc_z=arc_z,
                label="goto",
            )
        )
        continue

      if op not in ("grasp", "place", "place_until_contact"):
        raise ValueError(f"Unknown op {op!r}")
      is_place = op in ("place", "place_until_contact")
      stop_contact = op == "place_until_contact"
      max_lag = float(item.get("max_lag", 0.012))
      relief = float(item.get("relief", 0.002))
      hover = float(item.get("hover", HOVER))
      verify = op == "grasp" and bool(item.get("verify_grasp", False))
      threshold = float(item.get("threshold", GRASP_THRESHOLD))
      pitch = _item_pitch(item, emb.grasp_pitch(x, y))
      explicit_pitch = "pitch" in item or "pitch_deg" in item
      coaxial = bool(item.get("coaxial", explicit_pitch and pitch < 1.05))
      if op == "place" and not coaxial:
        clearance = float(item.get("clearance", PLACE_CLEARANCE))
        z += clearance
      # The jaws may open partway to fit between walls or close neighbours.
      opening = float(item.get("opening", 1.0))
      pre, post = (grip, opening) if is_place else (opening, 0.0)
      z = max(z, z_floor)
      down = np.array([x, y, z, yaw, pitch, az_off])
      if coaxial:
        app = emb.approach_vector(x, y, pitch, azimuth)
        standoff = np.array([x, y, z]) - hover * app
        standoff[2] = max(standoff[2], z_floor)
        st_az_off = (
            embodiment_lib.wrap_pi(
                azimuth - emb.heading(standoff[0], standoff[1])
            )
            if req_az is not None or abs(az_off) > 1e-6
            else 0.0
        )
        above = np.array([*standoff, yaw, pitch, st_az_off])
        post_pose = (
            np.array([x, y, max(z + hover, standoff[2]), yaw, pitch, az_off])
            if op == "grasp"
            else above.copy()
        )
      else:
        above = np.array([x, y, z + hover, yaw, pitch, az_off])
        post_pose = above.copy()
      rise = pose.copy()
      rise[2] = max(rise[2], above[2])
      lateral = float(np.hypot(above[0] - pose[0], above[1] - pose[1]))
      if item.get("arc_z") is not None:
        arc_z = float(item["arc_z"])
      elif lateral > 0.02:
        arc_z = float(max(pose[2], above[2]) + 0.015)
      else:
        arc_z = None
      descend_speed = float(
          item.get("speed", 0.03 if stop_contact else emb.approach_speed)
      )
      segs += [
          Segment(rise, grip, monitor_slip=is_place, label=f"{op}:rise"),
          # Settle at the pre-grasp/pre-place standoff so the approach starts on axis.
          Segment(
              above,
              pre,
              settle=True,
              arc_z=arc_z,
              monitor_slip=is_place,
              label=f"{op}:transit",
          ),
          Segment(
              down,
              pre,
              settle=True,
              speed=descend_speed,
              stop_on_contact=stop_contact,
              contact_lag=max_lag,
              contact_relief=relief,
              label=f"{op}:descend",
          ),
          Segment(
              down,
              post,
              dwell=GRIPPER_SECS,
              verify_grasp=verify,
              grasp_threshold=threshold,
              label=f"{op}:gripper",
          ),
          Segment(
              post_pose,
              post,
              settle=True,
              verify_grasp=verify,
              grasp_threshold=threshold,
              label=f"{op}:lift",
          ),
      ]
      pose, grip = post_pose, post
    self._segments = segs
    self._start = None
    self.status = "running"

  def step(self, obs: Mapping[str, Any]) -> np.ndarray:
    """Advances one control tick and returns the 7-dim action."""
    if self._start is None and self._segments:
      self._begin(self._segments[0])
    if self._start is not None:
      seg = self._segments[0]
      self._t += self._dt
      raw_s = min(self._t / self._duration, 1.0)
      s = 10 * raw_s**3 - 15 * raw_s**4 + 6 * raw_s**5  # Minimum jerk.
      self._pose = self._start + s * (self._goal - self._start)
      self._gripper = self._start_gripper + s * (
          seg.gripper - self._start_gripper
      )
      if seg.arc_z is not None:
        z0, z1 = float(self._start[2]), float(self._goal[2])
        bulge = max(0.0, float(seg.arc_z) - 0.5 * (z0 + z1))
        self._pose[2] = (1.0 - s) * z0 + s * z1 + 4.0 * s * (1.0 - s) * bulge
      self._pose[5] = embodiment_lib.wrap_pi(
          self._start[5] + s * self._az_delta
      )
      rel = self._rel0 + s * (self._rel1 - self._rel0)
      self._pose[3] = self._emb.heading(*self._pose[:2]) + self._pose[5] + rel
      if (
          seg.monitor_slip
          and self._hold is not None
          and self._t >= 3 * self._dt - 1e-9
      ):
        if self._grasp_ok(obs, *self._hold):
          self._slip_ticks = 0
        else:
          self._slip_ticks += 1
        if self._slip_ticks >= 3:
          logging.info(
              "%s: slip detected, gripper_state=%.4f, commanded=%.4f",
              seg.label,
              float(_obs(obs, "gripper_state")[0]),
              self._hold[0],
          )
          self._segments.clear()
          self._start = None
          self._gripper = 1.0
          self._start_gripper = 1.0
          self._hold = None
          self.status = "grasp_failed"
          self._settle_ticks_left = 0
          return np.array(
              [*self._pose[:3], self._gripper, *self._pose[3:6]],
              dtype=np.float32,
          )
      if seg.stop_on_contact and self._t >= 3 * self._dt - 1e-9:
        ee = _obs(obs, "ee_pos")
        axial_lag = float(np.dot(self._pose[:3] - ee, self._u_dir))
        progress_step = (
            float(np.linalg.norm(ee - self._prev_ee))
            if self._prev_ee is not None
            else 1.0
        )
        stalled = (
            progress_step < 5e-4
            and float(np.linalg.norm(self._pose[:3] - ee)) > seg.contact_lag
        )
        self._prev_ee = ee.copy()
        if axial_lag >= seg.contact_lag or stalled:
          self._contact_ticks += 1
        else:
          self._contact_ticks = 0
        if self._contact_ticks >= 3:
          self.stopped_on_contact = True
          self.contact_pos = [round(float(v), 4) for v in ee]
          self.status = "contact"
          self._pose[:3] = ee - seg.contact_relief * self._u_dir
          self._pose[3] = embodiment_lib.fold(self._pose[3])
          old_down = seg.pose[:3].copy()
          self._segments.pop(0)
          self._start = None
          for next_seg in self._segments:
            if next_seg.label == "gripper" or next_seg.label.endswith(
                ":gripper"
            ):
              next_seg.pose[:3] = self._pose[:3].copy()
            elif next_seg.label.endswith(":lift"):
              next_seg.pose[:3] = self._pose[:3] + (
                  next_seg.pose[:3] - old_down
              )
            else:
              break
          if not self._segments and self._settle_ticks > 0:
            self._settle_ticks_left = self._settle_ticks
          return np.array(
              [*self._pose[:3], self._gripper, *self._pose[3:6]],
              dtype=np.float32,
          )
      if self._t >= self._duration and self._finished(seg, obs):
        self._segments.pop(0)
        self._start = None
        self._pose = seg.pose.copy()
        self._pose[3] = embodiment_lib.fold(self._pose[3])
        self._gripper = seg.gripper
        self._start_gripper = seg.gripper
        ok = not seg.verify_grasp or self._grasp_ok(
            obs, seg.gripper, seg.grasp_threshold
        )
        if seg.verify_grasp and ok:
          self._hold = (seg.gripper, seg.grasp_threshold)
        if not ok:
          logging.info(
              "%s: empty grasp, gripper_state=%.4f, commanded=%.4f",
              seg.label,
              float(_obs(obs, "gripper_state")[0]),
              seg.gripper,
          )
          self._abort_grasp(seg)
        elif not self._segments:
          if self._settle_ticks > 0:
            self._settle_ticks_left = self._settle_ticks
          elif self.status == "running":
            self.status = "done"
    elif self._settle_ticks_left > 0:
      self._settle_ticks_left -= 1
      if self._settle_ticks_left == 0 and self.status == "running":
        self.status = "done"
    return np.array(
        [*self._pose[:3], self._gripper, *self._pose[3:6]], dtype=np.float32
    )

  def _band_yaw(
      self, yaw: float, x: float, y: float, az_off: float = 0.0
  ) -> float:
    """Wrist-relative jaw yaw, kept off the band edges where it would flip."""
    emb = self._emb
    off = emb.wrist_yaw_offset
    azimuth = emb.heading(x, y) + az_off
    rel = emb.relative_yaw(yaw, x, y, azimuth)
    return float(np.clip(rel, off - YAW_BAND_HALF, off + YAW_BAND_HALF))

  def _begin(self, seg: Segment) -> None:
    if self.status == "contact" and not (
        seg.label == "gripper" or seg.label.endswith((":gripper", ":lift"))
    ):
      self.status = "running"
    if self._hold is not None and abs(seg.gripper - self._hold[0]) > 1e-6:
      self._hold = None
    self._start = self._pose.copy()
    self._start_gripper = self._gripper
    self._goal = seg.pose.copy()
    # The IK processor folds yaw into the wrist band around the approach
    # azimuth; interpolating the relative yaw inside that band (rather than
    # world yaw the short way mod pi) keeps the wrist from flipping by pi
    # halfway through a move.
    self._rel0 = self._band_yaw(
        self._start[3], self._start[0], self._start[1], self._start[5]
    )
    self._rel1 = self._band_yaw(
        self._goal[3], self._goal[0], self._goal[1], self._goal[5]
    )
    self._az_delta = embodiment_lib.wrap_pi(self._goal[5] - self._start[5])
    vec = self._goal[:3] - self._start[:3]
    linear_dist = float(np.linalg.norm(vec))
    self._u_dir = (
        vec / linear_dist
        if linear_dist > 1e-4
        else np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )
    if seg.arc_z is not None:
      z0, z1 = float(self._start[2]), float(self._goal[2])
      bulge = max(0.0, float(seg.arc_z) - 0.5 * (z0 + z1))
      dist = float(
          np.sqrt(vec[0] ** 2 + vec[1] ** 2 + (abs(z1 - z0) + 2.0 * bulge) ** 2)
      )
    else:
      dist = linear_dist
    dang = max(
        abs(self._rel1 - self._rel0),
        abs(self._goal[4] - self._start[4]),
        abs(self._az_delta),
    )
    lin_speed = seg.speed * self._speed_scale
    ang_speed = ANGULAR_SPEED * self._speed_scale
    min_secs = MIN_SEGMENT_SECS / self._speed_scale
    if (
        dist < 1e-4
        and dang < 1e-4
        and abs(seg.gripper - self._start_gripper) < 1e-4
        and seg.dwell <= 0
    ):
      self._duration = self._dt
    else:
      self._duration = max(
          dist / lin_speed, dang / ang_speed, min_secs, seg.dwell
      )
    self._t = 0.0
    self._settle_t = 0.0
    self._prev_grip = -1.0
    self._prev_ee = None
    self._contact_ticks = 0
    self._slip_ticks = 0
    self._grip_settle_ticks = 0

  def _finished(self, seg: Segment, obs: Mapping[str, Any]) -> bool:
    """Settle gate: arm tracked (descend) or jaws stopped moving (gripper)."""
    self._settle_t += self._dt
    if self._settle_t > SETTLE_TIMEOUT_SECS:
      if seg.settle:
        logging.info(
            "%s: settle timeout, pinch %s vs target %s",
            seg.label,
            np.round(_obs(obs, "ee_pos"), 4),
            np.round(seg.pose[:3], 4),
        )
      return True
    if seg.settle:
      err = np.abs(_obs(obs, "ee_pos") - seg.pose[:3])
      return np.hypot(err[0], err[1]) < SETTLE_TOL and err[2] < 3 * SETTLE_TOL
    if seg.dwell > 0:
      grip = float(_obs(obs, "gripper_state")[0])
      if self._prev_grip < 0:
        self._prev_grip = grip
        self._grip_settle_ticks = 0
        return False
      if abs(grip - self._prev_grip) < 1.5e-3:
        self._grip_settle_ticks += 1
      else:
        self._grip_settle_ticks = 0
      self._prev_grip = grip
      return self._grip_settle_ticks >= 3
    return True

  def _abort_grasp(self, seg: Segment) -> None:
    """Empty jaws: open, lift, and drop the rest of the plan for re-planning."""
    segs = [
        Segment(seg.pose.copy(), 1.0, dwell=GRIPPER_SECS, label="abort:open"),
    ]
    if not seg.label.endswith(":lift"):
      lift = seg.pose.copy()
      lift[2] += HOVER
      segs.append(Segment(lift, 1.0, label="abort:lift"))
    self._segments = segs
    self._hold = None
    self.status = "grasp_failed"


# --- Agent backends -----------------------------------------------------------


class AgentBackend(Protocol):
  """Source of plans. `plan` is polled every tick while the executor is idle."""

  @property
  def turn(self) -> int:
    ...

  @property
  def max_turns(self) -> int:
    ...

  @property
  def exhausted(self) -> bool:
    ...

  def on_reset(self, obs: Mapping[str, Any]) -> None:
    ...

  def plan(self, obs: Mapping[str, Any]) -> Plan | None:
    ...

  def on_plan_done(
      self, obs: Mapping[str, Any], status: str, success: bool
  ) -> None:
    ...

  def on_step(self, obs: Mapping[str, Any]) -> None:
    ...


class OracleBackend:
  """Plans from privileged simulator state; re-plans after failed grasps."""

  def __init__(self, env: Any, task: Any, max_turns: int = 3):
    self._env = env
    self._task = task
    self._max_turns = max_turns
    self._turn = 0
    self._pending = True

  @property
  def turn(self) -> int:
    return self._turn

  @property
  def max_turns(self) -> int:
    return self._max_turns

  def on_reset(self, obs: Mapping[str, Any]) -> None:
    del obs
    self._turn = 0
    self._pending = True

  def poses(self) -> dict[str, np.ndarray]:
    sim = self._env.device_layer.sim
    mjm, mjd = sim.mj_model, sim.mj_data
    mjd.qpos[:] = sim.data.qpos[0].cpu().numpy()
    mujoco.mj_kinematics(mjm, mjd)
    out = {}
    for obj in self._task.objects:
      body = mjm.body(f"{obj.name}/{obj.name}").id
      q = mjd.xquat[body]
      yaw = np.arctan2(
          2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
      )
      out[obj.name] = np.array([*mjd.xpos[body], yaw])
    return out

  def plan(self, obs: Mapping[str, Any]) -> Plan | None:
    del obs
    if not self._pending or (0 < self._max_turns <= self._turn):
      return None
    self._pending = False
    self._turn += 1
    poses = self.poses()
    logging.info(
        "oracle plan %d from poses %s",
        self._turn,
        {k: np.round(v, 3).tolist() for k, v in poses.items()},
    )
    return self._task.oracle(poses)

  def on_plan_done(
      self, obs: Mapping[str, Any], status: str, success: bool
  ) -> None:
    del obs
    logging.info("oracle plan %d: %s success=%s", self._turn, status, success)
    # Re-plan from fresh poses whenever the task is not solved yet (a failed
    # grasp, or a cube nudged off the stack), up to `max_turns`.
    self._pending = not success

  def on_step(self, obs: Mapping[str, Any]) -> None:
    del obs

  @property
  def exhausted(self) -> bool:
    """True once no further plans will be issued."""
    return not self._pending or (0 < self._max_turns <= self._turn)


class BridgeBackend:
  """HTTP + file bridge for an external agent (IDE agent, CLI, notebook).

  `GET /obs` returns the latest observation payload; `POST /plan` with a JSON
  plan blocks until it has executed and returns the new payload;
  `POST /done` marks the episode finished; `POST /reset` requests an episode
  reset. The same payload is mirrored to `bridge_dir/obs.json` next to
  `wrist_cam.png`, alongside an episode evidence bundle (`obs_frames/`,
  `steps.jsonl`, and `report.md`).
  """

  def __init__(
      self,
      task_prompt: str,
      bridge_dir: str,
      http_port: int,
      emb: Embodiment | str,
      max_turns: int = 0,
  ):
    self._prompt = task_prompt
    self._emb = embodiment_lib.get(emb)
    self._executor = PlanExecutor(self._emb)
    self._max_turns = max_turns
    self._dir = epath.Path(bridge_dir)
    self._dir.mkdir(parents=True, exist_ok=True)
    (self._dir / "obs_frames").mkdir(parents=True, exist_ok=True)
    self._lock = threading.Lock()
    self._payload: dict[str, Any] = {}
    self._pending: Plan | None = None
    self._last_plan: Plan = []
    self._steps_log: list[dict[str, Any]] = []
    self._reset_requested = False
    self._done = False
    self._plan_done = threading.Event()
    self._turn = 0
    self._history: list[dict[str, Any]] = []
    self._image_qpos: np.ndarray | None = None  # Joints when imaged.
    self.grid_z = GRID_Z
    self._frame_0: Image.Image | None = None
    self._frame_1: Image.Image | None = None
    self._frame_2: Image.Image | None = None
    self._frame_1_captured: bool = False
    self._strategy: str = ""
    self._last_thought: str = ""
    self._gif_frames: list[Image.Image] = []
    self._server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", http_port), self._handler()
    )
    threading.Thread(target=self._server.serve_forever, daemon=True).start()
    self.http_port = self._server.server_address[1]
    (self._dir / "bridge_info.json").write_text(
        json.dumps({
            "http_url": f"http://127.0.0.1:{self.http_port}",
            "bridge_dir": str(self._dir),
            "pid": os.getpid(),
        })
    )

  @property
  def turn(self) -> int:
    return self._turn

  @property
  def max_turns(self) -> int:
    return self._max_turns

  @property
  def exhausted(self) -> bool:
    with self._lock:
      return self._pending is None and (
          self._done or (0 < self._max_turns <= self._turn)
      )

  def close(self) -> None:
    self._plan_done.set()  # Reply to any handler still waiting on a plan.
    self._server.shutdown()
    self._server.server_close()

  def _handler(self):
    backend = self

    class Handler(http.server.BaseHTTPRequestHandler):
      # pylint: disable=protected-access

      def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
        del format, args

      def _send(self, code: int, data: dict[str, Any]) -> None:
        raw = json.dumps(data, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

      def do_GET(self):  # pylint: disable=invalid-name
        with backend._lock:
          self._send(200, dict(backend._payload))

      def do_POST(self):  # pylint: disable=invalid-name
        body = self.rfile.read(
            int(self.headers.get("Content-Length", "0"))
        ).decode()
        if self.path.startswith("/reset"):
          backend._plan_done.clear()
          with backend._lock:
            backend._reset_requested = True
            backend._pending = None
            backend._done = False
            if body:
              try:
                data = json.loads(body)
                if isinstance(data, dict):
                  custom_task = data.get("task") or data.get("task_prompt")
                  if custom_task:
                    backend._prompt = str(custom_task)
              except Exception:
                pass
        elif self.path.startswith("/done"):
          strat, th = _extract_plan_metadata(body) if body else (None, None)
          with backend._lock:
            if strat:
              backend._strategy = strat
            if th:
              backend._last_thought = th
            backend._done = True
            backend._pending = None
            backend._last_plan = [{"op": "done"}]
            backend._payload["status"] = "episode_end"
            backend._payload["strategy"] = backend._strategy
            backend._payload["thought"] = backend._last_thought
            (backend._dir / "obs.json").write_text(
                json.dumps(backend._payload, indent=1)
            )
            payload = dict(backend._payload)
          backend._write_report(bool(payload.get("success")))
          backend._plan_done.set()
          self._send(200, payload)
          return
        elif self.path.startswith("/deproject"):
          try:
            req = json.loads(body)
            with backend._lock:
              qpos = backend._image_qpos
              z_p = float(req.get("z_plane", backend.grid_z))
            if qpos is None:
              self._send(400, {"error": "no image captured yet"})
              return
            x, y = kinematics(backend._emb).deproject(
                float(req["u"]), float(req["v"]), qpos, z_p
            )
            self._send(
                200,
                {
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "z_plane": round(z_p, 4),
                },
            )
          except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
          return
        elif self.path.startswith("/plan"):
          with backend._lock:
            if backend._done or (0 < backend._max_turns <= backend._turn):
              self._send(
                  400,
                  {
                      "error": (
                          f"episode exhausted (turn {backend._turn},"
                          f" max_turns {backend._max_turns})"
                      )
                  },
              )
              return
          try:
            try:
              data = json.loads(body)
              if isinstance(data, dict):
                custom_task = data.get("task") or data.get("task_prompt")
                if custom_task:
                  with backend._lock:
                    backend._prompt = str(custom_task)
            except Exception:
              pass
            strat, th = _extract_plan_metadata(body)
            with backend._lock:
              if strat:
                backend._strategy = strat
              backend._last_thought = th or ""
            raw_plan = parse_plan(body)
          except (ValueError, KeyError, SyntaxError, TypeError) as exc:
            self._send(400, {"error": str(exc)})
            return
          has_done = any(item.get("op") == "done" for item in raw_plan)
          plan = [item for item in raw_plan if item.get("op") != "done"]
          if has_done and not plan:
            with backend._lock:
              backend._done = True
              backend._pending = None
              backend._last_plan = [{"op": "done"}]
              backend._payload["status"] = "episode_end"
              backend._payload["strategy"] = backend._strategy
              backend._payload["thought"] = backend._last_thought
              (backend._dir / "obs.json").write_text(
                  json.dumps(backend._payload, indent=1)
              )
              payload = dict(backend._payload)
            backend._write_report(bool(payload.get("success")))
            backend._plan_done.set()
            self._send(200, payload)
            return
          backend._plan_done.clear()
          with backend._lock:
            if has_done:
              backend._done = True
            backend._pending = plan
        else:
          self._send(
              404,
              {
                  "error": (
                      "use GET /obs, POST /plan, POST /done, POST /reset,"
                      " POST /deproject"
                  )
              },
          )
          return
        if not backend._plan_done.wait(timeout=600.0):
          self._send(504, {"error": "plan execution timed out"})
          return
        with backend._lock:
          self._send(200, dict(backend._payload))

    return Handler

  def consume_reset_request(self) -> bool:
    with self._lock:
      requested, self._reset_requested = self._reset_requested, False
      return requested

  def has_pending_plan(self) -> bool:
    with self._lock:
      return self._pending is not None

  def on_reset(self, obs: Mapping[str, Any]) -> None:
    with self._lock:
      self._turn = 0
      self._done = False
      self._pending = None
      self._last_plan = []
      self._steps_log = []
      self._history = []
      self._strategy = ""
      self._last_thought = ""
    self._executor.stopped_on_contact = False
    self._executor.contact_pos = None
    self.grid_z = GRID_Z
    (self._dir / "obs_frames").mkdir(parents=True, exist_ok=True)
    steps_file = self._dir / "steps.jsonl"
    if steps_file.exists():
      steps_file.unlink()
    qpos = _obs(obs, "joint_pos") if "joint_pos" in obs else None
    calibration.generate_card(
        self._emb,
        self._dir / "calibration.png",
        rgb=obs.get("wrist_rgb"),
        qpos=qpos,
        grid_z=self.grid_z,
    )
    arm_qpos = (
        qpos
        if qpos is not None
        else np.asarray(
            self._emb.home_qpos[: self._emb.n_arm], dtype=np.float64
        )
    )
    reset_img = render_wrist_image(
        obs["wrist_rgb"], arm_qpos, self._emb, grid_z=self.grid_z
    )
    self._frame_0 = reset_img
    self._frame_1 = reset_img
    self._frame_2 = reset_img
    self._frame_1_captured = True
    self._gif_frames = [reset_img]
    self._write_turn_files(obs, status="reset", success=False, event="reset")
    self._plan_done.set()

  def plan(self, obs: Mapping[str, Any]) -> Plan | None:
    """Consumes and resolves any pending plan submitted over the HTTP bridge."""
    with self._lock:
      plan, self._pending = self._pending, None
      if plan is not None:
        self._turn += 1
    if plan is not None:
      self._frame_0 = render_wrist_image(
          obs["wrist_rgb"],
          _obs(obs, "joint_pos"),
          self._emb,
          grid_z=self.grid_z,
      )
      self._frame_1 = None
      self._frame_2 = None
      self._frame_1_captured = False
      plan = self._resolve(plan, obs)
      self._last_plan = list(plan)
      for item in plan:
        op = item.get("op", "goto")
        if op == "home":
          self.grid_z = GRID_Z
        elif "grid_z" in item:
          self.grid_z = float(item["grid_z"])
        elif "z_plane" in item:
          self.grid_z = float(item["z_plane"])
    return plan

  def on_step(self, obs: Mapping[str, Any]) -> None:
    """Captures mid-motion / pre-grasp hover frame during plan execution."""
    if self._frame_1_captured:
      return
    cur_seg = self._executor._segments[0] if self._executor._segments else None
    if cur_seg is not None and cur_seg.label.endswith(":descend"):
      self._frame_1 = render_wrist_image(
          obs["wrist_rgb"],
          _obs(obs, "joint_pos"),
          self._emb,
          grid_z=self.grid_z,
      )
      self._frame_1_captured = True
      return
    if self._executor._start is not None and self._executor._duration > 0:
      if self._executor._t >= self._executor._duration * 0.5:
        self._frame_1 = render_wrist_image(
            obs["wrist_rgb"],
            _obs(obs, "joint_pos"),
            self._emb,
            grid_z=self.grid_z,
        )
        if len(self._executor._segments) <= 1:
          self._frame_1_captured = True
          return
    if self._frame_1 is None:
      self._frame_1 = render_wrist_image(
          obs["wrist_rgb"],
          _obs(obs, "joint_pos"),
          self._emb,
          grid_z=self.grid_z,
      )

  def _resolve(self, plan: Plan, obs: Mapping[str, Any]) -> Plan:
    """Casts pixel targets with the joints of the image the agent saw."""
    qpos = self._image_qpos
    return resolve_pixels(
        plan, _obs(obs, "joint_pos") if qpos is None else qpos, self._emb
    )

  def on_plan_done(
      self, obs: Mapping[str, Any], status: str, success: bool
  ) -> None:
    event = "done" if (self._done or status == "episode_end") else "plan"
    self._write_turn_files(obs, status=status, success=success, event=event)
    self._plan_done.set()

  def _write_report(self, success: bool) -> None:
    """Writes a self-contained markdown episode summary to `report.md`."""
    max_t = self._max_turns if self._max_turns > 0 else "unlimited"
    strategy_str = self._strategy or "Autonomous closed-loop execution"
    gif_abs = str(self._dir / "rollout.gif")
    lines = [
        f"# Episode Rollout Trace: {self._emb.display_name}",
        "",
        f"![Rollout GIF]({gif_abs})",
        "",
        f"- **Embodiment**: {self._emb.display_name} (`{self._emb.name}`)",
        f"- **Task**: {self._prompt}",
        f"- **Agent Upfront Plan / Granularity**: {strategy_str}",
        f"- **Turn**: {self._turn} / {max_t}",
        f"- **Status**: {self._executor.status}",
        f"- **Success**: {bool(success)}",
        "",
        "## Turn-by-Turn Thinking Trace & Trajectory",
        "",
    ]
    for entry in self._steps_log:
      turn_idx = entry["turn"]
      img_abs = str(self._dir / entry["image"])
      traj_abs = str(self._dir / f"obs_frames/trajectory_{turn_idx:03d}.png")
      plan_str = json.dumps(entry["plan"])
      thought_str = entry.get("thought") or "(Initial reset observation)"
      lines.extend([
          (
              f"### Turn {turn_idx:03d} (`{entry['event']}`) —"
              f" `status={entry['status']}`"
          ),
          "",
          f"![Turn {turn_idx:03d} Trajectory Strip]({traj_abs})",
          "",
          f"- **Thinking Trace**: {thought_str}",
          f"- **Plan**: `{plan_str}`",
          (
              f"- **State**: `ee_pos={entry['ee_pos']}`,"
              f" `holding={entry['holding']}`,"
              f" `stopped_on_contact={entry['stopped_on_contact']}`,"
              f" `contact_pos={entry['contact_pos']}`"
          ),
          f"- **Snapshot**: [{entry['image']}]({img_abs})",
          "",
      ])
    (self._dir / "report.md").write_text("\n".join(lines) + "\n")

  def _write_turn_files(
      self,
      obs: Mapping[str, Any],
      status: str | None = None,
      success: bool = False,
      event: str | None = None,
  ) -> dict[str, Any]:
    """Writes `wrist_cam.png`, `obs.json`, `obs_frames/`, `steps.jsonl`, and `report.md`."""
    if status is not None:
      self._executor.status = status
    elif self._turn > 0 and self._executor.status in ("idle", "reset"):
      self._executor.status = "done"
    eff_status = self._executor.status
    if event is None:
      if eff_status == "reset" or (self._turn == 0 and not self._last_plan):
        event = "reset"
      elif self._done or eff_status == "episode_end":
        event = "done"
      else:
        event = "plan"

    qpos = _obs(obs, "joint_pos")
    self._image_qpos = qpos
    img = save_wrist_image(
        obs["wrist_rgb"],
        str(self._dir / "wrist_cam.png"),
        qpos,
        self._emb,
        grid_z=self.grid_z,
    )
    img_bytes = (self._dir / "wrist_cam.png").read_bytes()
    (self._dir / f"wrist_cam_{self._turn:02d}.png").write_bytes(img_bytes)
    frames_dir = self._dir / "obs_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rel_frame = f"obs_frames/turn_{self._turn:03d}.png"
    (self._dir / rel_frame).write_bytes(img_bytes)

    self._frame_2 = render_wrist_image(
        obs["wrist_rgb"], qpos, self._emb, grid_z=self.grid_z
    )
    if self._turn > 0 and self._frame_2 is not None:
      self._gif_frames.append(self._frame_2)
    gif_path = self._dir / "rollout.gif"
    if self._gif_frames:
      self._gif_frames[0].save(
          str(gif_path),
          save_all=True,
          append_images=self._gif_frames[1:],
          duration=600,
          loop=0,
      )
    f0 = self._frame_0 if self._frame_0 is not None else self._frame_2
    f1 = self._frame_1 if self._frame_1 is not None else self._frame_2
    f2 = self._frame_2
    strip = create_trajectory_strip([f0, f1, f2])
    strip_path = self._dir / "wrist_cam_trajectory.png"
    strip.save(str(strip_path), optimize=True)
    turn_strip_rel = f"obs_frames/trajectory_{self._turn:03d}.png"
    strip.save(str(self._dir / turn_strip_rel), optimize=True)

    ee, ori = _obs(obs, "ee_pos"), _obs(obs, "ee_ori")
    grip = float(_obs(obs, "gripper_state")[0])
    holding = self._executor.holding(obs)
    yaw_long = embodiment_lib.fold(float(ori[0]) - np.pi / 2)
    az_off = float(ori[2]) if len(ori) > 2 else 0.0
    azimuth = embodiment_lib.wrap_pi(
        self._emb.heading(float(ee[0]), float(ee[1])) + az_off
    )
    roll = embodiment_lib.fold(float(ori[0]) - azimuth)
    stopped_on_contact = bool(self._executor.stopped_on_contact)
    contact_pos = self._executor.contact_pos
    state = {
        "ee_pos": [round(v, 4) for v in ee],
        "yaw_deg": round(float(np.degrees(yaw_long)), 1),
        "jaw_yaw_deg": round(float(np.degrees(ori[0])), 1),
        "roll_deg": round(float(np.degrees(roll)), 1),
        "pitch": round(float(ori[1]), 3),
        "pitch_deg": round(float(np.degrees(ori[1])), 1),
        "azimuth_deg": round(float(np.degrees(azimuth)), 1),
        "gripper": round(grip, 3),
        "holding": holding,
        "stopped_on_contact": stopped_on_contact,
        "contact_pos": contact_pos,
        "grid_z": round(float(self.grid_z), 4),
    }
    self._history.append({"turn": self._turn, "status": eff_status, **state})
    payload = {
        "robot": self._emb.display_name,
        "task": self._prompt,
        "strategy": self._strategy,
        "thought": self._last_thought,
        "turn": self._turn,
        "max_turns": self._max_turns,
        "status": eff_status,
        "success": success,
        **state,
        "joint_pos": [round(v, 3) for v in qpos],
        "wrist_cam_path": img,
        "wrist_cam_trajectory_path": str(strip_path),
        "rollout_gif_path": str(gif_path),
        "report_path": str(self._dir / "report.md"),
        "calibration_image": str(self._dir / "calibration.png"),
        "history": self._history[-20:],
        "workspace": {
            "min": list(self._emb.workspace_min),
            "max": list(self._emb.workspace_max),
        },
    }
    (self._dir / "obs.json").write_text(json.dumps(payload, indent=1))
    step_record = {
        "turn": self._turn,
        "event": event,
        "strategy": self._strategy,
        "thought": self._last_thought,
        "plan": self._last_plan,
        "status": self._executor.status,
        "stopped_on_contact": stopped_on_contact,
        "contact_pos": contact_pos,
        "ee_pos": state["ee_pos"],
        "yaw_deg": state["yaw_deg"],
        "roll_deg": state["roll_deg"],
        "pitch_deg": state["pitch_deg"],
        "azimuth_deg": state["azimuth_deg"],
        "gripper": state["gripper"],
        "holding": state["holding"],
        "image": rel_frame,
        "wrist_cam_trajectory_path": str(strip_path),
    }
    self._steps_log.append(step_record)
    with open(self._dir / "steps.jsonl", "a", encoding="utf-8") as f:
      f.write(json.dumps(step_record) + "\n")
    self._write_report(success)
    with self._lock:
      self._payload = payload
    return payload

  _snapshot = _write_turn_files


class CLIBackend(BridgeBackend):
  """Runs a CLI model (e.g. `gemini -p`) each turn with the prompt + image."""

  def __init__(
      self,
      task_prompt: str,
      bridge_dir: str,
      cli_cmd: str,
      prompt_file: str,
      emb: Embodiment | str,
      max_turns: int = 20,
  ):
    super().__init__(
        task_prompt, bridge_dir, http_port=0, emb=emb, max_turns=max_turns
    )
    self._cli_cmd = cli_cmd
    self._instructions = epath.Path(prompt_file).read_text()

  def plan(self, obs: Mapping[str, Any]) -> Plan | None:
    if self._done or (0 < self._max_turns <= self._turn):
      self._done = True
      return None
    with self._lock:
      payload = dict(self._payload)
    if not payload:
      payload = self._write_turn_files(obs, "reset", False, event="reset")
    if payload.get("success"):
      self._done = True
      return None
    prompt = f"{self._instructions}\n\nOBSERVATION:\n{json.dumps(payload)}"
    out = subprocess.run(
        [*shlex.split(self._cli_cmd), prompt],
        check=True,
        capture_output=True,
        text=True,
        timeout=300.0,
    ).stdout
    self._turn += 1
    raw_plan = parse_plan(out)
    has_done = any(item.get("op") == "done" for item in raw_plan)
    plan = [item for item in raw_plan if item.get("op") != "done"]
    plan = self._resolve(plan, obs)
    self._last_plan = list(plan)
    for item in plan:
      op = item.get("op", "goto")
      if op == "home":
        self.grid_z = GRID_Z
      elif "grid_z" in item:
        self.grid_z = float(item["grid_z"])
      elif "z_plane" in item:
        self.grid_z = float(item["z_plane"])
    if has_done or not plan or (0 < self._max_turns <= self._turn):
      self._done = True
    return plan or None


class AgentPolicy:
  """Polls a backend for plans and executes them at the control rate.

  After `reset` the arm first moves to the embodiment's overview home pose;
  only then is the backend given its first observation, so sim and real
  episodes start from the same view.
  """

  def __init__(
      self,
      backend: AgentBackend,
      emb: Embodiment | str,
      dt: float = CONTROL_DT,
      speed_scale: float = 1.0,
  ):
    self._backend = backend
    self._executor = PlanExecutor(emb, dt, speed_scale=speed_scale)
    if isinstance(backend, BridgeBackend):
      backend._executor = self._executor  # pylint: disable=protected-access
    self._homing = False
    self._reported = True

  @property
  def executor(self) -> PlanExecutor:
    return self._executor

  @property
  def idle(self) -> bool:
    return not self._executor.busy and not self._homing and self._reported

  def reset(self, obs: Mapping[str, Any]) -> None:
    self._executor.sync(obs)
    self._executor.load([{"op": "home"}], obs)
    self._homing = True
    self._reported = True

  def step(self, obs: Mapping[str, Any], success: bool = False) -> torch.Tensor:
    if not self._executor.busy:
      if self._homing:
        self._homing = False
        self._backend.on_reset(obs)
      elif not self._reported:
        self._reported = True
        self._backend.on_plan_done(obs, self._executor.status, success)
      plan = self._backend.plan(obs)
      if plan is not None:
        if plan:
          self._executor.load(plan, obs)
          self._reported = False
        else:
          self._reported = True
          self._backend.on_plan_done(obs, "done", success)
    action = self._executor.step(obs)
    if hasattr(self._backend, "on_step"):
      self._backend.on_step(obs)
    return torch.as_tensor(action).unsqueeze(0)

  def finish(self, obs: Mapping[str, Any], success: bool) -> None:
    """Reports the terminal observation (e.g. task success) to the backend."""
    if not self._reported or success:
      self._reported = True
      self._backend.on_plan_done(obs, "episode_end", success)


class VideoRecorder:
  """Records side-by-side 3rd-person overview + annotated wrist-cam video."""

  def __init__(
      self,
      env: Any,
      path: str,
      emb: Embodiment | str,
      task_name: str = "",
      fps: int = 25,
  ):
    self._env = env
    self._emb = embodiment_lib.get(emb)
    self._path = str(path)
    self._task_name = task_name
    self._fps = fps
    sim = env.device_layer.sim
    self._model, self._data = sim.mj_model, sim.mj_data
    self._model.vis.global_.offwidth = max(
        int(self._model.vis.global_.offwidth), 480
    )
    self._model.vis.global_.offheight = max(
        int(self._model.vis.global_.offheight), 360
    )
    self._renderer = mujoco.Renderer(self._model, height=360, width=480)
    lookat, distance, azimuth, elevation = self._emb.overview_camera
    self._cam = mujoco.MjvCamera()
    self._cam.lookat[:] = lookat
    self._cam.distance = distance
    self._cam.azimuth = azimuth
    self._cam.elevation = elevation
    self._frames: list[bytes] = []

  def record(
      self,
      obs: Mapping[str, Any],
      policy: AgentPolicy,
      step: int,
      success: bool,
  ) -> None:
    self._data.qpos[:] = self._env.device_layer.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(self._model, self._data)
    self._renderer.update_scene(self._data, camera=self._cam)
    overview = Image.fromarray(self._renderer.render())
    wrist = render_wrist_image(
        obs["wrist_rgb"],
        _obs(obs, "joint_pos"),
        self._emb,
        grid_z=policy.executor.grid_z,
    ).resize((480, 360), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (960, 396), (22, 24, 30))
    canvas.paste(overview, (0, 36))
    canvas.paste(wrist, (480, 36))
    draw = ImageDraw.Draw(canvas)
    seg_label = (
        policy.executor._segments[0].label  # pylint: disable=protected-access
        if policy.executor._segments  # pylint: disable=protected-access
        else policy.executor.status
    )
    status_str = "SUCCESS" if success else policy.executor.status
    status_color = (80, 240, 120) if success else (220, 230, 245)
    header = (
        f"{self._emb.display_name}  |  task: {self._task_name}  | "
        f" t={step * CONTROL_DT:4.1f}s (step {step:4d})  |  seg:"
        f" {seg_label:<14s}  |  status: {status_str}"
    )
    draw.text((12, 10), header, fill=status_color)
    draw.text(
        (12, 44),
        "3rd-Person Overview",
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    draw.text(
        (492, 44),
        "Wrist Cam + 2cm FK Grid & Pinch Reticle",
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    draw.line([(480, 36), (480, 396)], fill=(60, 65, 80), width=2)
    self._frames.append(canvas.tobytes())

  def close(self) -> None:
    self._renderer.close()
    if not self._frames:
      return
    out = epath.Path(self._path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if str(out).endswith(".gif"):
      cmd = [
          "ffmpeg",
          "-y",
          "-f",
          "rawvideo",
          "-pix_fmt",
          "rgb24",
          "-s",
          "960x396",
          "-r",
          str(self._fps),
          "-i",
          "-",
          "-vf",
          (
              "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
              "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
          ),
          str(out),
      ]
    else:
      cmd = [
          "ffmpeg",
          "-y",
          "-f",
          "rawvideo",
          "-pix_fmt",
          "rgb24",
          "-s",
          "960x396",
          "-r",
          str(self._fps),
          "-i",
          "-",
          "-c:v",
          "libx264",
          "-pix_fmt",
          "yuv420p",
          "-crf",
          "18",
          str(out),
      ]
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    ) as proc:
      assert proc.stdin is not None
      for frame in self._frames:
        proc.stdin.write(frame)
      for _ in range(int(0.8 * self._fps)):
        proc.stdin.write(self._frames[-1])
      proc.stdin.close()
      stderr = proc.stderr.read() if proc.stderr is not None else b""
      retcode = proc.wait()
      if retcode != 0:
        raise subprocess.CalledProcessError(retcode, cmd, stderr=stderr)
    self._frames.clear()
