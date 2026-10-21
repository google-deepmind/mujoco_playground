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

"""In-context robot calibration card generator for VLM agents.

Generates a visual reference card (`calibration.png`) for an embodiment
(`so101`, `yam`) showing:
1. 2 cm grid labeled with world (X, Y) coordinates in centimetres.
2. Directional arrows:
   - `+X` (away from base, pointing UP in camera frame)
   - `+Y` (world left, pointing LEFT in camera frame)
   - `-Y` (world right, pointing RIGHT in camera frame)
   - `-X` (towards base, pointing DOWN in camera frame)
3. Pinch reticle: cyan crosshair at pinch center, jaw brackets `[ + ]` showing
   jaw closing axis, green line showing object `yaw_deg` axis.
4. Jaw angle reference at `yaw_deg = 0°`, `yaw_deg = 45°`, `yaw_deg = 90°`.
5. Metric scale bar (5 cm bar with exact pixel length).
"""

from collections.abc import Sequence
import functools
import math
from typing import Any

from absl import app
from absl import flags
from etils import epath
import mujoco
from mujoco_playground.tasks.agentic.tabletop_v0 import embodiment as embodiment_lib
import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch

Embodiment = embodiment_lib.Embodiment

OVERLAY_SCALE = 2  # 2x upscaled view for crisp legibility.
GRID_Z = 0.015  # Default height of the table grid in world metres.


class Kinematics:
  """Standalone FK model of an embodiment: wrist camera and pinch center.

  Joint positions are the `joint_pos` observation (arm joints then gripper).
  """

  def __init__(self, emb: Embodiment):
    self.emb = emb
    self.model = emb.spec_fn().compile()
    self.data = mujoco.MjData(self.model)
    self._cam = self.model.camera(emb.wrist_cam).id
    self._site = self.model.site(emb.pinch_site).id
    self._qadr = np.array(
        [self.model.joint(j).qposadr[0] for j in emb.arm_joints]
    )
    w, h = emb.camera_size
    self.image_size = (w, h)
    sensorsize = self.model.cam_sensorsize[self._cam]
    intrinsic = self.model.cam_intrinsic[self._cam]
    if sensorsize[0] > 0 and sensorsize[1] > 0:
      self._fx = float(intrinsic[0] / sensorsize[0] * w)
      self._fy = float(intrinsic[1] / sensorsize[1] * h)
      self._cx = float(w / 2.0 - intrinsic[2] / sensorsize[0] * w)
      self._cy = float(h / 2.0 + intrinsic[3] / sensorsize[1] * h)
    else:
      fovy = np.deg2rad(self.model.cam_fovy[self._cam])
      self._fx = self._fy = float((h / 2.0) / np.tan(fovy / 2.0))
      self._cx = float(w / 2.0)
      self._cy = float(h / 2.0)

  def _set(self, qpos: np.ndarray) -> None:
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    self.data.qpos[self._qadr] = q[: len(self._qadr)]
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_camlight(self.model, self.data)

  def camera_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    self._set(qpos)
    return (
        self.data.cam_xpos[self._cam].copy(),
        self.data.cam_xmat[self._cam].reshape(3, 3).copy(),
    )

  def project(
      self, xyz: Sequence[float] | np.ndarray, qpos: np.ndarray
  ) -> tuple[float, float] | None:
    """Projects a world point into wrist-camera pixels (u_px, v_px)."""
    cam_pos, cam_rot = self.camera_pose(qpos)
    p = cam_rot.T @ (np.asarray(xyz, dtype=np.float64) - cam_pos)
    if p[2] >= -0.015:
      return None
    return (
        self._cx + self._fx * p[0] / -p[2],
        self._cy - self._fy * p[1] / -p[2],
    )

  def deproject(
      self, u: float, v: float, qpos: np.ndarray, z_plane: float
  ) -> tuple[float, float]:
    """Casts a normalized or pixel (u, v) onto the plane z=z_plane."""
    w, h = self.image_size
    if u > 1.0 or v > 1.0:  # Pixels of the raw or the 2x overlay image.
      s = OVERLAY_SCALE if (u > w or v > h) else 1
      u, v = u / (w * s), v / (h * s)
    cam_pos, cam_rot = self.camera_pose(qpos)
    d = cam_rot @ np.array(
        [(u * w - self._cx) / self._fx, (self._cy - v * h) / self._fy, -1.0]
    )
    t = (z_plane - cam_pos[2]) / d[2] if abs(d[2]) > 1e-6 else 0.0
    t = float(np.clip(t, 0.01, 1.0))
    return float(cam_pos[0] + t * d[0]), float(cam_pos[1] + t * d[1])

  def pinch_center(self, qpos: np.ndarray) -> np.ndarray:
    self._set(qpos)
    return self.data.site_xpos[self._site].copy()

  def pinch_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (site_xpos, site_xmat 3x3) of the pinch site."""
    self._set(qpos)
    return (
        self.data.site_xpos[self._site].copy(),
        self.data.site_xmat[self._site].reshape(3, 3).copy(),
    )


@functools.lru_cache(maxsize=None)
def kinematics(robot: str | Embodiment) -> Kinematics:
  """Returns a cached `Kinematics` for an embodiment or its name."""
  return Kinematics(embodiment_lib.get(robot))


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[int, int, int] = (255, 255, 255),
    width: int = 2,
    head_len: float = 9.0,
    head_width: float = 5.0,
) -> None:
  """Draws a line with a filled arrowhead from p0 to p1."""
  x0, y0 = p0
  x1, y1 = p1
  draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
  dx, dy = x1 - x0, y1 - y0
  norm = math.hypot(dx, dy)
  if norm < 1e-3:
    return
  ux, uy = dx / norm, dy / norm
  vx, vy = -uy, ux
  bx, by = x1 - head_len * ux, y1 - head_len * uy
  left = (bx + head_width * vx, by + head_width * vy)
  right = (bx - head_width * vx, by - head_width * vy)
  draw.polygon([(x1, y1), left, right], fill=color)


def _draw_rotated_bracket(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    yaw_deg: float,
    color_jaws: tuple[int, int, int] = (0, 255, 255),
    color_axis: tuple[int, int, int] = (80, 255, 140),
    scale: float = 1.0,
) -> None:
  """Draws a 2D miniature jaw bracket & object axis diagram at `yaw_deg`."""
  xc, yc = center
  rad = math.radians(yaw_deg)
  # In image coordinates: +X (0 deg) points UP (0, -1), +Y (90 deg) points LEFT (-1, 0)
  tx = -math.sin(rad)
  ty = -math.cos(rad)
  # Closing axis is perpendicular to object long axis
  cx = -ty
  cy = tx

  # Draw green object axis line
  ax_len = 20.0 * scale
  draw.line(
      [
          (xc - ax_len * tx, yc - ax_len * ty),
          (xc + ax_len * tx, yc + ax_len * ty),
      ],
      fill=color_axis,
      width=max(1, int(2 * scale)),
  )

  # Draw cyan jaw brackets on either side along closing axis
  jaw_dist = 14.0 * scale
  pad_len = 8.0 * scale
  lip_len = 4.0 * scale

  for sign in (-1.0, 1.0):
    jx = xc + sign * jaw_dist * cx
    jy = yc + sign * jaw_dist * cy
    # Back pad
    b0 = (jx - pad_len * tx, jy - pad_len * ty)
    b1 = (jx + pad_len * tx, jy + pad_len * ty)
    # Lips pointing inward toward center
    l0 = (b0[0] - sign * lip_len * cx, b0[1] - sign * lip_len * cy)
    l1 = (b1[0] - sign * lip_len * cx, b1[1] - sign * lip_len * cy)
    draw.line([l0, b0, b1, l1], fill=color_jaws, width=max(1, int(2 * scale)))

  # Center crosshair
  draw.line(
      [(xc - 3 * scale, yc), (xc + 3 * scale, yc)],
      fill=color_jaws,
      width=1,
  )
  draw.line(
      [(xc, yc - 3 * scale), (xc, yc + 3 * scale)],
      fill=color_jaws,
      width=1,
  )


def render_calibration_card(
    emb: Embodiment | str,
    rgb: torch.Tensor | np.ndarray | None = None,
    qpos: np.ndarray | None = None,
    grid_z: float = GRID_Z,
) -> Image.Image:
  """Renders a 2x visual calibration reference card for the given embodiment.

  Shows:
  - 2 cm grid labeled with world (X, Y) coordinates in centimetres.
  - Directional arrows (+X away, +Y left, -Y right, -X towards base).
  - Pinch reticle: cyan crosshair, jaw brackets [ + ], green yaw_deg axis.
  - Jaw angle reference at yaw_deg = 0°, 45°, 90°.
  - Metric scale bar (5 cm bar with pixel length).

  Args:
    emb: Embodiment instance or registry key ("so101", "yam").
    rgb: Optional wrist camera RGB image (if None, renders on dark slate
      canvas).
    qpos: Optional arm joint positions (defaults to emb.home_qpos).
    grid_z: Height of the table grid in world metres.

  Returns:
    Annotated PIL Image (2x resolution of embodiment.camera_size).
  """
  emb = embodiment_lib.get(emb)
  kin = kinematics(emb)
  w, h = kin.image_size
  s = OVERLAY_SCALE
  img_w, img_h = w * s, h * s

  if qpos is None:
    qpos = np.asarray(emb.home_qpos, dtype=np.float64)

  if rgb is not None:
    arr = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
    arr = np.clip(np.asarray(arr).reshape(-1, *arr.shape[-3:])[0], 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((img_w, img_h), Image.Resampling.BILINEAR)
  else:
    # Clean dark technical slate background
    img = Image.new("RGB", (img_w, img_h), color=(22, 27, 34))

  draw = ImageDraw.Draw(img)

  def _px(p: Sequence[float] | np.ndarray) -> tuple[float, float] | None:
    uv = kin.project(p, qpos)
    return None if uv is None else (uv[0] * s, uv[1] * s)

  def _line(
      points: Sequence[Sequence[float] | np.ndarray],
      color=(170, 210, 255),
      width=1,
  ):
    pts = [_px(p) for p in points]
    pts = [
        p
        for p in pts
        if p is not None and -50 < p[0] < img_w + 50 and -50 < p[1] < img_h + 50
    ]
    if len(pts) >= 2:
      draw.line(pts, fill=color, width=width)

  # 1. 2 cm Grid over the workspace footprint at height grid_z.
  x0, y0 = (np.floor(100 * v / 2) * 0.02 for v in emb.workspace_min[:2])
  x1, y1 = (np.ceil(100 * v / 2) * 0.02 for v in emb.workspace_max[:2])
  xs = np.arange(x0, x1 + 1e-3, 0.02)
  ys = np.arange(y0, y1 + 1e-3, 0.02)

  for x in xs:
    _line(
        [(x, y, grid_z) for y in np.linspace(y0, y1, 2 * len(ys))],
        color=(100, 140, 190),
        width=1,
    )
  for y in ys:
    _line(
        [(x, y, grid_z) for x in np.linspace(x0, x1, 2 * len(xs))],
        color=(100, 140, 190),
        width=1,
    )

  # Label coordinates every 4 cm in yellow
  for x in xs:
    for y in ys:
      if round(100 * x) % 4 or round(100 * y) % 4:
        continue
      uv = _px((x, y, grid_z))
      if (
          uv is not None
          and 10 < uv[0] < img_w - 50
          and 10 < uv[1] < img_h - 100
      ):
        text = f"{100 * x:.0f},{100 * y:+.0f}"
        draw.text(
            (uv[0] + 3, uv[1] - 6),
            text,
            fill=(255, 240, 120),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

  # 2. Table-Projected Directional Axis Triad (Spawn Center)
  cx = float((emb.spawn_x[0] + emb.spawn_x[1]) / 2.0)
  cy = float((emb.spawn_y[0] + emb.spawn_y[1]) / 2.0)
  c_uv = _px((cx, cy, grid_z))
  px_p = _px((cx + 0.05, cy, grid_z))  # +X
  px_m = _px((cx - 0.04, cy, grid_z))  # -X
  py_p = _px((cx, cy + 0.05, grid_z))  # +Y
  py_m = _px((cx, cy - 0.05, grid_z))  # -Y

  if c_uv and px_p:
    _draw_arrow(draw, c_uv, px_p, color=(255, 130, 70), width=3, head_len=10)
    draw.text(
        (px_p[0] - 12, px_p[1] - 16),
        "+X",
        fill=(255, 140, 80),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
  if c_uv and px_m:
    _draw_arrow(draw, c_uv, px_m, color=(255, 130, 70), width=2, head_len=8)
    draw.text(
        (px_m[0] - 10, px_m[1] + 4),
        "-X",
        fill=(255, 140, 80),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
  if c_uv and py_p:
    _draw_arrow(draw, c_uv, py_p, color=(80, 210, 255), width=3, head_len=10)
    draw.text(
        (py_p[0] - 22, py_p[1] - 8),
        "+Y",
        fill=(80, 210, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
  if c_uv and py_m:
    _draw_arrow(draw, c_uv, py_m, color=(80, 210, 255), width=3, head_len=10)
    draw.text(
        (py_m[0] + 6, py_m[1] - 8),
        "-Y",
        fill=(80, 210, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

  # 3. Pinch Reticle: Cyan crosshair, 3D jaw brackets, green object axis
  pinch, rot = kin.pinch_pose(qpos)
  closing, tangent = rot[:, 0], rot[:, 1]
  for sign in (-1.0, 1.0):
    jaw_c = pinch + sign * 0.022 * closing
    b0 = jaw_c - 0.010 * tangent
    b1 = jaw_c + 0.010 * tangent
    _line(
        [b0 - sign * 0.005 * closing, b0, b1, b1 - sign * 0.005 * closing],
        color=(0, 255, 255),
        width=2,
    )
  _line(
      [pinch - 0.022 * tangent, pinch + 0.022 * tangent],
      color=(80, 255, 140),
      width=2,
  )
  pinch_uv = _px(pinch)
  if pinch_uv is not None:
    px, py = pinch_uv
    draw.line([(px - 10, py), (px + 10, py)], fill=(0, 255, 255), width=2)
    draw.line([(px, py - 10), (px, py + 10)], fill=(0, 255, 255), width=2)
    draw.text(
        (px + 14, py - 8),
        "Pinch [ + ]",
        fill=(0, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

  # 4. Top-Left HUD: Directional Reference Card
  card_x0, card_y0, card_x1, card_y1 = 8, 8, 220, 126
  draw.rectangle(
      [(card_x0, card_y0), (card_x1, card_y1)],
      fill=(15, 20, 28),
      outline=(70, 90, 120),
      width=1,
  )
  draw.text(
      (card_x0 + 8, card_y0 + 6),
      "DIRECTIONAL ARROWS",
      fill=(255, 215, 0),
      stroke_width=1,
      stroke_fill=(0, 0, 0),
  )

  ox, oy = card_x0 + 106, card_y0 + 68
  # +X arrow pointing UP
  _draw_arrow(draw, (ox, oy), (ox, oy - 32), color=(255, 130, 70), width=2)
  draw.text(
      (ox - 38, oy - 46),
      "+X (away / UP)",
      fill=(255, 140, 80),
      stroke_width=1,
      stroke_fill=(0, 0, 0),
  )
  # -X arrow pointing DOWN
  _draw_arrow(draw, (ox, oy), (ox, oy + 32), color=(255, 130, 70), width=2)
  draw.text(
      (ox - 42, oy + 34),
      "-X (base / DOWN)",
      fill=(255, 140, 80),
      stroke_width=1,
      stroke_fill=(0, 0, 0),
  )
  # +Y arrow pointing LEFT
  _draw_arrow(draw, (ox, oy), (ox - 38, oy), color=(80, 210, 255), width=2)
  draw.text(
      (card_x0 + 6, oy - 7),
      "+Y (L)",
      fill=(80, 210, 255),
      stroke_width=1,
      stroke_fill=(0, 0, 0),
  )
  # -Y arrow pointing RIGHT
  _draw_arrow(draw, (ox, oy), (ox + 38, oy), color=(80, 210, 255), width=2)
  draw.text(
      (ox + 42, oy - 7),
      "-Y (R)",
      fill=(80, 210, 255),
      stroke_width=1,
      stroke_fill=(0, 0, 0),
  )

  # 5. Top-Right HUD: Metric Scale Bar (5 cm)
  sbar_x0, sbar_y0, sbar_x1, sbar_y1 = img_w - 230, 8, img_w - 8, 92
  draw.rectangle(
      [(sbar_x0, sbar_y0), (sbar_x1, sbar_y1)],
      fill=(15, 20, 28),
      outline=(70, 90, 120),
      width=1,
  )
  draw.text(
      (sbar_x0 + 8, sbar_y0 + 6),
      "METRIC SCALE (5 cm)",
      fill=(255, 215, 0),
      stroke_width=1,
      stroke_fill=(0, 0, 0),
  )

  # Compute 5 cm projected length at table center
  p_m25 = _px((cx, cy - 0.025, grid_z))
  p_p25 = _px((cx, cy + 0.025, grid_z))
  if p_m25 and p_p25:
    bar_px = abs(p_p25[0] - p_m25[0])
  else:
    bar_px = 75.0

  bx0 = sbar_x0 + 16
  by0 = sbar_y0 + 44
  bx1 = bx0 + bar_px

  # Main horizontal bar
  draw.line([(bx0, by0), (bx1, by0)], fill=(255, 255, 255), width=3)
  # End ticks (0 cm and 5 cm)
  draw.line([(bx0, by0 - 7), (bx0, by0 + 7)], fill=(255, 255, 255), width=2)
  draw.line([(bx1, by0 - 7), (bx1, by0 + 7)], fill=(255, 255, 255), width=2)
  # Intermediate ticks (1..4 cm)
  for i in range(1, 5):
    tx = bx0 + i * (bar_px / 5.0)
    draw.line([(tx, by0 - 4), (tx, by0 + 4)], fill=(200, 200, 200), width=1)

  draw.text((bx0 - 4, by0 - 18), "0cm", fill=(200, 200, 200))
  draw.text((bx1 - 12, by0 - 18), "5cm", fill=(200, 200, 200))
  draw.text(
      (sbar_x0 + 14, by0 + 12),
      f"5.0 cm = {bar_px:.0f} px ({bar_px / 5.0:.1f} px/cm)",
      fill=(255, 240, 120),
  )

  # 6. Bottom HUD: Jaw Angle Reference (yaw_deg = 0°, 45°, 90°)
  jaw_x0, jaw_y0, jaw_x1, jaw_y1 = 8, img_h - 96, img_w - 8, img_h - 8
  draw.rectangle(
      [(jaw_x0, jaw_y0), (jaw_x1, jaw_y1)],
      fill=(15, 20, 28),
      outline=(70, 90, 120),
      width=1,
  )
  draw.text(
      (jaw_x0 + 10, jaw_y0 + 5),
      f"JAW ANGLE REFERENCE — {emb.display_name}  "
      "(Green: object long axis; Cyan [ ]: jaw closing axis)",
      fill=(255, 215, 0),
      stroke_width=1,
      stroke_fill=(0, 0, 0),
  )

  # Sub-panels for 0°, 45°, 90°
  angles = [
      (
          0.0,
          "yaw_deg = 0°",
          "Object || X (vertical in cam)",
          "Jaws close Y (L-R)",
      ),
      (45.0, "yaw_deg = 45°", "Object diagonal (NE-SW)", "Jaws close diagonal"),
      (
          90.0,
          "yaw_deg = 90°",
          "Object || Y (horizontal in cam)",
          "Jaws close X (F-B)",
      ),
  ]
  col_w = (jaw_x1 - jaw_x0) / 3.0
  for i, (ang, title, desc1, desc2) in enumerate(angles):
    sub_cx = jaw_x0 + i * col_w + 35
    sub_cy = jaw_y0 + 52
    _draw_rotated_bracket(draw, (sub_cx, sub_cy), yaw_deg=ang, scale=1.0)
    text_x = sub_cx + 32
    draw.text(
        (text_x, sub_cy - 22),
        title,
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    draw.text((text_x, sub_cy - 8), desc1, fill=(180, 220, 180))
    draw.text((text_x, sub_cy + 6), desc2, fill=(140, 210, 240))

  return img


def generate_card(
    emb: Embodiment | str,
    output_path: str | epath.Path,
    rgb: torch.Tensor | np.ndarray | None = None,
    qpos: np.ndarray | None = None,
    grid_z: float = GRID_Z,
) -> str:
  """Renders and saves calibration.png to `output_path`.

  Args:
    emb: Embodiment instance or registry key ("so101", "yam").
    output_path: Destination file path for calibration.png.
    rgb: Optional wrist camera RGB image.
    qpos: Optional arm joint positions.
    grid_z: Height of the table grid in world metres.

  Returns:
    The canonical string path of the saved calibration.png.
  """
  img = render_calibration_card(emb, rgb=rgb, qpos=qpos, grid_z=grid_z)
  out = epath.Path(output_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  img.save(str(out), optimize=True)
  return str(out)


generate_calibration_card = generate_card


def main(argv: Sequence[str]) -> None:
  del argv
  path = generate_card(
      emb=flags.FLAGS.robot,
      output_path=flags.FLAGS.output,
      grid_z=flags.FLAGS.grid_z,
  )
  print(f"Calibration card generated at {path}")


if __name__ == "__main__":
  flags.DEFINE_string("robot", "so101", "Embodiment name (so101, yam).")
  flags.DEFINE_string(
      "output", "/tmp/calibration.png", "Output path for calibration.png."
  )
  flags.DEFINE_float("grid_z", GRID_Z, "Grid height in world metres.")
  app.run(main)
