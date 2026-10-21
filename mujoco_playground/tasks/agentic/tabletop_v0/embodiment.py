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

"""Embodiments: everything robot-specific the agentic stack needs to know.

The IK controller, plan executor, wrist-camera geometry, observations, reset,
and environment config are all written against the `Embodiment` interface, so
adding an arm means writing a scene function and one registry entry here.

Tool frame convention: every embodiment exposes a pinch site at the midpoint
of the jaw pads whose z-axis points along the fingers (approach) and whose
x-axis points along the jaw closing direction. Orientation commands and
observations are two robot-independent angles of that frame:

  pitch: tilt of the approach axis below horizontal; pi/2 points straight
    down at the table, smaller values tilt the fingers outward.
  yaw: world yaw of the jaw closing axis, modulo pi (the jaws are symmetric).
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from typing import Any

import mujoco
from mujoco_playground.robots.so101.real import constant as so101_real
from mujoco_playground.robots.so101.sim import device as so101_sim_device
from mujoco_playground.robots.so101.sim import scene as so101_scene
from mujoco_playground.robots.yam.real import constant as yam_real
from mujoco_playground.robots.yam.sim import constant as yam_sim
from mujoco_playground.robots.yam.sim import device as yam_sim_device
from mujoco_playground.robots.yam.sim import scene as yam_scene
import numpy as np

PITCH_DOWN = np.pi / 2
ATTACHMENT_PREFIX = "robot/"

ROBOT_STATE = "robot_state"
QPOS_KEY = f"{ROBOT_STATE}/joint_pos"
EE_XPOS_KEY = "ee/xpos"
EE_XMAT_KEY = "ee/xmat"


def fold(angle: float) -> float:
  """Wraps an angle into [-pi/2, pi/2): jaw directions are defined mod pi."""
  return float((angle + np.pi / 2) % np.pi - np.pi / 2)


def wrap_pi(angle: float) -> float:
  """Wraps an angle into [-pi, pi)."""
  return float((angle + np.pi) % (2 * np.pi) - np.pi)


def tool_rotation(heading: float, rel_yaw: float, pitch: float) -> np.ndarray:
  """Pinch-site rotation (columns = site x, y, z axes) in world coordinates.

  Args:
    heading: Azimuth of the approach direction in world XY (defaults to radial).
    rel_yaw: Jaw closing roll around the approach axis (`0` = in vertical
      approach plane, `pi/2` = horizontal/tangential).
    pitch: Tilt of the approach axis below horizontal.

  Returns:
    3x3 rotation matrix whose columns are `[closing, normal, approach]`.
  """
  radial = np.array([np.cos(heading), np.sin(heading), 0.0])
  tangential = np.array([-np.sin(heading), np.cos(heading), 0.0])
  up = np.array([0.0, 0.0, 1.0])
  approach = np.cos(pitch) * radial - np.sin(pitch) * up
  ortho = np.sin(pitch) * radial + np.cos(pitch) * up
  closing = np.cos(rel_yaw) * ortho + np.sin(rel_yaw) * tangential
  return np.stack([closing, np.cross(approach, closing), approach], axis=1)


@dataclasses.dataclass(frozen=True)
class Embodiment:
  """Robot-specific constants of a single arm with a parallel gripper.

  Attributes:
    name: Registry key, e.g. "so101".
    display_name: Human-readable name for logs and videos.
    spec_fn: Returns the standalone robot `MjSpec` with the pinch site and the
      wrist camera; used for the scene and for FK/IK alike (sim and real).
    sim_device_cls: Simulation follower device class (`SO101Follower` or
      `YamFollower`).
    arm_joints: Arm joint names, in actuator order.
    gripper_joint: The single logical gripper channel driven by the follower.
    mj_gripper_joints: Underlying MJCF joint names for the gripper fingers.
    mj_gripper_range: Physical MJCF joint range `(closed, open)` corresponding
      to `gripper_obs_range`.
    gripper_cmd_range: Gripper joint targets (closed, open) for normalized
      commands 0 and 1, in the follower's gripper units.
    gripper_obs_range: Gripper joint positions (closed, open) that normalize the
      observed follower opening to [0, 1].
    gripper_released_norm: Observed normalized opening above which the jaws are
      considered to have let go of an object.
    pinch_site: Pinch-center site name (see the module docstring).
    jaw_tip_below_pinch: How far the fingertips reach past the pinch center.
    home_qpos: Reset joint positions in follower units, arm joints followed by
      the gripper.
    rest_qpos: Safe rest/shutdown joint positions in follower units.
    home_pose: Overview pose (x, y, z, pitch) that frames the table in the wrist
      camera; the agent's first observation is taken from here.
    base_xy: World XY of the vertical base (pan) axis. Radial directions, reach
      limits, and relative wrist angles are measured from it.
    wrist_yaw_offset: Jaw closing direction relative to the radial direction
      when the wrist roll joint is centered. Relative yaw targets are folded
      into [offset - pi/2, offset + pi/2] to stay inside the wrist range.
    workspace_min: Lower corner of the pinch-center safety box (world, m).
    workspace_max: Upper corner of the pinch-center safety box (world, m).
    spawn_x: Table X range in which objects are scattered.
    spawn_y: Table Y range in which objects are scattered.
    max_reach: Objects spawn within this distance of `base_xy`.
    topdown_reach: Beyond this distance from `base_xy`, grasps tilt the approach
      outward (lower pitch) to stay reachable.
    min_pitch: Lowest pitch used by the reach-dependent tilt.
    tracking_feedback_gain: Joint-error pre-compensation for position servos
      that droop under gravity (0 disables it).
    overview_camera: Third-person video camera (lookat xyz, distance, azimuth,
      elevation).
    approach_speed: Pinch speed (m/s) for the final vertical descent of a grasp
      or place. Arms whose position servos lag under fast joint motion (low
      gains) need a slower approach to keep the descent straight.
    wrist_cam: Wrist camera name in the model.
    camera_size: Wrist camera image (width, height).
  """

  name: str
  display_name: str
  spec_fn: Callable[[], mujoco.MjSpec]
  sim_device_cls: type[Any]
  arm_joints: tuple[str, ...]
  gripper_joint: str
  mj_gripper_joints: tuple[str, ...]
  mj_gripper_range: tuple[float, float]
  gripper_cmd_range: tuple[float, float]
  gripper_obs_range: tuple[float, float]
  gripper_released_norm: float
  pinch_site: str
  jaw_tip_below_pinch: float
  home_qpos: tuple[float, ...]
  rest_qpos: tuple[float, ...]
  home_pose: tuple[float, float, float, float]
  base_xy: tuple[float, float]
  wrist_yaw_offset: float
  workspace_min: tuple[float, float, float]
  workspace_max: tuple[float, float, float]
  spawn_x: tuple[float, float]
  spawn_y: tuple[float, float]
  max_reach: float
  topdown_reach: float
  min_pitch: float
  tracking_feedback_gain: float
  overview_camera: tuple[tuple[float, float, float], float, float, float]
  approach_speed: float = 0.25
  wrist_cam: str = "wrist_cam"
  camera_size: tuple[int, int] = (320, 240)

  @property
  def joints(self) -> tuple[str, ...]:
    """Follower joint channels: arm joints followed by the gripper channel."""
    return (*self.arm_joints, self.gripper_joint)

  @property
  def mj_joints(self) -> tuple[str, ...]:
    """MJCF joint names: arm joints followed by the MJCF gripper joint(s)."""
    return (*self.arm_joints, *self.mj_gripper_joints)

  @property
  def joint_position_keys(self) -> tuple[str, ...]:
    """Follower measurement keys for `self.joints`."""
    return tuple(f"follower/joint_position/{j}" for j in self.joints)

  @property
  def goal_position_keys(self) -> tuple[str, ...]:
    """Follower command keys for `self.joints`."""
    return tuple(f"follower/goal_position/{j}" for j in self.joints)

  def commanded_opening(self, gripper: float) -> float:
    """Observed normalized opening that a free gripper settles at for `gripper`.

    Commands and observations are normalized over different follower ranges
    (`gripper_cmd_range` vs `gripper_obs_range`); this maps a command into the
    observation scale so the two can be compared directly.

    Args:
      gripper: Normalized gripper command in [0 (closed), 1 (open)].

    Returns:
      The matching normalized observed opening in [0, 1].
    """
    lo, hi = self.gripper_cmd_range
    obs_lo, obs_hi = self.gripper_obs_range
    target = lo + float(np.clip(gripper, 0.0, 1.0)) * (hi - lo)
    return float(np.clip((target - obs_lo) / (obs_hi - obs_lo), 0.0, 1.0))

  def follower_to_mj_qpos(
      self, qpos: Sequence[float] | np.ndarray
  ) -> np.ndarray:
    """Converts follower joint positions (arm + gripper) to MJCF `mj_joints`."""
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    obs_lo, obs_hi = self.gripper_obs_range
    mj_lo, mj_hi = self.mj_gripper_range
    frac = float(
        np.clip((q[self.n_arm] - obs_lo) / (obs_hi - obs_lo), 0.0, 1.0)
    )
    g_mj = mj_lo + frac * (mj_hi - mj_lo)
    return np.array(
        [*q[: self.n_arm], *([g_mj] * len(self.mj_gripper_joints))],
        dtype=np.float64,
    )

  @property
  def n_arm(self) -> int:
    """Number of actuated arm joints (excluding the gripper)."""
    return len(self.arm_joints)

  @property
  def ee_site(self) -> str:
    """Pinch site name in the scene (with the attachment prefix)."""
    return f"{ATTACHMENT_PREFIX}{self.pinch_site}"

  @property
  def wrist_cam_fullname(self) -> str:
    """Full scene name of the wrist camera."""
    return f"{ATTACHMENT_PREFIX}{self.wrist_cam}"

  @property
  def wrist_rgb_key(self) -> str:
    """Feature key for the rendered wrist RGB image."""
    return f"{self.wrist_cam_fullname}_rgb"

  def heading(self, x: float, y: float) -> float:
    """Yaw of the radial direction from the base axis to (x, y)."""
    return float(np.arctan2(y - self.base_xy[1], x - self.base_xy[0]))

  def reach(self, x: float, y: float) -> float:
    return float(np.hypot(x - self.base_xy[0], y - self.base_xy[1]))

  def relative_yaw(
      self, yaw: float, x: float, y: float, azimuth: float | None = None
  ) -> float:
    """Jaw yaw relative to the approach azimuth, folded into the wrist band."""
    h = self.heading(x, y) if azimuth is None else float(azimuth)
    off = self.wrist_yaw_offset
    return off + fold(yaw - h - off)

  def grasp_pitch(self, x: float, y: float) -> float:
    """Top-down pitch, tilted outward beyond `topdown_reach`."""
    excess = self.reach(x, y) - self.topdown_reach
    return float(np.clip(PITCH_DOWN - 5.0 * excess, self.min_pitch, PITCH_DOWN))

  def home(self) -> np.ndarray:
    """Overview pose [x, y, z, yaw, pitch] with the wrist roll centered."""
    x, y, z, pitch = self.home_pose
    yaw = fold(self.heading(x, y) + self.wrist_yaw_offset)
    return np.array([x, y, z, yaw, pitch], dtype=np.float64)

  def approach_vector(
      self, x: float, y: float, pitch: float, azimuth: float | None = None
  ) -> np.ndarray:
    """Unit 3D finger approach direction (pinch site +z axis) in world frame."""
    h = self.heading(x, y) if azimuth is None else float(azimuth)
    return np.array([
        np.cos(pitch) * np.cos(h),
        np.cos(pitch) * np.sin(h),
        -np.sin(pitch),
    ])

  def tool_rotation(
      self,
      x: float,
      y: float,
      yaw: float,
      pitch: float,
      azimuth: float | None = None,
  ) -> np.ndarray:
    """Pinch-site rotation matrix (columns = site axes) for a 6-DoF pose."""
    h = self.heading(x, y) if azimuth is None else float(azimuth)
    return tool_rotation(h, self.relative_yaw(yaw, x, y, h), pitch)

  def orientation(
      self, pos: np.ndarray, rot: np.ndarray
  ) -> tuple[float, float, float]:
    """Inverse of `tool_rotation`: `(yaw mod pi, pitch, azimuth)` of a pose."""
    h0 = self.heading(float(pos[0]), float(pos[1]))
    closing, approach = rot[:, 0], rot[:, 2]
    horiz = float(np.hypot(approach[0], approach[1]))
    if self.n_arm >= 6 and horiz > 0.05:
      azimuth = float(np.arctan2(approach[1], approach[0]))
      pitch = float(np.arctan2(-approach[2], horiz))
    else:
      azimuth = h0
      radial0 = np.array([np.cos(h0), np.sin(h0), 0.0])
      pitch = float(np.arctan2(-approach[2], approach @ radial0))
    radial = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    tangential = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    ortho = np.sin(pitch) * radial + np.array([0.0, 0.0, np.cos(pitch)])
    r = float(np.arctan2(closing @ tangential, closing @ ortho))
    return fold(azimuth + r), pitch, azimuth

  def yaw_pitch(self, pos: np.ndarray, rot: np.ndarray) -> tuple[float, float]:
    """Inverse of `tool_rotation`: `(yaw mod pi, pitch)` of a pinch-site pose."""
    yaw, pitch, _ = self.orientation(pos, rot)
    return yaw, pitch


def _so101_spec() -> mujoco.MjSpec:
  """SO101 with `gripperframe` at the pad pinch center, z along the fingers.

  The real arm's state device computes `ee/xpos` and `ee/xmat` from this same
  site, so sim and hardware report the identical tool frame.
  """
  spec = so101_scene.so101()
  site = spec.site("gripperframe")
  site.pos = [0.008, 0.0, -0.085]
  site.quat = [0.0, 1.0, 0.0, 0.0]  # x: jaw closing, z: approach.
  return spec


def _yam_spec() -> mujoco.MjSpec:
  """YAM with `grasp_site` aligned to the agentic tool frame (x: closing, z: approach)."""
  spec = yam_scene.yam()
  # `arm` has its origin 1 cm above the base plate bottom.
  spec.worldbody.first_body().pos = [0.0, 0.0, 0.01]
  site = spec.site(yam_sim.EE_SITE_NAME)
  site.quat = [1.0, 0.0, 0.0, 0.0]  # x: jaw closing, z: approach.
  # Menagerie's D405 sensor is 16:9 (1280x720); crop sensor width to 4:3 so
  # rendering at 320x240 has square pixels matching the real 4:3 ISP stream.
  cam = spec.camera(yam_sim.WRIST_CAM_NAME)
  cam.sensor_size[0] = cam.sensor_size[1] * (320.0 / 240.0)
  while spec.keys:
    spec.delete(spec.keys[0])
  return spec


SO101 = Embodiment(
    name="so101",
    display_name="SO-101",
    spec_fn=_so101_spec,
    sim_device_cls=so101_sim_device.SO101Follower,
    arm_joints=tuple(so101_real.JOINTS[:5]),
    gripper_joint=so101_real.JOINTS[5],
    mj_gripper_joints=(so101_real.JOINTS[5],),
    mj_gripper_range=tuple(so101_real.GRIPPER_RAD_RANGE),
    gripper_cmd_range=(
        so101_real.gripper_rad_to_pct(-0.10),
        so101_real.gripper_rad_to_pct(1.40),
    ),
    gripper_obs_range=(so101_real.GRIPPER_MIN, so101_real.GRIPPER_MAX),
    gripper_released_norm=0.35,
    pinch_site="gripperframe",
    jaw_tip_below_pinch=0.020,
    home_qpos=(
        *so101_real.HOVER_POSE_OPEN_GRIPPER[:5],
        so101_real.gripper_rad_to_pct(so101_real.HOVER_POSE_OPEN_GRIPPER[5]),
    ),
    rest_qpos=tuple(so101_real.FOLLOWER_HOME),
    home_pose=(0.165, 0.05, 0.175, 1.05),
    base_xy=(0.04, 0.0),
    wrist_yaw_offset=0.0,
    workspace_min=(0.14, -0.18, 0.020),
    workspace_max=(0.34, 0.18, 0.24),
    spawn_x=(0.16, 0.27),
    spawn_y=(-0.13, 0.13),
    max_reach=0.24,
    topdown_reach=0.22,
    min_pitch=1.15,
    # Cancels steady-state droop of the STS3215 position servos.
    tracking_feedback_gain=0.55,
    overview_camera=((0.23, 0.0, 0.05), 0.46, 138.0, -26.0),
    approach_speed=0.14,
)

YAM = Embodiment(
    name="yam",
    display_name="YAM",
    spec_fn=_yam_spec,
    sim_device_cls=yam_sim_device.YamFollower,
    arm_joints=yam_real.ARM_JOINTS,
    gripper_joint=yam_real.GRIPPER_JOINT,
    mj_gripper_joints=yam_sim.FINGER_JOINTS,
    mj_gripper_range=yam_sim.GRIPPER_M_RANGE,
    gripper_cmd_range=(yam_real.GRIPPER_MIN, yam_real.GRIPPER_MAX),
    gripper_obs_range=(yam_real.GRIPPER_MIN, yam_real.GRIPPER_MAX),
    gripper_released_norm=0.6,
    pinch_site=yam_sim.EE_SITE_NAME,
    jaw_tip_below_pinch=0.010,
    # IK solution of `home_pose` (see embodiment_test).
    home_qpos=(0.0, 1.012, 1.498, -1.487, 0.0, 0.0, yam_real.GRIPPER_OPEN),
    rest_qpos=tuple(yam_real.FOLLOWER_HOME),
    home_pose=(0.20, 0.0, 0.30, 1.0),
    base_xy=(0.0, 0.0),
    # With joint6 centered the YAM jaws close tangentially.
    wrist_yaw_offset=np.pi / 2,
    workspace_min=(0.16, -0.22, 0.010),
    workspace_max=(0.50, 0.22, 0.40),
    spawn_x=(0.24, 0.40),
    spawn_y=(-0.14, 0.14),
    max_reach=0.44,
    topdown_reach=0.45,
    min_pitch=1.15,
    # Cancels steady-state droop of the low-gain DM4310 joint PD loop.
    tracking_feedback_gain=0.75,
    overview_camera=((0.30, 0.0, 0.08), 0.85, 138.0, -26.0),
    # The low-gain joint servos lag unevenly on fast descents, which swings
    # the pinch ~2 cm radially and lands an open finger on the object.
    approach_speed=0.10,
)

EMBODIMENTS: dict[str, Embodiment] = {e.name: e for e in (SO101, YAM)}


def get(robot: str | Embodiment) -> Embodiment:
  """Looks up an embodiment by name (instances pass through)."""
  if isinstance(robot, Embodiment):
    return robot
  if robot not in EMBODIMENTS:
    raise ValueError(
        f"Unknown robot {robot!r}; registered: {sorted(EMBODIMENTS)}"
    )
  return EMBODIMENTS[robot]
