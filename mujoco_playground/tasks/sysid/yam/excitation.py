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
"""Scripted 50 Hz excitation trajectories for YAM system identification.

Each trajectory is a (T, 7) array of commands in the I2RT SDK's units: six arm
joints in radians, then the gripper normalized to [0, 1] (0 closed). An episode
holds `REST_POSE`, ramps to a raised center pose along a minimum-jerk profile,
runs a smoothly tapered excitation about that pose, and ramps back to rest.
"""

import dataclasses
import types

from mujoco_playground.robots.yam.real import constant
import numpy as np

DT = constant.CONTROL_PERIOD
NUM_JOINTS = len(constant.JOINTS)
JOINT_LIMITS = np.array([constant.JOINT_BOUNDS[j] for j in constant.JOINTS])
# Margin kept inside the arm's joint limits during the active excitation.
ARM_LIMIT_MARGIN = 0.15

REST_POSE = np.asarray(constant.REST_POSE)
# A raised pose with the wrist well above the desk.
CENTER_POSE = np.array([0.0, 1.25, 1.30, -0.65, 0.0, 0.0, 0.5])
# A slightly shifted center for the held-out validation episode.
VALIDATION_CENTER_POSE = np.array([-0.10, 1.30, 1.25, -0.60, 0.10, -0.15, 0.5])
# Per-joint peak excitation amplitudes (rad, and normalized for the gripper).
AMPLITUDES = np.array([0.30, 0.25, 0.28, 0.35, 0.45, 0.55, 0.42])

RAMP_S = 3.5
TAPER_S = 2.0
HOLD_S = 0.5

EPISODES = types.MappingProxyType({
    "multisine": "ep01_multisine_all_joints",
    "chirp": "ep02_joint_chirp_and_reversals",
    "validation": "ep03_heldout_validation",
})


@dataclasses.dataclass(frozen=True)
class Episode:
  """A command trajectory and the samples where its excitation is active."""

  name: str
  q_cmd: np.ndarray  # (T, 7)
  active: np.ndarray  # (T,) bool


def min_jerk(tau: np.ndarray) -> np.ndarray:
  """Minimum-jerk profile 10 tau^3 - 15 tau^4 + 6 tau^5 on [0, 1]."""
  tau = np.clip(tau, 0.0, 1.0)
  return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def taper(t: np.ndarray, duration: float) -> np.ndarray:
  """C2 envelope: rises over the first TAPER_S, falls over the last TAPER_S.

  Unlike a sin^2 window, the minimum-jerk profile has zero slope and
  curvature at both ends, so the tapered excitation adds no jump in
  acceleration even when the raw signal is nonzero at the window's edges.

  Args:
    t: Times from the start of the active window, in s.
    duration: Length of the active window, in s.

  Returns:
    The envelope at `t`, in [0, 1].
  """
  return np.minimum(min_jerk(t / TAPER_S), min_jerk((duration - t) / TAPER_S))


def _assemble(name: str, center: np.ndarray, delta: np.ndarray) -> Episode:
  """Wraps an excitation about `center` with rest holds and ramps."""
  n_hold = int(round(HOLD_S / DT))
  n_ramp = int(round(RAMP_S / DT))
  a = min_jerk(np.arange(n_ramp + 1) / n_ramp)[:, None]
  ramp_in = REST_POSE + a * (center - REST_POSE)
  ramp_out = center + a * (REST_POSE - center)
  hold = np.repeat(REST_POSE[None], n_hold, axis=0)
  q = np.concatenate(
      [hold, ramp_in[:-1], center + delta, ramp_out[1:], hold], axis=0
  )
  active = np.zeros(len(q), dtype=bool)
  start = n_hold + n_ramp
  active[start : start + len(delta)] = True
  return Episode(name=name, q_cmd=q, active=active)


def multisine(
    duration: float,
    amplitudes: np.ndarray,
    seed: int,
    f_range: tuple[float, float] = (0.15, 1.05),
    n_sines: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
  """Per joint, a peak-normalized sum of non-harmonic random-phase sines."""
  rng = np.random.default_rng(seed)
  t = np.arange(int(round(duration / DT))) * DT
  grid = np.geomspace(*f_range, n_sines * NUM_JOINTS).reshape(n_sines, -1)
  delta = np.zeros((len(t), NUM_JOINTS))
  for j in range(NUM_JOINTS):
    freqs = np.clip(
        grid[:, (3 * j) % NUM_JOINTS] * rng.uniform(0.95, 1.05, n_sines),
        *f_range,
    )
    phases = rng.uniform(0, 2 * np.pi, n_sines)
    sig = np.sin(2 * np.pi * freqs[:, None] * t + phases[:, None]).sum(axis=0)
    delta[:, j] = amplitudes[j] * sig / np.max(np.abs(sig))
  return t, delta * taper(t, duration)[:, None]


def multisine_episode() -> Episode:
  """All joints at once: coupled inertia, armature and servo lag."""
  _, delta = multisine(16.0, AMPLITUDES, seed=101)
  return _assemble(EPISODES["multisine"], CENTER_POSE, delta)


def chirp_episode(window_s: float = 2.8, gap_s: float = 0.4) -> Episode:
  """One joint at a time: a slow reversal plus a 0.2 -> 1.1 Hz chirp."""
  n_win = int(round(window_s / DT))
  n_gap = int(round(gap_s / DT))
  tau = np.arange(n_win) * DT
  envelope = np.sin(np.pi * tau / window_s) ** 4  # C2-zero at the edges.
  f0, f1 = 0.2, 1.1
  chirp = np.sin(2 * np.pi * (f0 * tau + 0.5 * (f1 - f0) / window_s * tau**2))
  reversal = np.sin(2 * np.pi * tau / window_s)
  sig = envelope * (0.55 * reversal + 0.45 * chirp)
  sig /= np.max(np.abs(sig))
  blocks = [np.zeros((n_gap, NUM_JOINTS))]
  for j in range(NUM_JOINTS):
    block = np.zeros((n_win + n_gap, NUM_JOINTS))
    block[:n_win, j] = AMPLITUDES[j] * sig
    blocks.append(block)
  return _assemble(EPISODES["chirp"], CENTER_POSE, np.concatenate(blocks))


def _doublet(
    t: np.ndarray, t0: float, width: float = 1.0, trans: float = 0.4
) -> np.ndarray:
  """+1 then -1 smoothed step doublet starting at t0."""
  return (
      min_jerk((t - t0) / trans)
      - 2 * min_jerk((t - t0 - width) / trans)
      + min_jerk((t - t0 - 2 * width) / trans)
  )


def validation_episode() -> Episode:
  """Independent multisine plus step doublets, for held-out validation."""
  duration = 15.0
  t, delta = multisine(
      duration, 0.8 * AMPLITUDES, seed=303, f_range=(0.12, 0.95)
  )
  for j, (t0, a) in {1: (3.0, 0.10), 2: (6.5, 0.10), 3: (10.0, 0.15)}.items():
    delta[:, j] += a * _doublet(t, t0)
  delta *= taper(t, duration)[:, None]
  return _assemble(EPISODES["validation"], VALIDATION_CENTER_POSE, delta)


def all_episodes() -> list[Episode]:
  return [multisine_episode(), chirp_episode(), validation_episode()]


def limit_violations(episode: Episode) -> list[str]:
  """Commands outside the joint limits, or too close during excitation."""
  errors = []
  lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
  q = episode.q_cmd
  if np.any(q < lo) or np.any(q > hi):
    errors.append(f"{episode.name}: command outside the joint limits")
  arm = q[episode.active, : constant.NUM_ARM_JOINTS]
  margin = np.minimum(
      arm - lo[: constant.NUM_ARM_JOINTS], hi[: constant.NUM_ARM_JOINTS] - arm
  ).min(axis=0)
  for joint, m in zip(constant.ARM_JOINTS, margin, strict=True):
    if m < ARM_LIMIT_MARGIN:
      errors.append(f"{episode.name}: {joint} margin {m:.3f} rad")
  return errors
