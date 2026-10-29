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
"""Tests for YAM system identification."""

import tempfile

from absl.testing import absltest
from mujoco import sysid as mj_sysid
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.sim import scene
from mujoco_playground.tasks.sysid.yam import data_loader
from mujoco_playground.tasks.sysid.yam import excitation
from mujoco_playground.tasks.sysid.yam import preflight
from mujoco_playground.tasks.sysid.yam import sysid
import numpy as np

_TRUTH = {"joint2_armature": 0.05, "joint4_kv": 2.2, "joint3_kp": 60.0}


def _command(duration: float, seed: int) -> np.ndarray:
  _, delta = excitation.multisine(
      duration, excitation.AMPLITUDES, seed, f_range=(0.4, 1.5)
  )
  return excitation.CENTER_POSE + delta


def _synthesize(
    truth: mj_sysid.ParameterDict, q_cmd: np.ndarray, name: str
) -> data_loader.Trajectory:
  """'Measures' a rollout of the model with `truth` parameters."""
  n = len(q_cmd)
  t_ns = np.arange(n, dtype=np.int64) * 20_000_000
  placeholder = data_loader.from_arrays(
      name,
      t_ns,
      t_ns,
      np.repeat(q_cmd[:1], n, 0),
      q_cmd,
      np.zeros((n, 7)),
      crop_rest=False,
  )
  spec = scene.yam_menagerie()
  spec.option.timestep = 0.002
  ms = data_loader.to_model_sequences([placeholder], spec)
  p = truth.copy()
  residual_fn = mj_sysid.build_residual_fn(models_sequences=[ms])
  _, preds, _ = residual_fn(p.as_vector(), p, return_pred_all=True)
  pred = preds[0][0]
  keep = np.isin(np.round(placeholder.timestamps, 6), np.round(pred.times, 6))
  q_meas = pred.data.copy()
  q_meas[:, -1] /= 0.0475  # Left finger travel in m -> normalized.
  return data_loader.from_arrays(
      name, t_ns[keep], t_ns[keep], q_meas, q_cmd[keep], None, crop_rest=False
  )


class ExcitationTest(absltest.TestCase):

  def test_episodes_start_and_end_at_rest_inside_the_limits(self) -> None:
    for episode in excitation.all_episodes():
      self.assertEmpty(excitation.limit_violations(episode))
      np.testing.assert_allclose(episode.q_cmd[0], excitation.REST_POSE)
      np.testing.assert_allclose(episode.q_cmd[-1], excitation.REST_POSE)

  def test_commands_are_smooth(self) -> None:
    for episode in excitation.all_episodes():
      jerk = np.diff(episode.q_cmd, n=3, axis=0) / excitation.DT**3
      self.assertLess(np.abs(jerk).max(), 400.0, msg=episode.name)

  def test_preflight_passes_in_the_desk_scene(self) -> None:
    result = preflight.check_episode(excitation.multisine_episode())
    self.assertTrue(result.ok, msg=str(result))


class SysIdTest(absltest.TestCase):

  def test_parameters_freeze_joint1_kp_and_the_gripper(self) -> None:
    params = sysid.build_parameters()
    self.assertLen(list(params.keys()), 4 * len(sysid.ACTUATORS))
    self.assertTrue(params["joint1_kp"].frozen)
    self.assertTrue(params["gripper_kv"].frozen)
    self.assertFalse(params["joint2_kp"].frozen)
    self.assertEqual(params["joint4_kp"].nominal[0], 40.0)

  def test_wrist_camera_parameters_are_optional_and_start_at_cad(self) -> None:
    self.assertNotIn("wrist_camera_mass", sysid.build_parameters())
    params = sysid.build_parameters(wrist_camera=True)
    self.assertAlmostEqual(params["wrist_camera_mass"].nominal[0], 0.1087)
    self.assertLen(params["wrist_camera_ipos"].nominal, 3)

  def test_freezes_joints_and_parameters(self) -> None:
    params = sysid.build_parameters(
        freeze_joints=("joint6",), freeze_params=("joint3_armature",)
    )
    for attr in ("armature", "frictionloss", "kp", "kv"):
      self.assertTrue(params[f"joint6_{attr}"].frozen)
    self.assertTrue(params["joint3_armature"].frozen)
    self.assertFalse(params["joint3_kp"].frozen)
    with self.assertRaises(ValueError):
      sysid.build_parameters(freeze_params=("joint9_kp",))

  def test_flags_unidentified_values(self) -> None:
    reason = sysid.unidentified_reason
    self.assertEqual(reason(0.5, 0.032, (1e-4, 0.5), 0.01), "at bound")
    self.assertEqual(
        reason(64.7, 40.0, (2.0, 250.0), 27.3), "interval contains nominal"
    )
    self.assertEqual(reason(0.18, 0.5, (1e-4, 1.0), 0.15), "wide interval")
    self.assertIsNone(reason(0.175, 0.032, (1e-4, 0.5), 0.0105))

  def test_recovers_perturbed_parameters(self) -> None:
    truth = sysid.build_parameters()
    for k, v in _TRUTH.items():
      truth[k].value[:] = v
    trajectories = [
        _synthesize(truth, _command(3.0, s), f"syn{s}") for s in (1, 2)
    ]
    init = sysid.build_parameters()
    for k in init.keys():
      init[k].frozen = k not in _TRUTH
    with tempfile.TemporaryDirectory() as tmp:
      result = sysid.run_sysid(
          trajectories,
          save_path=tmp,
          max_iters=30,
          initial_params=init,
      )
    for k, v in _TRUTH.items():
      got = float(result.opt_params[k].value[0])
      self.assertLess(abs(got - v) / v, 0.05, msg=f"{k}: {got} vs {v}")

  def test_joint_order_matches_the_sdk(self) -> None:
    self.assertEqual(
        data_loader.SENSOR_JOINTS[: constant.NUM_ARM_JOINTS],
        constant.ARM_JOINTS,
    )


if __name__ == "__main__":
  absltest.main()
