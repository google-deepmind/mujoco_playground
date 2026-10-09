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

"""OP3 randomization and perturbation tests without external robot assets."""

from unittest import mock

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from absl.testing import absltest
from mujoco import mjx

from mujoco_playground._src import locomotion, mjx_env
from mujoco_playground._src.locomotion.op3 import joystick, randomize


def _small_op3_model() -> mjx.Model:
  """Minimal floating-base position-controlled MuJoCo model."""
  xml = """
  <mujoco>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 0.1"/>
      <body name="body_link" pos="0 0 1">
        <freejoint/>
        <geom type="sphere" size="0.1" mass="1"/>
        <body name="limb" pos="0 0 0.2">
          <joint name="hip" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.2"
                size="0.05" mass="0.5"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <position joint="hip" kp="21"/>
    </actuator>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  model.dof_armature[6:] = 0.12
  model.dof_frictionloss[6:] = 0.45
  model.dof_damping[6:] = 1.5
  return mjx.put_model(model)


def _mock_env(model: mjx.Model, pushes: bool) -> joystick.Joystick:
  """Set up only the fields needed by Joystick.step."""
  env = joystick.Joystick.__new__(joystick.Joystick)
  env._config = joystick.default_config()
  env._config.pushes = pushes
  env._ctrl_dt = env._config.ctrl_dt
  env._sim_dt = env._config.sim_dt
  env._mjx_model = model
  env._default_pose = jp.zeros(model.nu)
  env._lowers = -jp.ones(model.nu)
  env._uppers = jp.ones(model.nu)
  env._torso_body_id = randomize.TORSO_BODY_ID
  env._torso_mass = 3.0
  env.sample_command = lambda _rng: jp.zeros(3)
  env._get_obs = lambda *_args: jp.zeros(49)
  env._get_termination = lambda _data: jp.array(False)
  env._get_reward = lambda *_args: {"tracking_lin_vel": jp.array(0.0)}
  return env


def _state(model: mjx.Model) -> mjx_env.State:
  return mjx_env.State(
      data=mjx.make_data(model),
      obs=jp.zeros(147),
      reward=jp.array(0.0),
      done=jp.array(0.0),
      metrics={},
      info={
          "rng": jax.random.PRNGKey(0),
          "command": jp.zeros(3),
          "step": jp.array(0),
          "last_act": jp.zeros(model.nu),
          "last_last_act": jp.zeros(model.nu),
          "last_vel": jp.zeros(model.nv - 6),
          "pert_steps": jp.array(1),
          "steps_since_last_pert": jp.array(10),
          "steps_until_next_pert": jp.array(10),
          "pert_duration_seconds": jp.array(0.2),
          "pert_duration": jp.array(10),
          "direction": jp.array([1.0, 0.0, 0.0]),
          "vel_kick": jp.array(2.0),
          "motor_targets": jp.zeros(model.nu),
      },
  )


class Op3DomainRandomizationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = _small_op3_model()
    self.batch_size = 16
    self.keys = jax.random.split(jax.random.PRNGKey(123), self.batch_size)

  def test_registered_op3_randomizer(self):
    self.assertIs(
        locomotion.get_domain_randomizer("Op3Joystick"),
        randomize.domain_randomize,
    )

  def test_only_changed_model_fields_are_batched(self):
    changed = {
        "geom_friction",
        "dof_frictionloss",
        "dof_armature",
        "body_mass",
        "actuator_gainprm",
        "actuator_biasprm",
        "dof_damping",
    }
    updated, in_axes = randomize.domain_randomize(self.model, self.keys)
    for field in changed:
      self.assertEqual(getattr(in_axes, field), 0, field)
      self.assertEqual(getattr(updated, field).shape[0], self.batch_size, field)
    self.assertIsNone(in_axes.qpos0)
    self.assertIsNone(in_axes.geom_size)
    np.testing.assert_array_equal(updated.qpos0, self.model.qpos0)

  def test_friction_and_gain_ranges_and_independent_samples(self):
    updated, _ = randomize.domain_randomize(self.model, self.keys)
    friction = np.asarray(updated.geom_friction[:, 0, 0])
    self.assertTrue(np.all((friction >= 0.4) & (friction <= 1.0)))
    self.assertGreater(np.std(friction), 0.0)

    armature_ratio = np.asarray(
        updated.dof_armature[:, 6:] / self.model.dof_armature[6:]
    )
    self.assertTrue(np.all((armature_ratio >= 1.0) & (armature_ratio <= 1.05)))
    friction_ratio = np.asarray(
        updated.dof_frictionloss[:, 6:] / self.model.dof_frictionloss[6:]
    )
    self.assertTrue(np.all((friction_ratio >= 0.9) & (friction_ratio <= 1.1)))
    kd_ratio = np.asarray(
        updated.dof_damping[:, 6:] / self.model.dof_damping[6:]
    )
    self.assertTrue(np.all((kd_ratio >= 0.8) & (kd_ratio <= 1.2)))
    kp_ratio = np.asarray(
        updated.actuator_gainprm[:, :, 0] / self.model.actuator_gainprm[:, 0]
    )
    self.assertTrue(np.all((kp_ratio >= 0.8) & (kp_ratio <= 1.2)))
    np.testing.assert_allclose(
        updated.actuator_biasprm[:, :, 1],
        -updated.actuator_gainprm[:, :, 0],
    )

  def test_link_masses_remain_positive(self):
    updated, _ = randomize.domain_randomize(self.model, self.keys)
    self.assertTrue(np.all(np.asarray(updated.body_mass[:, 1:]) > 0))


class Op3PerturbationsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = _small_op3_model()

  def test_pushes_are_opt_in(self):
    self.assertFalse(joystick.default_config().pushes)

  def test_enabled_pushes_apply_force(self):
    env = _mock_env(self.model, pushes=True)
    state = _state(self.model)
    applied = env._maybe_apply_perturbation(state, jax.random.PRNGKey(42))
    self.assertGreater(float(applied.data.xfrc_applied[1, 0]), 0.0)
    self.assertEqual(float(applied.data.xfrc_applied[1, 1]), 0.0)

  def test_disabled_pushes_do_not_call_perturbation(self):
    env = _mock_env(self.model, pushes=False)
    state = _state(self.model)
    with (
        mock.patch.object(
            env,
            "_maybe_apply_perturbation",
            wraps=env._maybe_apply_perturbation,
        ) as perturb,
        mock.patch.object(
            mjx_env, "step", side_effect=lambda _model, data, *_args: data
        ),
    ):
      env.step(state, jp.zeros(self.model.nu))
      perturb.assert_not_called()

  def test_enabled_pushes_call_perturbation(self):
    env = _mock_env(self.model, pushes=True)
    state = _state(self.model)
    with (
        mock.patch.object(
            env, "_maybe_apply_perturbation", return_value=state
        ) as perturb,
        mock.patch.object(
            mjx_env, "step", side_effect=lambda _model, data, *_args: data
        ),
    ):
      env.step(state, jp.zeros(self.model.nu))
      perturb.assert_called_once()


if __name__ == "__main__":
  absltest.main()
