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
from absl.testing import absltest
import dm_env
import mujoco
from reaf.core import termination_checker
from reaf.simulation.torch._src.builders import config as builders_config
from reaf.simulation.torch._src.builders.flat_scene import build
from reaf.simulation.torch._src.builders.flat_scene import config
from reaf.simulation.torch._src.common import functional_reset
from reaf.simulation.torch._src.common import model_randomization
from reaf.simulation.torch._src.common import noise
from reaf.simulation.torch._src.common import resetter
from reaf.simulation.torch._src.core import environment
import torch
import warp as wp

_CARTPOLE_XML = """
<mujoco model="cartpole">
  <worldbody>
    <geom name="floor" type="plane" size="4 4 .2"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _create_env(
    resetters: dict[str, config.Component] | None = None,
) -> environment.Environment:
  cfg = config.Environment(
      base_scene=mujoco.MjSpec.from_string(_CARTPOLE_XML),
      resetters=resetters or {},
      sim=builders_config.Sim(
          nworld=4,
          xpu_device="cpu",
          impl="warp",
          simulation_timestep=0.01,
          control_timestep=0.01,
      ),
  )
  return build.build_env(cfg)


class DomainRandomizationResetterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    wp.init()
    wp.config.kernel_cache_dir = "/tmp/kernel_cache"

    self.env = _create_env()

    self.sim_device = self.env.device_layer
    self.model = self.sim_device.sim.model
    self.mj_model = self.sim_device.sim.mj_model
    self.xpu_device = self.env.xpu_device

  def test_on_startup_randomizes_fields(self):
    original_friction = self.model.geom_friction[:].clone()

    term = config.Term(
        func=model_randomization.randomize_field,
        field="geom_friction",
        pattern="cart",
        noise=noise.Noise(dist=noise.Uniform(low=0.1, high=0.9), op="replace"),
    )

    res = resetter.DomainRandomizationResetter([term])
    res.set_env(self.env)

    res.on_startup()

    new_friction = self.model.geom_friction[:]
    cart_id = self.mj_model.geom("cart").id
    floor_id = self.mj_model.geom("floor").id

    # Cart geom friction must change.
    self.assertFalse(
        torch.allclose(original_friction[:, cart_id], new_friction[:, cart_id])
    )
    # Floor geom friction must NOT change.
    self.assertTrue(
        torch.allclose(
            original_friction[:, floor_id], new_friction[:, floor_id]
        )
    )

  def test_on_reset_randomizes_fields(self):
    original_friction = self.model.geom_friction[:].clone()

    term = config.Term(
        func=model_randomization.randomize_field,
        field="geom_friction",
        pattern="cart",
        noise=noise.Noise(dist=noise.Uniform(low=0.1, high=0.9), op="replace"),
        trigger="reset",
    )

    res = resetter.DomainRandomizationResetter([term])
    res.set_env(self.env)

    # Startup should NOT trigger reset term.
    res.on_startup()
    self.assertTrue(
        torch.allclose(original_friction, self.model.geom_friction[:])
    )

    # Trigger reset strictly on envs 1 and 2
    env_mask = torch.tensor(
        [False, True, True, False], dtype=torch.bool, device=self.xpu_device
    )
    res.on_reset(config=None, env_mask=env_mask)

    new_friction = self.model.geom_friction[:]
    cart_id = self.mj_model.geom("cart").id

    # Worlds 1 and 2 must change from baseline!
    self.assertFalse(
        torch.allclose(original_friction[1, cart_id], new_friction[1, cart_id])
    )
    self.assertFalse(
        torch.allclose(original_friction[2, cart_id], new_friction[2, cart_id])
    )

    # Worlds 0 and 3 must NOT change from baseline.
    self.assertTrue(
        torch.allclose(original_friction[0, cart_id], new_friction[0, cart_id])
    )
    self.assertTrue(
        torch.allclose(original_friction[3, cart_id], new_friction[3, cart_id])
    )

  def test_reset_no_drift(self):
    cart_body_id = self.mj_model.body("cart").id
    base_mass = self.model.body_mass[:, cart_body_id].clone()
    base_inertia = self.model.body_inertia[:, cart_body_id].clone()

    startup_term = config.Term(
        func=model_randomization.randomize_field,
        field="body_mass",
        pattern="cart",
        noise=noise.Noise(dist=noise.Uniform(low=2.0, high=2.0), op="scale"),
        trigger="startup",
    )
    reset_term = config.Term(
        func=model_randomization.randomize_field,
        field="body_mass",
        pattern="cart",
        noise=noise.Noise(dist=noise.Uniform(low=0.5, high=0.5), op="add"),
        trigger="reset",
    )

    env = _create_env(
        resetters={
            "dr": config.Component(
                resetter.DomainRandomizationResetter,
                terms=[startup_term, reset_term],
            ),
        }
    )
    model = env.device_layer.sim.model

    expected_startup_mass = base_mass * 2.0
    torch.testing.assert_close(
        model.body_mass[:, cart_body_id], expected_startup_mass
    )

    expected_reset_mass = expected_startup_mass + 0.5
    expected_scale = expected_reset_mass / (base_mass + 1e-9)
    expected_reset_inertia = base_inertia * expected_scale.unsqueeze(-1)

    for _ in range(5):
      env.reset()
      torch.testing.assert_close(
          model.body_mass[:, cart_body_id], expected_reset_mass
      )
      torch.testing.assert_close(
          model.body_inertia[:, cart_body_id],
          expected_reset_inertia,
          rtol=1e-5,
          atol=1e-5,
      )

    # Partial reset: overwrite worlds [0, 1] with a custom value, then reset
    # only world 0; world 0 should be restored + randomized to expected_reset_mass
    # while world 1 keeps its custom value.
    model.body_mass[0:2, cart_body_id] = 99.0
    partial_mask = torch.tensor(
        [True, False, False, False], dtype=torch.bool, device=self.xpu_device
    )
    env.reset_with_options(
        options=env.default_reset_options(), env_mask=partial_mask
    )
    torch.testing.assert_close(
        model.body_mass[0, cart_body_id], expected_reset_mass[0]
    )
    torch.testing.assert_close(
        model.body_mass[1, cart_body_id],
        torch.tensor(99.0, device=self.xpu_device),
    )

  def test_composite_resetters(self):
    cart_body_id = self.mj_model.body("cart").id
    base_mass = self.model.body_mass[:, cart_body_id].clone()

    dr1 = resetter.DomainRandomizationResetter([
        config.Term(
            func=model_randomization.randomize_field,
            field="body_mass",
            pattern="cart",
            noise=noise.Noise(
                dist=noise.Uniform(low=2.0, high=2.0), op="scale"
            ),
            trigger="startup",
        ),
        config.Term(
            func=model_randomization.randomize_field,
            field="body_mass",
            pattern="cart",
            noise=noise.Noise(dist=noise.Uniform(low=0.5, high=0.5), op="add"),
            trigger="reset",
        ),
    ])
    dr2 = resetter.DomainRandomizationResetter([
        config.Term(
            func=model_randomization.randomize_field,
            field="body_mass",
            pattern="cart",
            noise=noise.Noise(dist=noise.Uniform(low=1.0, high=1.0), op="add"),
            trigger="startup",
        ),
        config.Term(
            func=model_randomization.randomize_field,
            field="body_mass",
            pattern="cart",
            noise=noise.Noise(
                dist=noise.Uniform(low=3.0, high=3.0), op="scale"
            ),
            trigger="reset",
        ),
    ])

    comp = resetter.CompositeResetter([dr1, dr2])
    comp.set_env(self.env)
    comp.on_startup()
    self.model.clear_defaults()

    expected_startup = base_mass * 2.0 + 1.0
    torch.testing.assert_close(
        self.model.body_mass[:, cart_body_id], expected_startup
    )

    expected_reset = (expected_startup + 0.5) * 3.0
    all_mask = torch.ones(4, dtype=torch.bool, device=self.xpu_device)
    for _ in range(5):
      comp.do_reset(config=None, env_mask=all_mask)  # pyrefly: ignore[bad-specialization]
      torch.testing.assert_close(
          self.model.body_mass[:, cart_body_id], expected_reset
      )

  def test_functional_reset_in_composite_and_strips_type_param(self):
    startup_called = []

    def set_startup(dc, scale: float):
      del dc
      startup_called.append(scale)

    def set_qpos(dc, env_mask, value: float):
      dc.sim.data.qpos[env_mask, 0] = value

    class SetQvelResetter(environment.EnvironmentReset):

      def on_reset(self, unused_config, env_mask):
        del unused_config
        self._env.device_layer.sim.data.qvel[env_mask, 0] = 2.0

    func_resetter = functional_reset.FunctionalEnvironmentReset([
        config.Term(
            func=set_startup,
            type="startup",
            scale=1.5,
        ),
        config.Term(
            func=set_qpos,
            type="reset",
            value=0.5,
        ),
    ])

    comp = resetter.CompositeResetter([SetQvelResetter(), func_resetter])
    comp.set_env(self.env)
    comp.on_startup()
    self.assertEqual(startup_called, [1.5])

    all_mask = torch.ones(4, dtype=torch.bool, device=self.xpu_device)
    comp.do_reset(config=None, env_mask=all_mask)  # pyrefly: ignore[bad-specialization]

    torch.testing.assert_close(
        self.sim_device.sim.data.qpos[:, 0],
        torch.full((4,), 0.5, device=self.xpu_device),
    )
    torch.testing.assert_close(
        self.sim_device.sim.data.qvel[:, 0],
        torch.full((4,), 2.0, device=self.xpu_device),
    )

  def test_functional_reset_invalid_term_type_raises(self):
    with self.assertRaisesRegex(ValueError, "Invalid reset term type: None"):
      functional_reset.FunctionalEnvironmentReset([
          config.Term(lambda dc: None),
      ])
    with self.assertRaisesRegex(
        ValueError, "Invalid reset term type: 'invalid'"
    ):
      functional_reset.FunctionalEnvironmentReset([
          config.Term(lambda dc: None, type="invalid"),
      ])


class NanTerminationResetterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    wp.init()
    wp.config.kernel_cache_dir = "/tmp/kernel_cache"

  def test_nan_termination_resetter_detects_nan_and_inf(self):
    env = _create_env()
    env.reset()
    data = env.device_layer.sim.data

    res = resetter.NanTerminationResetter()
    res.set_env(env)

    # All finite initially.
    result = res.check_termination()
    assert result is not None
    torch.testing.assert_close(
        result,
        torch.full(
            (4,),
            termination_checker.TerminationResult.DO_NOT_TERMINATE,
            dtype=torch.int32,
        ),
    )

    # Inject NaN/Inf across qpos, qvel, qacc, and qacc_warmstart.
    for field_name, bad_val in (
        ("qpos", float("nan")),
        ("qvel", float("inf")),
        ("qacc", float("-inf")),
        ("qacc_warmstart", float("nan")),
    ):
      env.reset()
      getattr(data, field_name)[1, 0] = bad_val
      result = res.check_termination()
      assert result is not None
      expected = torch.tensor(
          [
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
              termination_checker.TerminationResult.TERMINATE,
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
          ],
          dtype=torch.int32,
      )
      torch.testing.assert_close(result, expected)

  def test_build_env_nan_termination_disabled_by_default(self):
    env = _create_env()
    comp = env.environment_reset
    self.assertIsInstance(comp, resetter.CompositeResetter)
    assert isinstance(comp, resetter.CompositeResetter)
    self.assertEmpty(comp._resetters)
    self.assertIsNone(comp.check_termination())

  def test_build_env_nan_termination_enabled(self):
    cfg = config.Environment(
        base_scene=mujoco.MjSpec.from_string(_CARTPOLE_XML),
        nan_termination=True,
        sim=builders_config.Sim(
            nworld=4,
            xpu_device="cpu",
            impl="warp",
            simulation_timestep=0.01,
            control_timestep=0.01,
        ),
    )
    env = build.build_env(cfg)
    env.reset()
    comp = env.environment_reset
    self.assertIsInstance(comp, resetter.CompositeResetter)
    assert isinstance(comp, resetter.CompositeResetter)
    self.assertTrue(
        any(
            isinstance(r, resetter.NanTerminationResetter)
            for r in comp._resetters
        )
    )

    # Corrupt world 2 velocity with NaN before stepping.
    env.device_layer.sim.data.qvel[2, 0] = float("nan")
    ts = env.step({})
    expected_step_type = torch.tensor(
        [
            dm_env.StepType.MID,
            dm_env.StepType.MID,
            dm_env.StepType.LAST,
            dm_env.StepType.MID,
        ],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(ts.step_type, expected_step_type)


if __name__ == "__main__":
  absltest.main()
