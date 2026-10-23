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
"""Tests for the Environment class."""

from collections.abc import Mapping
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
import numpy as np
from reaf.simulation.torch._src.core import device as device_module
from reaf.simulation.torch._src.core import device_layer as device_layer_module
from reaf.simulation.torch._src.core import discount_provider as discount_provider_module
from reaf.simulation.torch._src.core import environment
from reaf.simulation.torch._src.core import observation_space_adapter
from reaf.simulation.torch._src.core import reward_provider
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import task_layer as task_layer_module
from reaf.simulation.torch._src.core import termination_checker
from reaf.simulation.torch._src.core import torch_wrapper
from reaf.simulation.torch._src.core import types
import torch

_PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.01"/>
  <worldbody>
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 1" size="0.05"/>
      <inertial pos="0 0 0.5" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge"/>
  </actuator>
</mujoco>
"""


class PendulumDevice(device_module.Device):
  """Simple device that exposes the pendulum joint angles."""

  def __init__(self, home_qpos: float = 0.0):
    self._home_qpos = home_qpos

  @property
  def name(self) -> str:
    return "pendulum"

  def substep_commands_spec(self) -> dict[str, specs.Array]:
    return {"motor_ctrl": specs.Array(shape=(1,), dtype=np.float32)}

  def substep_measurements_spec(self) -> dict[str, specs.Array]:
    return {"joint_pos": specs.Array(shape=(1,), dtype=np.float32)}

  def set_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      consumed_substep_commands: Mapping[str, torch.Tensor],
  ) -> None:
    simulator.data.ctrl[:, 0] = consumed_substep_commands["motor_ctrl"]

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    return {"joint_pos": simulator.data.qpos[:, 0:1].clone()}

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: Any = None,
  ) -> None:
    if self._home_qpos != 0.0:
      if env_mask is None:
        simulator.data.qpos[:, 0] = self._home_qpos
      else:
        simulator.data.qpos[:, 0] = torch.where(
            env_mask, self._home_qpos, simulator.data.qpos[:, 0]
        )


class ClockDevice(device_module.Device):
  """Simple clock device for testing."""

  def __init__(self, name: str = "clock", time_output_name: str = "time"):
    self._name = name
    self._time_output_name = time_output_name

  @property
  def name(self) -> str:
    return self._name

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return {self._time_output_name: specs.Array(shape=(), dtype=np.float32)}

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    return {self._time_output_name: simulator.data.time.clone()}

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: Any = None,
  ) -> None:
    del simulator, env_mask


class DefaultDiscountProvider(discount_provider_module.DiscountProvider):
  """Returns constant discount of 1.0."""

  def __init__(self, nworld: int, xpu_device: str = "cpu"):
    self._nworld = nworld
    self._xpu_device = xpu_device

  @property
  def name(self) -> str:  # pyrefly: ignore[bad-override]
    return "default_discount"

  def required_features_keys(self) -> set[str]:
    return set()

  def discount_spec(self) -> specs.Array:
    return specs.Array(shape=(), dtype=np.float32)

  def compute_discount(
      self,
      required_features: Mapping[str, torch.Tensor],
      termination_state: Any,
  ) -> torch.Tensor:
    del required_features, termination_state
    return torch.ones(
        self._nworld, dtype=torch.float32, device=self._xpu_device
    )


class ConstantRewardProvider(reward_provider.RewardProvider):
  """Returns zero reward."""

  def __init__(self, nworld: int, xpu_device: str):
    self._nworld = nworld
    self._xpu_device = xpu_device

  @property
  def name(self) -> str:  # pyrefly: ignore[bad-override]
    return "constant_reward"

  def required_features_keys(self) -> set[str]:
    return set()

  def reward_spec(self) -> specs.Array:
    return specs.Array(shape=(), dtype=np.float32)

  def compute_reward(
      self, features: Mapping[str, torch.Tensor]
  ) -> torch.Tensor:
    return torch.zeros(
        self._nworld, dtype=torch.float32, device=self._xpu_device
    )


class StaggeredTerminationChecker(termination_checker.TerminationChecker):
  """Truncates each env when time exceeds a per-env timeout."""

  def __init__(self, timeouts: torch.Tensor, time_feature_name: str):
    self._timeouts = timeouts
    self._time_feature_name = time_feature_name

  def name(self) -> str:
    return "staggered_termination"

  def check_termination(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> torch.Tensor:
    time = required_features[self._time_feature_name]
    return torch.where(
        time > self._timeouts + 1e-6,
        termination_checker.TerminationResult.TRUNCATE,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

  def required_features_keys(self) -> set[str]:
    return {self._time_feature_name}


class PendulumReset(environment.EnvironmentReset):
  """Resets the pendulum simulation."""

  def on_reset(self, config, env_mask) -> None:
    pass


class PendulumObservationAdapter(
    observation_space_adapter.ObservationSpaceAdapter
):
  """Maps pendulum features to observations."""

  def __init__(self, nworld: int = 1):
    self._nworld = nworld

  def observations_from_features(
      self, features: Mapping[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    return {
        "joint_pos": features["joint_pos"].float(),
        "time": features["time"].unsqueeze(-1).float(),
    }

  def observation_spec(self) -> gdmr_types.ObservationSpec:  # pyrefly: ignore[invalid-type-var]
    return {
        "joint_pos": specs.Array(shape=(self._nworld, 1), dtype=np.float32),
        "time": specs.Array(shape=(self._nworld, 1), dtype=np.float32),
    }

  def task_features_keys(self) -> set[str]:
    return {"joint_pos", "time"}


def _build_pendulum_env(
    nworld: int,
    xpu_device: str,
    timeouts: torch.Tensor,
    impl: types.Impl,
    home_qpos: float = 0.0,
    environment_reset_fn: environment.EnvironmentReset | None = None,
    auto_reset_mode: environment.AutoResetMode = environment.AutoResetMode.SYNC,
) -> environment.Environment:
  """Builds a simple pendulum environment for testing."""
  mj_spec = mujoco.MjSpec.from_string(_PENDULUM_XML)
  control_timestep = 0.02
  pendulum = PendulumDevice(home_qpos=home_qpos)
  clock = ClockDevice(name="clock", time_output_name="time")

  sim = simulator_module.Simulator(
      mj_spec=mj_spec,
      impl=impl,
      nworld=nworld,
      xpu_device=xpu_device,
  )
  dev_layer = device_layer_module.DeviceLayer(
      simulator=sim,
      devices=[pendulum, clock],
      control_timestep=control_timestep,
  )

  reward = ConstantRewardProvider(nworld=nworld, xpu_device=xpu_device)
  discount = DefaultDiscountProvider(nworld=nworld, xpu_device=xpu_device)
  terminations = [
      StaggeredTerminationChecker(
          timeouts=timeouts,
          time_feature_name="time",
      ),
  ]
  task_layer = task_layer_module.TaskLayer(
      commands_processors=[],
      features_producers=[],
      reward_provider=reward,
      termination_checkers=terminations,
      discount_provider=discount,
      features_observers=[],
      loggers=[],
      xpu_device=xpu_device,
      nworld=nworld,
  )
  reset = (
      environment_reset_fn
      if environment_reset_fn is not None
      else PendulumReset()
  )
  obs_adapter = PendulumObservationAdapter(nworld=nworld)
  return environment.Environment(
      device_layer=dev_layer,
      task_layer=task_layer,
      observation_space_adapter=obs_adapter,
      environment_reset=reset,
      auto_reset_mode=auto_reset_mode,
  )


class WarpEnvironmentTest(absltest.TestCase):
  """Environment tests using the Warp backend."""

  IMPL = types.Impl.WARP
  XPU_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

  def test_entity_reset_sets_home_qpos(self):
    """Verify entity.reset() sets home_qpos and is reflected in observations."""
    nworld = 2
    timeouts = torch.tensor(
        [1.0, 1.0], dtype=torch.float32, device=self.XPU_DEVICE
    )
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
        home_qpos=0.75,
    )
    timestep = env.reset()
    torch.testing.assert_close(
        timestep.observation["joint_pos"],
        torch.tensor(
            [[0.75], [0.75]], dtype=torch.float32, device=self.XPU_DEVICE
        ),
    )

  def test_user_on_reset_overrides_entity_home_qpos(self):
    """Verify user on_reset() overrides entity home_qpos and is reflected."""
    nworld = 2
    timeouts = torch.tensor(
        [1.0, 1.0], dtype=torch.float32, device=self.XPU_DEVICE
    )
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
        home_qpos=0.75,
        environment_reset_fn=OverridePendulumReset(override_pos=1.5),
    )
    timestep = env.reset()
    torch.testing.assert_close(
        timestep.observation["joint_pos"],
        torch.tensor(
            [[1.5], [1.5]], dtype=torch.float32, device=self.XPU_DEVICE
        ),
    )

  def test_conditional_reset(self):
    """Verify that reset only sets step_type=FIRST for reset envs."""
    nworld = 4
    # Staggered timeouts: envs 0,1 timeout at 0.04s, envs 2,3 at 0.08s.
    timeouts = torch.tensor(
        [0.04, 0.04, 0.08, 0.08], dtype=torch.float32, device=self.XPU_DEVICE
    )
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
    )

    # Initial reset: all envs should have step_type=FIRST.
    timestep = env.reset()
    self.assertTrue(
        torch.all(timestep.step_type == dm_env.StepType.FIRST),
        f"Expected all FIRST after reset, got: {timestep.step_type}",
    )

    # Step until envs 0,1 reach LAST (timeout at 0.04s) while 2,3 are still MID.
    # control_timestep=0.02, so after 3 steps (0.06s), envs 0,1 should be LAST.
    action = {
        "motor_ctrl": torch.zeros(
            1, dtype=torch.float32, device=self.XPU_DEVICE
        )
    }
    for _ in range(3):
      timestep = env.step(action)

    # Envs 0,1 should be LAST, envs 2,3 should be MID.
    expected_step_types = torch.tensor(
        [
            dm_env.StepType.LAST,
            dm_env.StepType.LAST,
            dm_env.StepType.MID,
            dm_env.StepType.MID,
        ],
        dtype=torch.uint8,
        device=self.XPU_DEVICE,
    )
    self.assertTrue(
        torch.all(timestep.step_type == expected_step_types),
        f"Expected {expected_step_types}, got: {timestep.step_type}",
    )

    # Next step should trigger auto-reset ONLY for envs 0,1.
    timestep = env.step(action)

    # After auto-reset: envs 0,1 should be FIRST, envs 2,3 should be MID.
    expected_step_types = torch.tensor(
        [
            dm_env.StepType.FIRST,
            dm_env.StepType.FIRST,
            dm_env.StepType.MID,
            dm_env.StepType.MID,
        ],
        dtype=torch.uint8,
        device=self.XPU_DEVICE,
    )
    self.assertTrue(
        torch.all(timestep.step_type == expected_step_types),
        f"Expected {expected_step_types}, got: {timestep.step_type}",
    )

  def test_reset_observations_preserved_after_reset(self):
    """Verify reset observations are preserved after auto-reset."""
    nworld = 4
    # Staggered timeouts: envs 0,1 timeout at 0.04s, envs 2,3 at 0.08s.
    timeouts = torch.tensor(
        [0.04, 0.04, 0.08, 0.08], dtype=torch.float32, device=self.XPU_DEVICE
    )
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
    )

    _ = env.reset()
    action = {
        "motor_ctrl": torch.zeros(
            1, dtype=torch.float32, device=self.XPU_DEVICE
        )
    }

    # Step until envs 0,1 reach LAST (3 steps = 0.06s > 0.04s timeout).
    for _ in range(3):
      timestep = env.step(action)

    # Envs 0,1 should be LAST, envs 2,3 should be MID.
    self.assertTrue(
        torch.all(timestep.step_type[:2] == dm_env.StepType.LAST),
        f"Envs 0,1 should be LAST, got: {timestep.step_type}",
    )

    # Store time for reset envs before the auto-reset step.
    time_before_01 = timestep.observation["time"][:2].clone()
    time_before_23 = timestep.observation["time"][2:].clone()

    # Next step triggers auto-reset for envs 0,1 while stepping envs 2,3.
    timestep = env.step(action)

    # Envs 0,1 should now be FIRST.
    self.assertTrue(
        torch.all(timestep.step_type[:2] == dm_env.StepType.FIRST),
        f"Envs 0,1 should be FIRST after reset, got: {timestep.step_type}",
    )

    # Time for reset envs should be ~0 (less than before).
    time_after_01 = timestep.observation["time"][:2]
    self.assertTrue(
        torch.all(time_after_01 < time_before_01),
        f"Envs 0,1 time should be reset: before={time_before_01}, "
        f"after={time_after_01}",
    )

    # Time for non-reset envs should have advanced.
    time_after_23 = timestep.observation["time"][2:]
    self.assertTrue(
        torch.all(time_after_23 > time_before_23),
        f"Envs 2,3 time should advance: before={time_before_23}, "
        f"after={time_after_23}",
    )

  def test_auto_reset_mode_no_sync_and_disabled(self):
    """Verify AutoResetMode.NO_SYNC auto-resets and DISABLED does not."""
    nworld = 4
    timeouts = torch.tensor(
        [0.04, 0.04, 0.08, 0.08], dtype=torch.float32, device=self.XPU_DEVICE
    )
    env_no_sync = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
        auto_reset_mode=environment.AutoResetMode.NO_SYNC,
    )
    _ = env_no_sync.reset()
    action = {
        "motor_ctrl": torch.zeros(
            1, dtype=torch.float32, device=self.XPU_DEVICE
        )
    }
    for _ in range(3):
      ts = env_no_sync.step(action)
    self.assertTrue(torch.all(ts.step_type[:2] == dm_env.StepType.LAST))
    ts_next = env_no_sync.step(action)
    self.assertTrue(torch.all(ts_next.step_type[:2] == dm_env.StepType.FIRST))
    self.assertTrue(torch.all(ts_next.step_type[2:] == dm_env.StepType.MID))

    env_disabled = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
        auto_reset_mode=environment.AutoResetMode.DISABLED,
    )
    _ = env_disabled.reset()
    for _ in range(3):
      ts = env_disabled.step(action)
    self.assertTrue(torch.all(ts.step_type[:2] == dm_env.StepType.LAST))
    ts_next = env_disabled.step(action)
    # With DISABLED, envs 0,1 are NOT automatically reset to FIRST.
    self.assertFalse(torch.any(ts_next.step_type[:2] == dm_env.StepType.FIRST))


class OverridePendulumReset(environment.EnvironmentReset):
  """Overrides pendulum joint angle on reset."""

  def __init__(self, override_pos: float):
    self._override_pos = override_pos

  def on_reset(self, config, env_mask) -> None:
    del config
    sim = self._env.device_layer.sim
    sim.data.qpos[:, 0] = torch.where(
        env_mask, self._override_pos, sim.data.qpos[:, 0]
    )


class CppEnvironmentTest(absltest.TestCase):
  """Environment tests using the CPP backend (nworld=1 only)."""

  IMPL = types.Impl.CPP
  XPU_DEVICE = "cpu"

  def test_entity_reset_sets_home_qpos(self):
    """Verify entity.reset() sets home_qpos and is reflected in observations."""
    nworld = 1
    timeouts = torch.tensor([1.0], dtype=torch.float32, device=self.XPU_DEVICE)
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
        home_qpos=0.75,
    )
    timestep = env.reset()
    torch.testing.assert_close(
        timestep.observation["joint_pos"],
        torch.tensor([[0.75]], dtype=torch.float32, device=self.XPU_DEVICE),
    )

  def test_user_on_reset_overrides_entity_home_qpos(self):
    """Verify user on_reset() overrides entity home_qpos and is reflected."""
    nworld = 1
    timeouts = torch.tensor([1.0], dtype=torch.float32, device=self.XPU_DEVICE)
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
        home_qpos=0.75,
        environment_reset_fn=OverridePendulumReset(override_pos=1.5),
    )
    timestep = env.reset()
    torch.testing.assert_close(
        timestep.observation["joint_pos"],
        torch.tensor([[1.5]], dtype=torch.float32, device=self.XPU_DEVICE),
    )

  def test_reset_and_step_cycle(self):
    """Verify reset/step cycle works with CPP backend."""
    nworld = 1
    # Single env times out at 0.04s.
    timeouts = torch.tensor([0.04], dtype=torch.float32, device=self.XPU_DEVICE)
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
    )

    # Initial reset: should have step_type=FIRST.
    timestep = env.reset()
    self.assertEqual(timestep.step_type.item(), dm_env.StepType.FIRST)

    # Step 1: time=0.02, step_type=MID
    action = {
        "motor_ctrl": torch.zeros(
            1, dtype=torch.float32, device=self.XPU_DEVICE
        )
    }
    timestep = env.step(action)
    self.assertEqual(timestep.step_type.item(), dm_env.StepType.MID)

    # Step 2: time=0.04, step_type=MID (not > 0.04 + epsilon)
    timestep = env.step(action)
    self.assertEqual(timestep.step_type.item(), dm_env.StepType.MID)

    # Step 3: time=0.06 > 0.04 + epsilon -> LAST
    timestep = env.step(action)
    self.assertEqual(
        timestep.step_type.item(),
        dm_env.StepType.LAST,
        f"Expected LAST, got {timestep.step_type.item()}, "
        f"time={timestep.observation['time'].item():.4f}",
    )

    # Step 4: auto-reset should kick in -> FIRST
    timestep = env.step(action)
    self.assertEqual(timestep.step_type.item(), dm_env.StepType.FIRST)

    # Time should be reset to 0.
    time_after = timestep.observation["time"]
    self.assertEqual(time_after.item(), 0.0)


class MassRandomizationReset(environment.EnvironmentReset):
  """Writes body_mass at startup and/or reset (requires FULL set_const)."""

  def __init__(self, on_startup: bool, on_reset: bool):
    self._write_on_startup = on_startup
    self._write_on_reset = on_reset
    self.num_writes = 0

  def _write_mass(self) -> None:
    model = self._env.device_layer.sim.model
    model.body_mass[:, 1] = model.body_mass[:, 1] + 1.0
    self.num_writes += 1

  def on_startup(self) -> None:
    if self._write_on_startup:
      self._write_mass()

  def on_reset(self, config, env_mask) -> None:
    del config, env_mask
    if self._write_on_reset:
      self._write_mass()


class EnvironmentSetConstTest(parameterized.TestCase):
  """The Environment runs set_const() and forward() after startup/reset."""

  def _build_env(self, impl, reset_fn):
    nworld = 2 if impl == types.Impl.WARP else 1
    xpu_device = (
        "cuda:0"
        if impl == types.Impl.WARP and torch.cuda.is_available()
        else "cpu"
    )
    timeouts = torch.ones(nworld, dtype=torch.float32, device=xpu_device)
    return _build_pendulum_env(
        nworld=nworld,
        xpu_device=xpu_device,
        timeouts=timeouts,
        impl=impl,
        environment_reset_fn=reset_fn,
    )

  @parameterized.parameters(types.Impl.WARP, types.Impl.CPP)
  def test_startup_write_runs_set_const_once(self, impl):
    reset_fn = MassRandomizationReset(on_startup=True, on_reset=False)
    env = self._build_env(impl, reset_fn)
    sim = env.device_layer.sim
    # Startup already flushed the pending level.
    self.assertEqual(reset_fn.num_writes, 1)
    self.assertEqual(
        sim.model.pending_set_const, torch_wrapper.SetConstLevel.NONE
    )
    # body_subtreemass of the pole = new body_mass.
    torch.testing.assert_close(
        sim.model.body_subtreemass[:, 1], sim.model.body_mass[:, 1]
    )

  @parameterized.parameters(types.Impl.WARP, types.Impl.CPP)
  def test_reset_write_runs_set_const_then_forward_once(self, impl):
    reset_fn = MassRandomizationReset(on_startup=False, on_reset=True)
    env = self._build_env(impl, reset_fn)
    sim = env.device_layer.sim
    calls = []
    real_set_const, real_forward = sim.set_const, sim.forward

    def set_const_spy():
      level = real_set_const()
      calls.append(("set_const", level))
      return level

    def forward_spy():
      calls.append(("forward", None))
      real_forward()

    with (
        mock.patch.object(sim, "set_const", side_effect=set_const_spy),
        mock.patch.object(sim, "forward", side_effect=forward_spy),
    ):
      env.reset()

    self.assertEqual(
        calls,
        [("set_const", torch_wrapper.SetConstLevel.FULL), ("forward", None)],
    )
    torch.testing.assert_close(
        sim.model.body_subtreemass[:, 1], sim.model.body_mass[:, 1]
    )

  @parameterized.parameters(types.Impl.WARP, types.Impl.CPP)
  def test_reset_without_model_writes_skips_set_const(self, impl):
    env = self._build_env(impl, PendulumReset())
    sim = env.device_layer.sim
    with (
        mock.patch.object(sim, "forward", wraps=sim.forward) as forward,
        mock.patch.object(
            simulator_module.mujoco, "mj_setConst"
        ) as mj_set_const,
        mock.patch.object(simulator_module.mjw, "set_const") as mjw_set_const,
    ):
      env.reset()
    forward.assert_called_once()
    mj_set_const.assert_not_called()
    mjw_set_const.assert_not_called()

  def test_device_layer_without_simulator_skips_set_const(self):
    """Hardware device layers whose `sim` is None only run forward()."""
    env = self._build_env(types.Impl.WARP, PendulumReset())
    real_layer = env.device_layer
    forward_calls = []

    class SimlessDeviceLayer:
      """Delegates to a real DeviceLayer but exposes no Simulator."""

      sim = None

      def forward(self):
        forward_calls.append(1)
        real_layer.forward()

      def __getattr__(self, name):
        return getattr(real_layer, name)

    with mock.patch.object(
        simulator_module.Simulator, "set_const", autospec=True
    ) as set_const:
      env = environment.Environment(
          device_layer=SimlessDeviceLayer(),  # pyrefly: ignore[bad-argument-type]
          task_layer=env._task_layer,
          observation_space_adapter=env._observation_space_adapter,
          environment_reset=PendulumReset(),
      )
      env.reset()
    set_const.assert_not_called()
    # One forward at startup and one after reset.
    self.assertLen(forward_calls, 2)


if __name__ == "__main__":
  absltest.main()
