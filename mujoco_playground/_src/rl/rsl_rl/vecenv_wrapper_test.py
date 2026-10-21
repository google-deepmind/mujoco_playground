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

"""Tests for RslRlVecEnvWrapper."""

from collections.abc import Mapping

from absl.testing import absltest
from dm_env import specs
import mujoco
from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
import numpy as np
from reaf.core import observation_space_adapter
from reaf.simulation.torch import clock_device
from reaf.simulation.torch import CompositeResetter
from reaf.simulation.torch import default_discount_provider
from reaf.simulation.torch import device as device_module
from reaf.simulation.torch import device_layer as device_layer_module
from reaf.simulation.torch import environment
from reaf.simulation.torch import NanTerminationResetter
from reaf.simulation.torch import reward_provider
from reaf.simulation.torch import simulator as simulator_module
from reaf.simulation.torch import task_layer as task_layer_module
from reaf.simulation.torch import termination_checker
from reaf.simulation.torch import types
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

  def __init__(self, nworld: int, xpu_device: str = "cpu"):
    self._nworld = nworld
    self._xpu_device = xpu_device

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
      env_mask=None,
  ) -> None:
    del simulator, env_mask

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    del simulator


class ConstantRewardProvider(reward_provider.RewardProvider):
  """Returns zero reward."""

  def __init__(self, nworld: int, xpu_device: str):
    self._nworld = nworld
    self._xpu_device = xpu_device

  def name(self) -> str:
    return "constant_reward"

  def required_features_keys(self) -> set[str]:
    return set()

  def reward_spec(self) -> specs.Array:
    return specs.Array(shape=(1,), dtype=np.float32)

  def compute_reward(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> torch.Tensor:
    del required_features
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

  def __init__(self, sim_device):
    self._sim_device = sim_device

  def do_reset(self, config, env_mask) -> None:
    self._sim_device.reset(env_mask)


class PendulumObservationAdapter(
    observation_space_adapter.ObservationSpaceAdapter
):
  """Maps pendulum features to observations."""

  def __init__(self, nworld: int = 1):
    self._nworld = nworld

  def observations_from_features(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    return {
        "joint_pos": required_features["joint_pos"].float(),
        "time": required_features["time"].unsqueeze(-1).float(),
    }

  def observation_spec(self) -> Mapping[str, specs.Array]:
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
) -> environment.Environment:
  """Builds a simple pendulum environment for testing."""
  mj_spec = mujoco.MjSpec.from_string(_PENDULUM_XML)
  control_timestep = 0.02

  pendulum = PendulumDevice(nworld=nworld, xpu_device=xpu_device)
  clock = clock_device.ClockDevice(
      name="clock",
      time_output_name="time",
  )

  sim = simulator_module.Simulator(
      mj_spec=mj_spec,
      impl=impl,
      nworld=nworld,
      xpu_device=xpu_device,
  )
  device_layer = device_layer_module.DeviceLayer(
      simulator=sim,
      control_timestep=control_timestep,
      devices=[pendulum, clock],
  )

  rew = ConstantRewardProvider(nworld=nworld, xpu_device=xpu_device)
  discount = default_discount_provider.DefaultDiscountProvider(
      nworld=nworld, xpu_device=xpu_device
  )
  terminations = [
      StaggeredTerminationChecker(
          timeouts=timeouts,
          time_feature_name="time",
      ),
  ]
  task_layer = task_layer_module.TaskLayer(
      commands_processors=[],
      features_producers=[],
      reward_provider=rew,
      termination_checkers=terminations,
      discount_provider=discount,
      features_observers=[],
      loggers=[],
      xpu_device=xpu_device,
      nworld=nworld,
  )
  reset = PendulumReset(device_layer)
  obs_adapter = PendulumObservationAdapter(nworld=nworld)
  return environment.Environment(
      device_layer=device_layer,
      task_layer=task_layer,
      observation_space_adapter=obs_adapter,
      environment_reset=reset,
  )


class VecEnvWrapperTest(absltest.TestCase):
  """Tests for RslRlVecEnvWrapper."""

  IMPL = types.Impl.CPP
  DEVICE = "cpu"

  def test_gym_style_auto_reset(self):
    """Verify gym-style auto-reset: done envs get S_0 observations."""
    nworld = 1
    # Timeout at 0.04s.
    timeouts = torch.tensor([0.04], dtype=torch.float32, device=self.DEVICE)
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
    )

    wrapper = vecenv_wrapper.RslRlVecEnvWrapper(env, max_episode_length=1000)

    # Step 3 times to trigger termination (0.06s > 0.04s).
    action = torch.zeros(nworld, 1, dtype=torch.float32, device=self.DEVICE)
    for _ in range(3):
      obs, _, dones, _ = wrapper.step(action)

    # Env should be done.
    self.assertEqual(dones.item(), 1)
    self.assertEqual(obs["time"].item(), 0.0)

  def test_episode_length_buf_tracking(self):
    """Verify episode_length_buf tracks steps and resets correctly."""
    nworld = 1
    timeouts = torch.tensor([10.0], dtype=torch.float32, device=self.DEVICE)
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
    )

    wrapper = vecenv_wrapper.RslRlVecEnvWrapper(env, max_episode_length=1000)

    # Initially episode_length_buf should be 0.
    self.assertTrue(torch.all(wrapper.episode_length_buf == 0))

    # Step 3 times.
    action = torch.zeros(nworld, 1, dtype=torch.float32, device=self.DEVICE)
    for _ in range(3):
      wrapper.step(action)

    # Env should have episode_length_buf = 3 after 3 steps.
    self.assertEqual(wrapper.episode_length_buf.item(), 3)

  def test_episode_length_buf_writes_time(self):
    """Verify setting episode_length_buf overwrites data.time."""
    nworld = 1
    timeouts = torch.full(
        (nworld,), 1.0, dtype=torch.float32, device=self.DEVICE
    )
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
    )

    wrapper = vecenv_wrapper.RslRlVecEnvWrapper(env, max_episode_length=50)
    control_timestep = wrapper._control_timestep

    # Get access to data.time for verification.
    sim_device = wrapper.unwrapped.device_layer

    # Initially data.time should be 0.
    self.assertTrue(torch.all(sim_device.sim.data.time == 0.0))

    # Set episode_length_buf to random values (simulating RSL-RL's
    length = torch.tensor([10], device=self.DEVICE)
    wrapper.episode_length_buf = length

    # Verify episode_length_buf was set.
    torch.testing.assert_close(wrapper.episode_length_buf, length)

    # Verify data.time was overwritten to episode_length_buf * control_timestep.
    expected_time = length.to(torch.float64) * control_timestep
    actual_time = torch.as_tensor(sim_device.sim.data.time)
    torch.testing.assert_close(
        actual_time,
        expected_time,
        atol=1e-6,
        rtol=1e-6,
    )

  def test_nan_termination_resetter_metrics_logged(self):
    """Verify NanTerminationResetter terminates NaN envs and logs metrics."""
    nworld = 1
    timeouts = torch.full(
        (nworld,), 10.0, dtype=torch.float32, device=self.DEVICE
    )
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
    )
    comp = CompositeResetter(
        [PendulumReset(env.device_layer), NanTerminationResetter()]
    )
    comp.set_env(env)
    env._environment_reset = comp  # pylint: disable=protected-access

    wrapper = vecenv_wrapper.RslRlVecEnvWrapper(env, max_episode_length=100)
    wrapper.unwrapped.device_layer.sim.mj_model.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_AUTORESET
    )
    # Inject NaN into world 0 velocity so it remains NaN after sim step.
    wrapper.unwrapped.device_layer.sim.data.qvel[0, 0] = float("nan")
    action = torch.zeros(nworld, 1, dtype=torch.float32, device=self.DEVICE)
    _, _, dones, extras = wrapper.step(action)

    torch.testing.assert_close(
        dones, torch.tensor([1], dtype=torch.long, device=self.DEVICE)
    )
    self.assertIn("Metrics/nan_termination/nan_terminated", extras["log"])
    torch.testing.assert_close(
        extras["log"]["Metrics/nan_termination/nan_terminated"],
        torch.tensor(1.0, dtype=torch.float32, device=self.DEVICE),
    )


if __name__ == "__main__":
  absltest.main()
