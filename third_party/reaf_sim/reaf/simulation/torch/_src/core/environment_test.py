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
from collections.abc import Sequence
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
import numpy as np
from reaf.core import features_observer
from reaf.core import features_producer as core_features_producer
from reaf.core import logger as logger_module
from reaf.core import observation_space_adapter as core_observation_space_adapter
from reaf.simulation.torch._src.builders.flat_scene import config as flat_scene_config
from reaf.simulation.torch._src.common import functional_observation
from reaf.simulation.torch._src.core import device as device_module
from reaf.simulation.torch._src.core import device_layer as device_layer_module
from reaf.simulation.torch._src.core import discount_provider as discount_provider_module
from reaf.simulation.torch._src.core import environment
from reaf.simulation.torch._src.core import feature_producer as feature_producer_module
from reaf.simulation.torch._src.core import metric_provider as metric_provider_module
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

  def get_measurements(
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

  def get_measurements(
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


class HookRecordingObservationAdapter(PendulumObservationAdapter):
  """Records the adapter lifecycle hook calls."""

  def __init__(self, nworld: int = 1):
    super().__init__(nworld=nworld)
    self.on_step_step_counts: list[list[int]] = []
    self.begin_stepping_masks: list[list[bool]] = []
    self.begin_stepping_step_counts: list[list[int]] = []

  def on_step(self, features: Mapping[str, torch.Tensor]) -> None:
    self.on_step_step_counts.append(features["step_count"].tolist())

  def on_begin_stepping(
      self,
      features: Mapping[str, torch.Tensor],
      env_mask: torch.Tensor | None = None,
  ) -> None:
    assert env_mask is not None
    self.begin_stepping_masks.append(env_mask.tolist())
    self.begin_stepping_step_counts.append(features["step_count"].tolist())


class StepSumMetricProvider(metric_provider_module.MetricProvider):
  """Accumulates `scale * time` and step count per episode."""

  def __init__(self, nworld: int, xpu_device: str):
    self.scale = torch.ones(nworld, dtype=torch.float32, device=xpu_device)
    self._sums = torch.zeros(nworld, dtype=torch.float32, device=xpu_device)
    self._steps = torch.zeros(nworld, dtype=torch.int64, device=xpu_device)

  def update_metrics(self, features: Mapping[str, torch.Tensor]) -> None:
    self._sums += self.scale * features["time"]
    self._steps += 1

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    if env_mask is None:
      self._sums.zero_()
      self._steps.zero_()
    else:
      self._sums.masked_fill_(env_mask, 0.0)
      self._steps.masked_fill_(env_mask, 0)

  def get_episode_metrics(
      self, env_mask: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    denom = env_mask.sum().clamp(min=1).to(torch.float32)
    return {
        "time_sum": torch.where(env_mask, self._sums, 0.0).sum() / denom,
        "steps": (
            torch.where(env_mask, self._steps.to(torch.float32), 0.0).sum()
            / denom
        ),
    }


class CountingFeaturesObserver(features_observer.FeaturesObserver):
  """Counts `observe_features` calls."""

  def __init__(self):
    self.calls = 0

  @property
  def name(self) -> str:
    return "counting_observer"

  def observe_features(self, features: Mapping[str, Any]) -> None:
    del features  # Unused.
    self.calls += 1


class CountingLogger(logger_module.Logger):
  """Counts `record_measurements` and `record_features` calls."""

  def __init__(self):
    self.measurements_calls = 0
    self.features_calls = 0

  @property
  def name(self) -> str:
    return "counting_logger"

  def record_measurements(self, measurements: Mapping[str, Any]) -> None:
    del measurements  # Unused.
    self.measurements_calls += 1

  def record_features(self, features: Mapping[str, Any]) -> None:
    del features  # Unused.
    self.features_calls += 1


def _time_feature(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
  return features["time"].unsqueeze(-1)


class StepCounterProducer(feature_producer_module.FeatureProducer):
  """Counts control steps per world in `on_step`; `produce_features` is pure."""

  def __init__(self, nworld: int, xpu_device: str):
    self._count = torch.zeros(nworld, dtype=torch.int64, device=xpu_device)
    self.on_step_calls = 0
    self.begin_stepping_masks: list[torch.Tensor] = []

  @property
  def name(self) -> str:
    return "step_counter"

  def required_features_keys(self) -> set[str]:
    return {"time"}

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    return {"step_count": specs.Array(shape=(), dtype=np.int64)}

  def on_step(self, required_features: Mapping[str, torch.Tensor]) -> None:
    del required_features  # Unused.
    self.on_step_calls += 1
    self._count += 1

  def on_begin_stepping(
      self,
      required_features: Mapping[str, torch.Tensor],
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del required_features  # Unused.
    assert env_mask is not None
    self.begin_stepping_masks.append(env_mask.clone())
    self._count.masked_fill_(env_mask, 0)

  def produce_features(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    del required_features  # Unused.
    return {"step_count": self._count.clone()}


class DoubleTimeProducer(feature_producer_module.FeatureProducer):
  """Producer that keeps the default no-op lifecycle hooks."""

  @property
  def name(self) -> str:
    return "double_time"

  def required_features_keys(self) -> set[str]:
    return {"time"}

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    return {"double_time": specs.Array(shape=(), dtype=np.float32)}

  def produce_features(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    return {"double_time": 2.0 * required_features["time"]}


class CoreDoubleTimeProducer(core_features_producer.FeaturesProducer):
  """`DoubleTimeProducer` on the core base class, which has no hooks."""

  @property
  def name(self) -> str:
    return "double_time"

  def required_features_keys(self) -> set[str]:
    return {"time"}

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    return {"double_time": specs.Array(shape=(), dtype=np.float32)}

  def produce_features(  # pyrefly: ignore[bad-override]
      self, required_features: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    return {"double_time": 2.0 * required_features["time"]}


class RecordingLogger(logger_module.Logger):
  """Records clones of the given features on every `record_features` call."""

  def __init__(self, keys: Sequence[str]):
    self._keys = keys
    self.records: list[dict[str, torch.Tensor]] = []

  @property
  def name(self) -> str:
    return "recording_logger"

  def record_features(  # pyrefly: ignore[bad-override]
      self, features: Mapping[str, torch.Tensor]
  ) -> None:
    self.records.append({k: features[k].clone() for k in self._keys})


def _time_history_adapter(
    nworld: int,
    xpu_device: str,
    history_length: int = 2,
) -> functional_observation.FunctionalObservationAdapter:
  """Observes `time` as `[t - H, ..., t - 1, t]` (oldest to newest)."""
  return functional_observation.FunctionalObservationAdapter(
      observations={
          "time_history": flat_scene_config.ObsGroup(
              terms={
                  "time": (
                      flat_scene_config.Obs(_time_feature).configure(
                          history_length=history_length,
                          output_shape=(1,),
                          dtype="float32",
                      )
                  ),
              },
              enable_noise=False,
          ),
      },
      xpu_device=xpu_device,
      nworld=nworld,
  )


def _build_pendulum_env(
    nworld: int,
    xpu_device: str,
    timeouts: torch.Tensor,
    impl: types.Impl,
    home_qpos: float = 0.0,
    environment_reset_fn: environment.EnvironmentReset | None = None,
    auto_reset_mode: environment.AutoResetMode = environment.AutoResetMode.SYNC,
    metric_provider: metric_provider_module.MetricProvider | None = None,
    observation_adapter: (
        observation_space_adapter.ObservationSpaceAdapter | None
    ) = None,
    features_observers: Sequence[features_observer.FeaturesObserver] = (),
    loggers: Sequence[logger_module.Logger] = (),
    features_producers: Sequence[feature_producer_module.FeatureProducer] = (),
    use_default_observation_adapter: bool = False,
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
      features_producers=features_producers,
      reward_provider=reward,
      metric_provider=metric_provider,
      termination_checkers=terminations,
      discount_provider=discount,
      features_observers=features_observers,
      loggers=loggers,
      xpu_device=xpu_device,
      nworld=nworld,
  )
  reset = (
      environment_reset_fn
      if environment_reset_fn is not None
      else PendulumReset()
  )
  if use_default_observation_adapter:
    obs_adapter = None
  elif observation_adapter is not None:
    obs_adapter = observation_adapter
  else:
    obs_adapter = PendulumObservationAdapter(nworld=nworld)
  return environment.Environment(
      device_layer=dev_layer,
      task_layer=task_layer,
      observation_space_adapter=obs_adapter,
      environment_reset=reset,
      auto_reset_mode=auto_reset_mode,
  )


class WarpEnvironmentTest(parameterized.TestCase):
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

  @parameterized.parameters(
      environment.AutoResetMode.SYNC,
      environment.AutoResetMode.NO_SYNC,
      environment.AutoResetMode.DISABLED,
  )
  def test_metrics_staggered_auto_reset(self, mode):
    """Verify metric lifecycle across reset, partial reset, and auto-reset."""
    metrics = StepSumMetricProvider(nworld=2, xpu_device=self.XPU_DEVICE)
    metrics.scale[:] = torch.tensor([1.0, 2.0], device=self.XPU_DEVICE)
    timeouts = torch.tensor(
        [0.03, 0.07], dtype=torch.float32, device=self.XPU_DEVICE
    )
    env = _build_pendulum_env(
        nworld=2,
        xpu_device=self.XPU_DEVICE,
        timeouts=timeouts,
        impl=self.IMPL,
        auto_reset_mode=mode,
        metric_provider=metrics,
    )
    action = {
        "motor_ctrl": torch.zeros(
            1, dtype=torch.float32, device=self.XPU_DEVICE
        )
    }

    w0 = torch.tensor([True, False], device=self.XPU_DEVICE)
    w1 = torch.tensor([False, True], device=self.XPU_DEVICE)

    def assert_world_metrics(
        w0_sum: float, w1_sum: float, w0_steps: float, w1_steps: float
    ) -> None:
      ep_w0 = env.task_layer.get_episode_metrics(w0)
      ep_w1 = env.task_layer.get_episode_metrics(w1)
      torch.testing.assert_close(
          ep_w0["time_sum"], torch.tensor(w0_sum, device=self.XPU_DEVICE)
      )
      torch.testing.assert_close(
          ep_w1["time_sum"], torch.tensor(w1_sum, device=self.XPU_DEVICE)
      )
      torch.testing.assert_close(
          ep_w0["steps"], torch.tensor(w0_steps, device=self.XPU_DEVICE)
      )
      torch.testing.assert_close(
          ep_w1["steps"], torch.tensor(w1_steps, device=self.XPU_DEVICE)
      )

    env.reset()
    assert_world_metrics(0.0, 0.0, 0.0, 0.0)

    # Step 1 (0.02s, MID) -> Step 2 (0.04s): world 0 hits LAST, world 1 MID.
    env.step(action)
    ts = env.step(action)
    self.assertEqual(ts.last().tolist(), [True, False])
    assert_world_metrics(0.06, 0.12, 2.0, 2.0)

    # Change world 0 scale so Ep 2 (0.60) cannot match a stale Ep 1 (0.06).
    metrics.scale[0] = 10.0

    if mode == environment.AutoResetMode.DISABLED:
      # Manual partial reset between steps must wipe world 0 without updating 1.
      env.reset_with_options(
          options=env.default_reset_options(), env_mask=ts.last()
      )
      assert_world_metrics(0.0, 0.12, 0.0, 2.0)

      # Two more steps: world 0 (Ep 2, 2 steps = 0.60) and world 1 (Ep 1, 4
      # steps = 0.40) both hit LAST simultaneously -> mean = 0.50, steps = 3.0.
      env.step(action)
      ts = env.step(action)
      self.assertEqual(ts.last().tolist(), [True, True])
      assert_world_metrics(0.60, 0.40, 2.0, 4.0)
      ep_both = env.task_layer.get_episode_metrics(ts.last())
      torch.testing.assert_close(
          ep_both["time_sum"], torch.tensor(0.50, device=self.XPU_DEVICE)
      )
      torch.testing.assert_close(
          ep_both["steps"], torch.tensor(3.0, device=self.XPU_DEVICE)
      )
    else:
      # Step 3: world 0 auto-resets to FIRST (sums & steps wiped), world 1 at 3.
      ts = env.step(action)
      self.assertEqual(ts.first().tolist(), [True, False])
      assert_world_metrics(0.0, 0.24, 0.0, 3.0)

      # Step 4: world 0 on step 1 of Ep 2; world 1 hits LAST with all 4 steps.
      ts = env.step(action)
      self.assertEqual(ts.last().tolist(), [False, True])
      assert_world_metrics(0.20, 0.40, 1.0, 4.0)

      # Step 5: world 0 hits LAST (Ep 2) on the same step world 1 resets FIRST.
      ts = env.step(action)
      self.assertEqual(ts.last().tolist(), [True, False])
      self.assertEqual(ts.first().tolist(), [False, True])
      assert_world_metrics(0.60, 0.0, 2.0, 0.0)

    # Mid-episode full reset clears non-zero state; 1-step FIRST -> LAST works.
    timeouts[:] = 0.01
    env.reset()
    assert_world_metrics(0.0, 0.0, 0.0, 0.0)
    ts = env.step(action)
    self.assertEqual(ts.last().tolist(), [True, True])
    assert_world_metrics(0.20, 0.04, 1.0, 1.0)
    ep_one_step = env.task_layer.get_episode_metrics(ts.last())
    torch.testing.assert_close(
        ep_one_step["time_sum"], torch.tensor(0.12, device=self.XPU_DEVICE)
    )
    torch.testing.assert_close(
        ep_one_step["steps"], torch.tensor(1.0, device=self.XPU_DEVICE)
    )

  # TaskLayer lifecycle tests (b/581877129): `reset_with_options(env_mask)`
  # recomputes features and observations for every world, so state must only
  # advance in the lifecycle hooks, and only for `env_mask` worlds on reset.

  def _zero_action(self) -> dict[str, torch.Tensor]:
    return {
        "motor_ctrl": torch.zeros(
            1, dtype=torch.float32, device=self.XPU_DEVICE
        )
    }

  def _assert_time_history(
      self, observation: torch.Tensor, expected_steps: list[list[int]]
  ) -> None:
    """Asserts `observation` equals `expected_steps * control_timestep`."""
    torch.testing.assert_close(
        observation,
        torch.tensor(
            expected_steps, dtype=torch.float32, device=self.XPU_DEVICE
        )
        * 0.02,
        atol=1e-5,
        rtol=0.0,
    )

  def _check_history_without_resets(
      self, mode: environment.AutoResetMode
  ) -> None:
    """History holds the last H distinct steps when no world resets."""
    nworld = 2
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=torch.full((nworld,), 10.0, device=self.XPU_DEVICE),
        impl=self.IMPL,
        auto_reset_mode=mode,
        observation_adapter=_time_history_adapter(
            nworld=nworld, xpu_device=self.XPU_DEVICE
        ),
    )
    ts = env.reset()
    self._assert_time_history(ts.observation["time_history"], [[0, 0, 0]] * 2)
    for k in range(1, 5):
      ts = env.step(self._zero_action())
      self._assert_time_history(
          ts.observation["time_history"],
          [[max(k - 2, 0), max(k - 1, 0), k]] * 2,
      )

  def _check_history_partial_reset(
      self, mode: environment.AutoResetMode
  ) -> None:
    """A partial reset leaves the history of non-reset worlds unchanged."""
    nworld = 2
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        # World 0 is LAST after step 2 (0.04s) and resets on step 3.
        timeouts=torch.tensor([0.03, 10.0], device=self.XPU_DEVICE),
        impl=self.IMPL,
        auto_reset_mode=mode,
        observation_adapter=_time_history_adapter(
            nworld=nworld, xpu_device=self.XPU_DEVICE
        ),
    )
    env.reset()
    env.step(self._zero_action())
    ts = env.step(self._zero_action())
    self.assertEqual(ts.last().tolist(), [True, False])
    self._assert_time_history(
        ts.observation["time_history"], [[0, 1, 2], [0, 1, 2]]
    )

    ts = env.step(self._zero_action())
    self.assertEqual(ts.first().tolist(), [True, False])
    self._assert_time_history(
        ts.observation["time_history"], [[0, 0, 0], [1, 2, 3]]
    )

    ts = env.step(self._zero_action())
    self._assert_time_history(
        ts.observation["time_history"], [[0, 0, 1], [2, 3, 4]]
    )

  def test_no_sync_matches_sync(self):
    """`NO_SYNC` and `SYNC` produce the same timesteps."""
    nworld = 2
    timesteps = {}
    for mode in (
        environment.AutoResetMode.SYNC,
        environment.AutoResetMode.NO_SYNC,
    ):
      env = _build_pendulum_env(
          nworld=nworld,
          xpu_device=self.XPU_DEVICE,
          timeouts=torch.tensor([0.03, 0.07], device=self.XPU_DEVICE),
          impl=self.IMPL,
          auto_reset_mode=mode,
          observation_adapter=_time_history_adapter(
              nworld=nworld, xpu_device=self.XPU_DEVICE
          ),
      )
      ts = env.reset()
      # Clone: `reset_with_options` splices into the last observation in place.
      trajectory = [(
          ts.step_type.clone(),
          ts.observation["time_history"].clone(),
      )]
      for _ in range(8):
        ts = env.step(self._zero_action())
        trajectory.append(
            (ts.step_type.clone(), ts.observation["time_history"].clone())
        )
      timesteps[mode] = trajectory

    sync = timesteps[environment.AutoResetMode.SYNC]
    no_sync = timesteps[environment.AutoResetMode.NO_SYNC]
    for step, ((sync_type, sync_obs), (no_sync_type, no_sync_obs)) in enumerate(
        zip(sync, no_sync)
    ):
      torch.testing.assert_close(
          no_sync_type, sync_type, msg=f"step_type differs at step {step}"
      )
      torch.testing.assert_close(
          no_sync_obs, sync_obs, msg=f"observation differs at step {step}"
      )

  def _check_observers_and_loggers_once_per_step(
      self, mode: environment.AutoResetMode
  ) -> None:
    """Observers and loggers see each step once when no world resets."""
    nworld = 2
    observer = CountingFeaturesObserver()
    logger = CountingLogger()
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        timeouts=torch.full((nworld,), 10.0, device=self.XPU_DEVICE),
        impl=self.IMPL,
        auto_reset_mode=mode,
        features_observers=[observer],
        loggers=[logger],
    )
    env.reset()
    observer.calls = 0
    logger.measurements_calls = 0
    logger.features_calls = 0

    num_steps = 3
    for _ in range(num_steps):
      env.step(self._zero_action())

    self.assertEqual(observer.calls, num_steps)
    self.assertEqual(logger.measurements_calls, num_steps)
    self.assertEqual(logger.features_calls, num_steps)

  @parameterized.parameters(
      environment.AutoResetMode.SYNC,
      environment.AutoResetMode.DISABLED,
  )
  def test_observation_history_without_resets(self, mode):
    self._check_history_without_resets(mode)

  def test_observation_history_without_resets_no_sync(self):
    self._check_history_without_resets(environment.AutoResetMode.NO_SYNC)

  def test_observation_history_partial_reset_sync(self):
    self._check_history_partial_reset(environment.AutoResetMode.SYNC)

  def test_observation_history_partial_reset_no_sync(self):
    self._check_history_partial_reset(environment.AutoResetMode.NO_SYNC)

  def test_observers_and_loggers_run_once_per_step_sync(self):
    self._check_observers_and_loggers_once_per_step(
        environment.AutoResetMode.SYNC
    )

  def test_observers_and_loggers_run_once_per_step_no_sync(self):
    self._check_observers_and_loggers_once_per_step(
        environment.AutoResetMode.NO_SYNC
    )

  @parameterized.parameters(
      environment.AutoResetMode.SYNC,
      environment.AutoResetMode.NO_SYNC,
  )
  def test_records_on_partial_reset(self, mode):
    """One record per returned timestep, with FIRST features for reset worlds."""
    logger = RecordingLogger(keys=("time",))
    env = _build_pendulum_env(
        nworld=2,
        xpu_device=self.XPU_DEVICE,
        # World 0 is LAST after step 2 (0.04s) and resets on step 3.
        timeouts=torch.tensor([0.03, 10.0], device=self.XPU_DEVICE),
        impl=self.IMPL,
        auto_reset_mode=mode,
        loggers=[logger],
    )
    env.reset()
    for _ in range(4):
      env.step(self._zero_action())

    # Reset + 4 steps. On step 3, world 0 is recorded with its reset features.
    self._assert_time_history(
        torch.stack([r["time"] for r in logger.records]).float(),
        [[0, 0], [1, 1], [2, 2], [0, 3], [1, 4]],
    )

  def test_records_on_manual_partial_reset(self):
    """`DISABLED`: a manual partial reset records all worlds once."""
    logger = RecordingLogger(keys=("time",))
    env = _build_pendulum_env(
        nworld=2,
        xpu_device=self.XPU_DEVICE,
        # World 0 is LAST after step 2 (0.04s).
        timeouts=torch.tensor([0.03, 10.0], device=self.XPU_DEVICE),
        impl=self.IMPL,
        auto_reset_mode=environment.AutoResetMode.DISABLED,
        loggers=[logger],
    )
    env.reset()
    env.step(self._zero_action())
    ts = env.step(self._zero_action())
    self.assertEqual(ts.last().tolist(), [True, False])
    env.reset_with_options(
        options=env.default_reset_options(), env_mask=ts.last()
    )
    for _ in range(2):
      env.step(self._zero_action())

    # Reset + 2 steps + manual reset of world 0 + 2 steps.
    self._assert_time_history(
        torch.stack([r["time"] for r in logger.records]).float(),
        [[0, 0], [1, 1], [2, 2], [0, 2], [1, 3], [2, 4]],
    )

  @parameterized.parameters(
      environment.AutoResetMode.SYNC,
      environment.AutoResetMode.NO_SYNC,
      environment.AutoResetMode.DISABLED,
  )
  def test_producer_hooks_and_records_on_partial_reset(self, mode):
    """Hooks run once per step / per reset world; records match timesteps."""
    nworld = 2
    counter = StepCounterProducer(nworld=nworld, xpu_device=self.XPU_DEVICE)
    adapter = HookRecordingObservationAdapter(nworld=nworld)
    logger = RecordingLogger(keys=("time", "step_count", "double_time"))
    env = _build_pendulum_env(
        nworld=nworld,
        xpu_device=self.XPU_DEVICE,
        # World 0 is LAST after step 2 (0.04s) and auto-resets on step 3.
        timeouts=torch.tensor([0.03, 10.0], device=self.XPU_DEVICE),
        impl=self.IMPL,
        auto_reset_mode=mode,
        observation_adapter=adapter,
        features_producers=[counter, DoubleTimeProducer()],
        loggers=[logger],
    )
    env.reset()
    env.step(self._zero_action())
    ts = env.step(self._zero_action())
    if mode == environment.AutoResetMode.DISABLED:
      # Manually reset world 0 between steps 2 and 3.
      env.reset_with_options(
          options=env.default_reset_options(), env_mask=ts.last()
      )
    for _ in range(2):
      env.step(self._zero_action())

    self.assertEqual(counter.on_step_calls, 4)
    begin_stepping_masks = [m.tolist() for m in counter.begin_stepping_masks]
    if mode == environment.AutoResetMode.NO_SYNC:
      # NO_SYNC resets every step, with an all-False mask when no world is done.
      self.assertEqual(
          begin_stepping_masks,
          [
              [True, True],
              [False, False],
              [False, False],
              [True, False],
              [False, False],
          ],
      )
    else:
      self.assertEqual(begin_stepping_masks, [[True, True], [True, False]])
    # The adapter hooks run with the same masks, after the producers: reset
    # worlds already have their reset step count.
    self.assertEqual(adapter.begin_stepping_masks, begin_stepping_masks)
    for mask, counts in zip(
        adapter.begin_stepping_masks, adapter.begin_stepping_step_counts
    ):
      self.assertEqual([c for m, c in zip(mask, counts) if m], [0] * sum(mask))
    if mode == environment.AutoResetMode.DISABLED:
      # World 0 was reset between steps 2 and 3.
      expected_on_step_counts = [[1, 1], [2, 2], [1, 3], [2, 4]]
    else:
      # World 0 auto-resets at the end of step 3, after `on_step`.
      expected_on_step_counts = [[1, 1], [2, 2], [3, 3], [1, 4]]
    self.assertEqual(adapter.on_step_step_counts, expected_on_step_counts)

    if mode == environment.AutoResetMode.DISABLED:
      # Reset + 2 steps + manual reset of world 0 + 2 steps.
      expected_steps = [[0, 0], [1, 1], [2, 2], [0, 2], [1, 3], [2, 4]]
    else:
      # Reset + 4 steps. On step 3, world 0 is recorded with its reset (FIRST)
      # features, world 1 with its step ones.
      expected_steps = [[0, 0], [1, 1], [2, 2], [0, 3], [1, 4]]
    self.assertEqual(
        [r["step_count"].tolist() for r in logger.records], expected_steps
    )
    self._assert_time_history(
        torch.stack([r["time"] for r in logger.records]).float(),
        expected_steps,
    )
    for record in logger.records:
      torch.testing.assert_close(record["double_time"], 2.0 * record["time"])

  def test_default_observation_adapter(self):
    """Without an adapter, observations are the task layer features."""
    env = _build_pendulum_env(
        nworld=2,
        xpu_device=self.XPU_DEVICE,
        # World 0 is LAST after step 2 (0.04s) and auto-resets on step 3.
        timeouts=torch.tensor([0.03, 10.0], device=self.XPU_DEVICE),
        impl=self.IMPL,
        use_default_observation_adapter=True,
    )
    env.reset()
    env.step(self._zero_action())
    env.step(self._zero_action())
    ts = env.step(self._zero_action())
    self.assertEqual(ts.first().tolist(), [True, False])
    torch.testing.assert_close(
        ts.observation["time"],
        torch.tensor([0.0, 0.06], device=self.XPU_DEVICE),
        atol=1e-5,
        rtol=0.0,
        check_dtype=False,
    )

  def test_environment_rejects_core_observation_space_adapter(self):
    core_adapter = mock.create_autospec(
        core_observation_space_adapter.ObservationSpaceAdapter, instance=True
    )
    with self.assertRaisesRegex(TypeError, "observation space adapter"):
      _build_pendulum_env(
          nworld=1,
          xpu_device=self.XPU_DEVICE,
          timeouts=torch.ones(1, device=self.XPU_DEVICE),
          impl=self.IMPL,
          observation_adapter=core_adapter,
      )

  def test_task_layer_rejects_core_features_producer(self):
    with self.assertRaisesRegex(TypeError, "CoreDoubleTimeProducer"):
      task_layer_module.TaskLayer(
          commands_processors=[],
          features_producers=[CoreDoubleTimeProducer()],  # pyrefly: ignore[bad-argument-type]
          termination_checkers=[],
          xpu_device=self.XPU_DEVICE,
          nworld=1,
      )


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
