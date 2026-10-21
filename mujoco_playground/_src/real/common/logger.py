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

"""Episode logging to MCAP using the safari-sdk episodic logger."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import time
from typing import Any

import dm_env
from dm_env import specs
from etils import epath
from gdm_robotics.interfaces import types as gdmr_types
from mujoco_playground._src.real.common.instruction_feature import INSTRUCTION_KEY
import numpy as np


def _logged_spec(spec: Any) -> Any:
  """A spec that an episode stores.

  MCAP has no boolean array type and logs should not have a batch dim.

  Args:
    spec: The spec the environment declares.

  Returns:
    The same spec, unbatched, with any boolean dtype widened to uint8.
  """
  shape = spec.shape[1:] if spec.shape and spec.shape[0] == 1 else spec.shape
  if isinstance(spec, specs.StringArray):
    return specs.StringArray(shape=shape, name=spec.name)
  dtype = np.uint8 if spec.dtype in (bool, np.bool_) else spec.dtype
  if isinstance(spec, specs.BoundedArray):
    return specs.BoundedArray(
        shape=shape,
        dtype=dtype,
        minimum=spec.minimum,
        maximum=spec.maximum,
        name=spec.name,
    )
  return specs.Array(shape=shape, dtype=dtype, name=spec.name)


def _bounded(spec: specs.Array) -> specs.BoundedArray:
  """The spec as a `BoundedArray`, which the SDK requires of actions."""
  if isinstance(spec, specs.BoundedArray):
    return spec
  return specs.BoundedArray(
      shape=spec.shape,
      dtype=spec.dtype,
      minimum=-np.inf,
      maximum=np.inf,
      name=spec.name,
  )


def _to_numpy(value: Any) -> Any:
  """Brings torch tensors back to the host, without importing torch here."""
  if hasattr(value, "detach") and hasattr(value, "cpu"):
    return value.detach().cpu().numpy()
  if isinstance(value, Mapping):
    return {k: _to_numpy(v) for k, v in value.items()}
  return value


def _conform(value: Any, spec: Any) -> Any:
  """Casts a value, or a tree of them, into what `spec` declares."""
  if value is None:  # A reset timestep carries no reward or discount.
    return None
  if isinstance(spec, specs.StringArray):
    return np.asarray(str(value), dtype=object)
  value = _to_numpy(value)
  if isinstance(value, Mapping):
    return {k: _conform(v, spec[k]) for k, v in value.items()}
  array = np.asarray(value)
  if array.shape == (1,) + spec.shape:
    array = array[0]  # The environment's single world.
  # Anything still the wrong shape is a real disagreement, and the SDK's own
  # validation says so with the key name attached. `astype` rather than
  # `ascontiguousarray` because the latter promotes a scalar to shape (1,).
  return array.astype(spec.dtype, order="C")


class _EpisodicLogger:
  """An episodic logger amenable to runloop based collection."""

  def __init__(
      self,
      logger: Any,
      *,
      timestep_spec: gdmr_types.TimeStepSpec,
      action_spec: Mapping[str, specs.Array],
      policy_extra_spec: Mapping[str, specs.Array],
      timestamp_key: str,
      instruction: str,
  ):
    self._logger = logger
    self._spec = timestep_spec
    self._action_spec = action_spec
    self._policy_extra_spec = policy_extra_spec
    self._timestamp_key = timestamp_key
    self._instruction = instruction

  def _conform_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    observation = dict(timestep.observation)
    observation.setdefault(INSTRUCTION_KEY, self._instruction)
    observation.setdefault(self._timestamp_key, time.time_ns())

    spec = self._spec.observation
    assert isinstance(spec, Mapping)
    missing = spec.keys() - observation.keys()
    if missing:
      raise ValueError(
          f"Observation is missing {sorted(missing)}, which the episode"
          f" declares. It has {sorted(observation)}."
      )

    conformed = {key: _conform(observation[key], spec[key]) for key in spec}
    # The SDK stamps each MCAP message with this, as a Python int.
    conformed[self._timestamp_key] = int(conformed[self._timestamp_key])

    return timestep._replace(
        reward=_conform(timestep.reward, self._spec.reward),
        discount=_conform(timestep.discount, self._spec.discount),
        observation=conformed,
    )

  def reset(self, timestep: dm_env.TimeStep) -> None:
    self._logger.reset(self._conform_timestep(timestep))

  def record_action_and_next_timestep(
      self,
      action: Any,
      next_timestep: dm_env.TimeStep,
      policy_extra: Any = None,
  ) -> None:
    self._logger.record_action_and_next_timestep(
        action=_conform(action, self._action_spec),
        next_timestep=self._conform_timestep(next_timestep),
        policy_extra=_conform(policy_extra or {}, self._policy_extra_spec),
    )

  def write(self) -> None:
    self._logger.write()

  def stop(self) -> None:
    self._logger.stop()

  def close(self) -> None:
    self.stop()


def create_logger(
    *,
    timestep_spec: gdmr_types.TimeStepSpec,
    action_spec: Mapping[str, specs.Array],
    policy_extra_spec: Mapping[str, specs.Array],
    image_observation_keys: Sequence[str],
    timestamp_key: str,
    task_id: str,
    agent_id: str,
    output_directory: epath.PathLike,
    control_timestep_seconds: float,
    outcome: Callable[[], str | None] | None = None,
) -> _EpisodicLogger:
  """Creates an episodic MCAP logger for a real or deployed robot.

  The result implements the logger interface
  `gdm_robotics.runtime.runloop.Runloop` expects: `reset`,
  `record_action_and_next_timestep`, `write`, `stop` and `close`.

  Args:
    timestep_spec: What the environment says its timesteps look like. A leading
      world dimension of 1 is dropped; the episode records one robot.
    action_spec: What the policy says its actions look like, likewise.
    policy_extra_spec: What else the policy emits alongside its action.
    image_observation_keys: Which observations are camera frames, to be stored
      as jpeg. Everything else is recorded as proprioception, so pass the empty
      sequence for a policy whose "camera" observation is really a float tensor.
    timestamp_key: The observation holding each step's wall-clock time in
      nanoseconds. If the robot does not measure one, it is stamped here.
    task_id: What is being done. Also the default instruction.
    agent_id: Which robot is doing it.
    output_directory: Where episodes are written, under a YYYY/MM/DD tree.
    control_timestep_seconds: The rate the episode was recorded at.
    outcome: Returns "success", "failure", "invalid" or None, and is called at
      the end of each episode to tag it. Episodes go out "unlabeled" without it.

  Returns:
    The logger.
  """
  try:
    from safari_sdk.logging.python import episodic_logger  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top
    from safari_sdk.logging.python import session_metadata  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top
  except ImportError as exc:
    raise ImportError(
        "safari-sdk is required for MCAP logging. Run: uv sync --extra real"
    ) from exc

  directory = epath.Path(output_directory)
  directory.mkdir(parents=True, exist_ok=True)

  assert isinstance(timestep_spec.observation, Mapping)
  observation_spec = {
      key: _logged_spec(spec) for key, spec in timestep_spec.observation.items()
  }
  # Every episode carries these two, measured or not.
  observation_spec.setdefault(INSTRUCTION_KEY, specs.StringArray(shape=()))
  observation_spec.setdefault(
      timestamp_key, specs.Array(shape=(), dtype=np.int64)
  )

  logged_timestep_spec = gdmr_types.TimeStepSpec(
      step_type=timestep_spec.step_type,
      reward=_logged_spec(timestep_spec.reward),
      discount=_logged_spec(timestep_spec.discount),
      observation=observation_spec,
  )
  logged_action_spec = {
      key: _bounded(_logged_spec(spec)) for key, spec in action_spec.items()
  }
  outcome_fn = outcome or (lambda: None)

  raw_logger = episodic_logger.EpisodicLogger.create(
      episodic_logger.EpisodicLoggerConfig(
          agent_id=agent_id,
          task_id=task_id,
          output_directory=str(directory),
          image_observation_keys=list(image_observation_keys),
          proprioceptive_observation_keys=[
              key
              for key in observation_spec
              if key not in image_observation_keys
              and key not in (INSTRUCTION_KEY, timestamp_key)
          ],
          timestep_spec=logged_timestep_spec,
          action_spec=logged_action_spec,
          policy_extra_spec=policy_extra_spec,
          timestamp_key=timestamp_key,
          metadata_config=session_metadata.SessionMetadataConfig(
              embodiment_version=agent_id,
              control_timestep_seconds=control_timestep_seconds,
              is_success_provider=lambda: outcome_fn() == "success",
              dynamic_episode_taggers=[lambda: [outcome_fn() or "unlabeled"]],
              dynamic_metadata_provider=lambda: {
                  "task_id": task_id,
                  "agent_id": agent_id,
              },
          ),
      )
  )
  return _EpisodicLogger(
      raw_logger,
      timestep_spec=logged_timestep_spec,
      action_spec=logged_action_spec,
      policy_extra_spec=policy_extra_spec,
      timestamp_key=timestamp_key,
      instruction=task_id,
  )
