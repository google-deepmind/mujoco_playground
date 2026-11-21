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

"""Tests for episodic MCAP logging."""

import pathlib
import subprocess
import sys
import tempfile

from absl.testing import absltest
import dm_env
from dm_env import specs
from etils import epath
from gdm_robotics.interfaces import environment as gdmr_env
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
from gdm_robotics.runtime import runloop as runloop_lib
from mujoco_playground._src.real.common import log_reader
from mujoco_playground._src.real.common import logger
from mujoco_playground._src.real.common import view_log
import numpy as np
import torch

try:
  from safari_sdk.logging.python import episodic_logger  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top,unused-import

  _HAS_SAFARI_SDK = True
except ImportError:
  _HAS_SAFARI_SDK = False

_TIMESTAMP = "timestamp"


def timestep_spec(**observation) -> gdmr_types.TimeStepSpec:
  return gdmr_types.TimeStepSpec(
      step_type=specs.BoundedArray(
          shape=(), dtype=np.int32, minimum=0, maximum=2
      ),
      reward=specs.Array(shape=(), dtype=np.float32),
      discount=specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0.0, maximum=1.0
      ),
      observation=observation,
  )


def mcaps_in(directory: str) -> list[pathlib.Path]:
  return list(pathlib.Path(directory).glob("**/*.mcap"))


class LoggedSpecTest(absltest.TestCase):
  """What an episode declares, given what the environment declares."""

  def test_drops_the_leading_world_dimension(self):
    spec = logger._logged_spec(specs.Array(shape=(1, 6), dtype=np.float32))  # pylint: disable=protected-access
    self.assertEqual(spec.shape, (6,))

  def test_keeps_a_leading_dimension_that_is_not_a_single_world(self):
    spec = logger._logged_spec(specs.Array(shape=(16, 16, 3), dtype=np.uint8))  # pylint: disable=protected-access
    self.assertEqual(spec.shape, (16, 16, 3))

  def test_widens_booleans_mcap_cannot_store(self):
    spec = logger._logged_spec(specs.Array(shape=(2,), dtype=bool))  # pylint: disable=protected-access
    self.assertEqual(spec.dtype, np.uint8)

  def test_keeps_bounds(self):
    spec = logger._logged_spec(  # pylint: disable=protected-access
        specs.BoundedArray(
            shape=(1, 3), dtype=np.float32, minimum=-1.0, maximum=1.0
        )
    )
    self.assertIsInstance(spec, specs.BoundedArray)
    self.assertEqual(spec.shape, (3,))
    np.testing.assert_array_equal(spec.maximum, 1.0)


@absltest.skipUnless(
    _HAS_SAFARI_SDK, "safari-sdk is required for MCAP logging."
)
class LoggerTest(absltest.TestCase):

  def test_writes_an_episode_of_numpy(self):
    spec = timestep_spec(**{
        _TIMESTAMP: specs.Array(shape=(), dtype=np.int64),
        "wrist_camera": specs.Array(shape=(16, 16, 3), dtype=np.uint8),
        "qpos": specs.Array(shape=(6,), dtype=np.float32),
    })
    action_spec = {
        "act": specs.BoundedArray(
            shape=(6,), dtype=np.float32, minimum=-1.0, maximum=1.0
        )
    }

    def observation(stamp: int) -> dict[str, object]:
      return {
          _TIMESTAMP: stamp,
          "wrist_camera": np.zeros((16, 16, 3), dtype=np.uint8),
          "qpos": np.zeros(6, dtype=np.float32),
      }

    with tempfile.TemporaryDirectory() as tmp_dir:
      episode = logger.create_logger(
          timestep_spec=spec,
          action_spec=action_spec,
          policy_extra_spec={},
          image_observation_keys=["wrist_camera"],
          timestamp_key=_TIMESTAMP,
          task_id="test_task",
          agent_id="test_agent",
          output_directory=tmp_dir,
          control_timestep_seconds=0.02,
          outcome=lambda: "success",
      )
      episode.reset(dm_env.restart(observation(100)))
      episode.record_action_and_next_timestep(
          {"act": np.zeros(6, dtype=np.float32)},
          dm_env.termination(reward=1.0, observation=observation(200)),
          {},
      )
      episode.write()
      episode.stop()

      self.assertLen(mcaps_in(tmp_dir), 1)

  def test_reads_back_an_episode(self):
    spec = timestep_spec(**{
        _TIMESTAMP: specs.Array(shape=(), dtype=np.int64),
        "qpos": specs.Array(shape=(6,), dtype=np.float32),
    })
    action_spec = {
        "act": specs.BoundedArray(
            shape=(6,), dtype=np.float32, minimum=-1.0, maximum=1.0
        )
    }
    qpos = np.arange(6, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
      episode = logger.create_logger(
          timestep_spec=spec,
          action_spec=action_spec,
          policy_extra_spec={},
          image_observation_keys=[],
          timestamp_key=_TIMESTAMP,
          task_id="test_task",
          agent_id="test_agent",
          output_directory=tmp_dir,
          control_timestep_seconds=0.02,
      )
      episode.reset(dm_env.restart({_TIMESTAMP: 100, "qpos": qpos}))
      episode.record_action_and_next_timestep(
          {"act": np.ones(6, dtype=np.float32)},
          dm_env.termination(
              reward=1.0, observation={_TIMESTAMP: 200, "qpos": qpos}
          ),
          {},
      )
      episode.write()
      episode.stop()
      (path,) = mcaps_in(tmp_dir)
      messages = list(log_reader.iter_features(epath.Path(path)))
      app_id, shapes = view_log.session_info(epath.Path(path))
      # In a fresh process, because other tests may have loaded TensorFlow.
      reads_without_tensorflow = subprocess.run(
          [
              sys.executable,
              "-c",
              (
                  "import sys\n"
                  "from etils import epath\n"
                  "from mujoco_playground._src.real.common import log_reader\n"
                  "from mujoco_playground._src.real.common import view_log\n"
                  f"path = epath.Path({str(path)!r})\n"
                  "list(log_reader.iter_features(path))\n"
                  "view_log.session_info(path)\n"
                  "sys.exit('tensorflow' in sys.modules)\n"
              ),
          ],
          check=False,
      )
      self.assertEqual(reads_without_tensorflow.returncode, 0)

    timesteps = [f for t, f in messages if t == log_reader.TIMESTEP_TOPIC]
    actions = [f for t, f in messages if t == log_reader.ACTION_TOPIC]
    self.assertLen(timesteps, 2)
    self.assertLen(actions, 2)
    np.testing.assert_array_equal(
        timesteps[1]["observation/qpos"].float_list.value, qpos
    )
    np.testing.assert_array_equal(
        actions[0]["action/act"].float_list.value, 1.0
    )
    self.assertEqual(app_id, "test_task")
    self.assertEqual(shapes["qpos"], (6,))

  def test_writes_an_episode_of_batched_torch(self):
    spec = timestep_spec(
        camera=specs.Array(shape=(1, 16, 16, 3), dtype=np.uint8),
        joint_pos=specs.Array(shape=(1, 6), dtype=np.float32),
    )
    action_spec = {
        "act": specs.BoundedArray(
            shape=(1, 6), dtype=np.float32, minimum=-1.0, maximum=1.0
        )
    }

    def observation(fill: int) -> dict[str, torch.Tensor]:
      return {
          "camera": torch.full((1, 16, 16, 3), fill, dtype=torch.uint8),
          "joint_pos": torch.full((1, 6), 0.5, dtype=torch.float32),
      }

    with tempfile.TemporaryDirectory() as tmp_dir:
      episode = logger.create_logger(
          timestep_spec=spec,
          action_spec=action_spec,
          policy_extra_spec={},
          image_observation_keys=["camera"],
          timestamp_key=_TIMESTAMP,
          task_id="torch_task",
          agent_id="torch_robot",
          output_directory=tmp_dir,
          control_timestep_seconds=0.02,
      )
      # Neither the instruction nor a timestamp is measured; both are supplied.
      episode.reset(dm_env.restart(observation(0)))
      episode.record_action_and_next_timestep(
          {"act": torch.zeros((1, 6), dtype=torch.float32)},
          dm_env.termination(
              reward=torch.tensor([1.0], dtype=torch.float32),
              observation=observation(120),
          ),
          {},
      )
      episode.write()
      episode.close()

      self.assertLen(mcaps_in(tmp_dir), 1)

  def test_rejects_an_observation_the_episode_does_not_declare(self):
    spec = timestep_spec(
        qpos=specs.Array(shape=(6,), dtype=np.float32),
        qvel=specs.Array(shape=(6,), dtype=np.float32),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      episode = logger.create_logger(
          timestep_spec=spec,
          action_spec={"act": specs.Array(shape=(6,), dtype=np.float32)},
          policy_extra_spec={},
          image_observation_keys=[],
          timestamp_key=_TIMESTAMP,
          task_id="task",
          agent_id="robot",
          output_directory=tmp_dir,
          control_timestep_seconds=0.02,
      )
      with self.assertRaisesRegex(ValueError, "qvel"):
        episode.reset(dm_env.restart({"qpos": np.zeros(6, dtype=np.float32)}))
      episode.stop()


@absltest.skipUnless(
    _HAS_SAFARI_SDK, "safari-sdk is required for MCAP logging."
)
class RunloopTest(absltest.TestCase):
  """The logger as the runloop drives it."""

  class Environment(gdmr_env.Environment):

    def __init__(self):
      self._steps = 0

    def default_reset_options(self):
      return None

    def reset_with_options(self, *, options=None):
      del options
      self._steps = 0
      return dm_env.restart(self._observation())

    def step(self, action):
      del action
      self._steps += 1
      if self._steps >= 3:
        return dm_env.termination(
            reward=torch.tensor(1.0, dtype=torch.float32),
            observation=self._observation(),
        )
      return dm_env.transition(
          reward=torch.tensor(0.0, dtype=torch.float32),
          observation=self._observation(),
      )

    def _observation(self):
      return {
          "wrist_cam": torch.full((1, 16, 16, 3), 50, dtype=torch.uint8),
          "joint_pos": torch.full(
              (1, 6), 0.1 * self._steps, dtype=torch.float32
          ),
      }

    def timestep_spec(self):
      return timestep_spec(
          wrist_cam=specs.Array(shape=(1, 16, 16, 3), dtype=np.uint8),
          joint_pos=specs.Array(shape=(1, 6), dtype=np.float32),
      )

    def action_spec(self):
      return {
          "act": specs.BoundedArray(
              shape=(1, 6), dtype=np.float32, minimum=-1.0, maximum=1.0
          )
      }

  class Policy(gdmr_policy.Policy):

    def initial_state(self):
      return ()

    def step_spec(self, timestep_spec):
      del timestep_spec
      return (
          {
              "act": specs.BoundedArray(
                  shape=(1, 6), dtype=np.float32, minimum=-1.0, maximum=1.0
              )
          },
          {},
      ), ()

    def step(self, timestep, prev_state=()):
      del timestep, prev_state
      return ({"act": torch.zeros((1, 6), dtype=torch.float32)}, {}), ()

  def test_records_an_episode_end_to_end(self):
    env = self.Environment()

    with tempfile.TemporaryDirectory() as tmp_dir:
      episode = logger.create_logger(
          timestep_spec=env.timestep_spec(),
          action_spec=env.action_spec(),
          policy_extra_spec={},
          image_observation_keys=["wrist_cam"],
          timestamp_key=_TIMESTAMP,
          task_id="runloop_test_task",
          agent_id="runloop_robot",
          output_directory=tmp_dir,
          control_timestep_seconds=0.02,
      )
      runloop_lib.Runloop(
          environment=env, policy=self.Policy(), loggers=[episode]
      ).run(num_episodes=1)
      episode.close()

      self.assertLen(mcaps_in(tmp_dir), 1)


if __name__ == "__main__":
  absltest.main()
