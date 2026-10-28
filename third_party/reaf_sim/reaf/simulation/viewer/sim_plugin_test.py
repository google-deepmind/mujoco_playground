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
"""Tests for the REAF Studio SimPlugin."""

from absl.testing import absltest
import dm_env
import mujoco
from mujoco.studio import launch_thread
from mujoco.studio import messages
from mujoco.studio import sim
from mujoco.studio import viewer_handle
import numpy as np
from reaf.simulation.viewer import messages as reaf_messages
from reaf.simulation.viewer import sim_plugin

_XML = """
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom size=".1"/>
    </body>
    <body name="goal" mocap="true">
      <geom size=".05" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class _FakeBackend:
  """A backend whose episodes terminate after `episode_length` steps."""

  def __init__(self, episode_length: int = 3) -> None:
    self.episode_length = episode_length
    self.n_steps = 0
    self.n_resets = 0
    self.n_syncs = 0
    self.perturbations: list[np.ndarray] = []
    self.mocaps: list[tuple[np.ndarray, np.ndarray]] = []
    self.instructions: list[str] = []

  def reset(self) -> dm_env.TimeStep:
    self.n_resets += 1
    self.n_steps = 0
    return dm_env.restart(observation=None)

  def step(self) -> dm_env.TimeStep:
    self.n_steps += 1
    if self.n_steps >= self.episode_length:
      return dm_env.termination(reward=1.0, observation=None)
    return dm_env.transition(reward=1.0, observation=None)

  def sync_to_viewer(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    del model, data
    self.n_syncs += 1

  def apply_perturbation(self, xfrc_applied_cpu: np.ndarray) -> None:
    self.perturbations.append(xfrc_applied_cpu)

  def apply_mocap(self, mocap_pos: np.ndarray, mocap_quat: np.ndarray) -> None:
    # Copy: the plugin passes views into the handle's MjData.
    self.mocaps.append((mocap_pos.copy(), mocap_quat.copy()))

  def set_instruction(self, instruction: str) -> None:
    self.instructions.append(instruction)


def _pause_snapshot(
    pause_state: sim.PauseState,
) -> messages.StepControlSnapshot:
  return messages.StepControlSnapshot(
      pause_state=pause_state, speed=1.0, noise_scale=0.0, noise_rate=0.0
  )


class SimPluginTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.backend = _FakeBackend()
    self.model = mujoco.MjModel.from_xml_string(_XML)
    self.data = mujoco.MjData(self.model)
    self._start()

  def _start(self, **kwargs) -> None:
    """Creates a ViewerHandle with a SimPlugin built from `kwargs`."""
    self.viewer_endpoint, sim_endpoint = launch_thread.make_thread_endpoints()
    self.handle = viewer_handle.ViewerHandle(
        sim_endpoint,
        sim_plugins=[sim_plugin.SimPlugin(self.backend, **kwargs)],
    )

  def _sync(self) -> list[dm_env.TimeStep]:
    """Runs one sim loop iteration; returns the published timesteps."""
    model, data = self.handle.sync(self.model, self.data)
    assert model is not None and data is not None
    self.model, self.data = model, data
    self.sim_events = self.viewer_endpoint.get_sim_events()
    return [
        event.timestep
        for event in self.sim_events
        if isinstance(event, reaf_messages.TimestepEvent)
    ]

  def test_first_step_resets_then_steps_while_running(self):
    (first,) = self._sync()
    self.assertTrue(first.first())
    (second,) = self._sync()
    self.assertEqual(second.step_type, dm_env.StepType.MID)
    self.assertEqual(self.backend.n_resets, 1)
    self.assertEqual(self.backend.n_steps, 1)
    self.assertEqual(self.backend.n_syncs, 2)

  def test_paused_does_not_step_until_single_step(self):
    self._sync()  # Initial reset.
    self.viewer_endpoint.send_to_sim(
        _pause_snapshot(sim.PauseState.NORMAL_PAUSED)
    )
    self.assertEmpty(self._sync())
    self.assertEqual(self.backend.n_steps, 0)

    self.viewer_endpoint.send_to_sim(messages.SingleStepEvent())
    self.assertLen(self._sync(), 1)
    self.assertEmpty(self._sync())
    self.assertEqual(self.backend.n_steps, 1)
    # The mirror is synced every iteration, even while paused.
    self.assertEqual(self.backend.n_syncs, 4)

  def test_request_pause_event_pauses_and_unpauses(self):
    self._sync()  # Initial reset.
    self.viewer_endpoint.send_to_sim(
        messages.RequestPauseEvent(pause_state=sim.PauseState.NORMAL_PAUSED)
    )
    self.assertEmpty(self._sync())
    self.assertEqual(self.backend.n_steps, 0)
    self.assertEqual(
        self.sim_events,
        [messages.RequestPauseEvent(pause_state=sim.PauseState.NORMAL_PAUSED)],
    )

    self.viewer_endpoint.send_to_sim(
        messages.RequestPauseEvent(pause_state=sim.PauseState.UNPAUSED)
    )
    self.assertLen(self._sync(), 1)
    self.assertEqual(self.backend.n_steps, 1)

  def test_auto_reset_resets_after_last_timestep(self):
    step_types = []
    for _ in range(5):
      step_types.extend(ts.step_type for ts in self._sync())
    self.assertEqual(
        step_types,
        [
            dm_env.StepType.FIRST,
            dm_env.StepType.MID,
            dm_env.StepType.MID,
            dm_env.StepType.LAST,
            dm_env.StepType.FIRST,
        ],
    )

  def test_without_auto_reset_pauses_and_waits_for_reset_event(self):
    self._start(auto_reset=False)
    for _ in range(4):
      self._sync()
    pause_events = [
        e for e in self.sim_events if isinstance(e, messages.RequestPauseEvent)
    ]
    self.assertEqual(
        pause_events,
        [messages.RequestPauseEvent(pause_state=sim.PauseState.NORMAL_PAUSED)],
    )
    self.assertEmpty(self._sync())  # Episode ended: nothing advances.
    self.assertEqual(self.backend.n_resets, 1)

    self.viewer_endpoint.send_to_sim(messages.ResetEvent())
    (timestep,) = self._sync()
    self.assertTrue(timestep.first())
    self.assertEqual(self.backend.n_resets, 2)

  def test_perturbation_is_forwarded_as_nbody_by_6(self):
    self._sync()
    xfrc = np.arange(self.model.nbody * 6, dtype=np.float64)
    self.viewer_endpoint.send_to_sim(
        messages.PerturbSnapshot(
            state=xfrc, state_sig=int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
        )
    )
    self._sync()
    self.assertLen(self.backend.perturbations, 1)
    np.testing.assert_array_equal(
        self.backend.perturbations[0], xfrc.reshape(self.model.nbody, 6)
    )

  def test_mocap_state_is_forwarded(self):
    self._sync()
    # A running drag sends both mocap components; a paused drag also sends
    # qpos. Either way the backend gets the full (nmocap, 3/4) arrays.
    sig = (
        int(mujoco.mjtState.mjSTATE_QPOS)
        | int(mujoco.mjtState.mjSTATE_MOCAP_POS)
        | int(mujoco.mjtState.mjSTATE_MOCAP_QUAT)
    )
    mocap_data = mujoco.MjData(self.model)
    mocap_data.mocap_pos[0] = [0.1, 0.2, 0.3]
    mocap_data.mocap_quat[0] = [0.0, 1.0, 0.0, 0.0]
    state = np.empty(mujoco.mj_stateSize(self.model, sig))
    mujoco.mj_getState(self.model, mocap_data, state, sig)
    self.viewer_endpoint.send_to_sim(
        messages.StateEvent(state=state, state_sig=sig)
    )
    self._sync()
    self.assertLen(self.backend.mocaps, 1)
    mocap_pos, mocap_quat = self.backend.mocaps[0]
    np.testing.assert_array_equal(mocap_pos, [[0.1, 0.2, 0.3]])
    np.testing.assert_array_equal(mocap_quat, [[0.0, 1.0, 0.0, 0.0]])

  def test_mocap_position_only_state_keeps_mirror_orientation(self):
    self._sync()
    sig = int(mujoco.mjtState.mjSTATE_MOCAP_POS)
    self.viewer_endpoint.send_to_sim(
        messages.StateEvent(state=np.array([0.1, 0.2, 0.3]), state_sig=sig)
    )
    self._sync()
    mocap_pos, mocap_quat = self.backend.mocaps[0]
    np.testing.assert_array_equal(mocap_pos, [[0.1, 0.2, 0.3]])
    np.testing.assert_array_equal(mocap_quat, [[1.0, 0.0, 0.0, 0.0]])

  def test_state_without_mocap_is_not_forwarded(self):
    self._sync()
    sig = int(mujoco.mjtState.mjSTATE_QPOS)
    self.viewer_endpoint.send_to_sim(
        messages.StateEvent(state=self.data.qpos.copy(), state_sig=sig)
    )
    self._sync()
    self.assertEmpty(self.backend.mocaps)

  def test_mocap_state_for_another_model_is_dropped(self):
    self._sync()
    sig = int(mujoco.mjtState.mjSTATE_MOCAP_POS)
    self.viewer_endpoint.send_to_sim(
        messages.StateEvent(state=np.zeros(6), state_sig=sig)  # Two bodies.
    )
    self._sync()
    self.assertEmpty(self.backend.mocaps)

  def test_instruction_is_forwarded(self):
    self.viewer_endpoint.send_to_sim(
        reaf_messages.InstructionEvent(instruction='pick up the cube')
    )
    self._sync()
    self.assertEqual(self.backend.instructions, ['pick up the cube'])

  def test_model_change_is_reverted(self):
    original = self.model
    self._sync()
    self.viewer_endpoint.get_sim_events()
    other = mujoco.MjModel.from_xml_string('<mujoco/>')
    self.viewer_endpoint.send_to_sim(messages.ModelEvent(model=other))

    self._sync()

    self.assertIs(self.model, original)
    self.assertIs(self.handle.model, original)
    self.assertEqual(self.data.qpos.shape, (original.nq,))


if __name__ == '__main__':
  absltest.main()
