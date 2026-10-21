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
"""Simulation-side plugin for REAF Studio Viewer."""

import logging
import time
from typing import Protocol
from typing import runtime_checkable

import dm_env
import mujoco
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import plugin
from mujoco.experimental.studio import sim
import numpy as np
from reaf.simulation.viewer import messages as reaf_messages

_module_logger = logging.getLogger(__name__)

# Seconds to sleep per sim loop iteration when nothing advances (e.g. paused).
_IDLE_DURATION = 1.0 / 60.0

# The mjtState components the viewer edits by dragging a mocap body.
_MOCAP_SIG = int(mujoco.mjtState.mjSTATE_MOCAP_POS) | int(
    mujoco.mjtState.mjSTATE_MOCAP_QUAT
)


@runtime_checkable
class Instructable(Protocol):
  """Protocol for environments or wrappers that accept language instructions."""

  def set_instruction(self, instruction: str) -> None:
    """Sets the active task instruction."""
    ...


def set_env_instruction(environment: object, instruction: str) -> None:
  """Forwards an instruction to any Instructable layer in an environment stack."""
  env: object | None = environment
  while env is not None:
    if isinstance(env, Instructable):
      env.set_instruction(instruction)
    env = getattr(env, 'environment', getattr(env, '_env', None))


class SimBackend(Protocol):
  """Protocol bridging a simulation runtime (JAX/Torch) to CPU MjModel/MjData."""

  def reset(self) -> dm_env.TimeStep:
    """Resets the environment and returns the initial single-world timestep."""
    ...

  def step(self) -> dm_env.TimeStep:
    """Steps the agent and environment and returns the next single-world timestep."""
    ...

  def sync_to_viewer(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Syncs simulation state into CPU (model, data) and runs mj_forward."""
    ...

  def apply_perturbation(self, xfrc_applied_cpu: np.ndarray) -> None:
    """Applies an (nbody, 6) xfrc_applied array to the underlying simulation."""
    ...

  def apply_mocap(self, mocap_pos: np.ndarray, mocap_quat: np.ndarray) -> None:
    """Applies mocap positions (nmocap, 3) and quaternions (nmocap, 4)."""
    ...

  # Forwarded through SimBackend because the backend owns the environment instance.
  def set_instruction(self, instruction: str) -> None:
    """Forwards a text instruction to the environment if supported."""
    ...


class SimPlugin(plugin.SimPlugin):
  """Simulation-side REAF plugin that steps a SimBackend on StepEvent.

  The backend owns the simulation state; the CPU ``MjData`` handed to plugins
  is a mirror that ``on_step`` rewrites from the backend every iteration. The
  first ``StepEvent`` resets the environment, and every timestep change (step
  or reset) is published to the viewer as a ``TimestepEvent``.

  All handlers use the default priority and never consume messages. To observe
  the synced simulation after each step, handle ``PostStepEvent``.
  """

  def __init__(
      self,
      backend: SimBackend,
      *,
      step_duration: float | None = None,
      auto_reset: bool = True,
  ) -> None:
    self._backend = backend
    self._step_duration = step_duration
    self._auto_reset = auto_reset

    self._timestep: dm_env.TimeStep | None = None
    self._paused = False
    self._single_step = False
    self._reset_requested = False
    # The model the backend was built for, and whether it must be restored
    # after a model change (e.g. a file dropped onto the viewer).
    self._model: mujoco.MjModel | None = None
    self._restore_model = False

  def _set_pause_state(self, pause_state: sim.PauseState) -> None:
    """Updates the pause state and notifies the viewer on change."""
    paused = pause_state != sim.PauseState.UNPAUSED
    if paused != self._paused:
      self._paused = paused
      if self._handle is not None:
        self._handle.send_to_viewer(
            messages.RequestPauseEvent(pause_state=pause_state)
        )

  @messages.handler
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    """Saves the initial model and marks any replacement to be restored."""
    if self._model is None:
      self._model = event.model
    self._restore_model = event.model is not self._model

  @messages.handler
  def on_pre_step(self, _: messages.PreStepEvent) -> None:
    """Restores the backend's model if it was replaced."""
    if self._restore_model and self._handle is not None:
      _module_logger.warning(
          'The REAF viewer does not support changing the model; restoring it.'
      )
      assert self._model is not None
      self._handle.set_model(self._model)

  @messages.handler
  def on_step_control(self, event: messages.StepControlSnapshot) -> None:
    """Mirrors the viewer's pause state."""
    self._paused = event.pause_state != sim.PauseState.UNPAUSED

  @messages.handler
  def on_reset(self, _: messages.ResetEvent) -> None:
    """Requests an environment reset on the next step."""
    self._reset_requested = True

  @messages.handler
  def on_single_step(self, _: messages.SingleStepEvent) -> None:
    """Requests a single environment step while paused."""
    self._single_step = True

  @messages.handler
  def on_pause(self, event: messages.RequestPauseEvent) -> None:
    """Handles pause requests from the viewer or other sim-side plugins."""
    self._set_pause_state(event.pause_state)

  @messages.handler
  def on_instruction(self, event: reaf_messages.InstructionEvent) -> None:
    """Forwards instructions from the viewer to the environment."""
    self._backend.set_instruction(event.instruction)

  @messages.handler
  def on_perturb(self, event: messages.PerturbSnapshot) -> None:
    """Forwards perturbation forces from the viewer to the backend."""
    # The state is mjSTATE_XFRC_APPLIED: nbody x 6 values, flattened.
    self._backend.apply_perturbation(np.reshape(event.state, (-1, 6)))

  @messages.handler
  def on_state(self, event: messages.StateEvent) -> None:
    """Forwards mocap poses edited in the viewer, e.g. a dragged target body.

    ``ViewerHandle`` applies the event to the CPU mirror after this handler,
    but ``on_step`` rewrites the mirror from the backend, so the backend has to
    receive the pose itself. Only the mocap components are forwarded; other
    state edits (e.g. ``qpos`` from a paused drag of a free body) stay in the
    mirror until the next sync overwrites them.
    """
    if not event.state_sig & _MOCAP_SIG or self._handle is None:
      return
    model, data = self._handle.model, self._handle.data
    if model is None or data is None:
      return
    # Drop events built for another model, e.g. sent before a model change.
    if len(event.state) != mujoco.mj_stateSize(model, event.state_sig):
      return
    # The event may carry one or both mocap components, possibly with other
    # state; unpack it through the mirror rather than by offset arithmetic.
    mujoco.mj_setState(model, data, event.state, event.state_sig)
    self._backend.apply_mocap(data.mocap_pos, data.mocap_quat)

  @messages.handler
  def on_step(self, event: messages.StepEvent) -> None:
    """Resets or steps the backend, then syncs it into the CPU mirror."""
    start_time = time.time()
    ended = self._timestep is not None and self._timestep.last()
    new_timestep = None
    if (
        self._reset_requested
        or self._timestep is None
        or (ended and self._auto_reset)
    ):
      new_timestep = self._backend.reset()
      _module_logger.info('Environment reset.')
    elif not ended and (not self._paused or self._single_step):
      new_timestep = self._backend.step()
    self._reset_requested = False
    self._single_step = False

    # Leave event.data current so PostStepEvent observers see synced state.
    self._backend.sync_to_viewer(event.model, event.data)

    if new_timestep is not None:
      self._timestep = new_timestep
      if self._handle is not None:
        self._handle.send_to_viewer(
            reaf_messages.TimestepEvent(timestep=new_timestep)
        )
      duration = self._step_duration
    else:
      # Nothing advanced (paused, or the episode ended without auto-reset):
      # idle instead of busy-spinning the sim loop.
      duration = _IDLE_DURATION
    if (
        self._timestep is not None
        and self._timestep.last()
        and not self._auto_reset
    ):
      self._set_pause_state(sim.PauseState.NORMAL_PAUSED)
    if duration is not None:
      remaining = duration - (time.time() - start_time)
      if remaining > 0:
        time.sleep(remaining)
