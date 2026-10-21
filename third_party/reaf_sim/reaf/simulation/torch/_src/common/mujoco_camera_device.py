# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entity that renders the scene using MuJoCo rendering."""

import enum
import logging
import math
from typing import Any
from typing import Mapping
from typing import Self

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch

_DEFAULT_NAME_PREFIX = 'MujocoCameraDevice'
_logger = logging.getLogger(__name__)


class RenderMode(enum.Flag):
  """Render modes for the various camera entities.

  These flags can be combined to render multiple outputs.
  """

  RGB = enum.auto()
  DEPTH = enum.auto()
  SEGMENTATION = enum.auto()

  def __str__(self):
    members = []
    if self.RGB in self:
      members.append(self._value_to_string(self.RGB))
    if self.DEPTH in self:
      members.append(self._value_to_string(self.DEPTH))
    if self.SEGMENTATION in self:
      members.append(self._value_to_string(self.SEGMENTATION))
    return ' | '.join(members)

  def _value_to_string(self, value: Self) -> str:
    """Converts an integer value to a string."""
    if value == self.RGB:
      return 'rgb'
    elif value == self.DEPTH:
      return 'depth'
    elif value == self.SEGMENTATION:
      return 'segmentation'
    else:
      raise ValueError(f'Unknown value: {value}')


def _qpos_to_numpy(qpos: torch.Tensor) -> np.ndarray:
  """Copies a `(nq,)` or `(nworld, nq)` qpos tensor to a host `(nq,)` array."""
  # Currently only supports nworld=1.
  if qpos.ndim == 2:
    qpos = qpos[0]
  return qpos.detach().cpu().numpy().copy()


class MujocoCameraDevice(device.Device):
  """Entity that renders the scene using MuJoCo rendering."""

  _camera_id: int = -1
  _renderer: mujoco.Renderer | None
  _model: mujoco.MjModel
  _data: mujoco.MjData | None
  _xpu_device: str

  def __init__(
      self,
      *,
      camera_name: str,
      width: int,
      height: int,
      render_rate_hz: float | None,
      delay_ms: float = 0.0,
      scene_option: mujoco.MjvOption | None = None,
      render_mode: RenderMode = RenderMode.RGB,
      name: str | None = None,
      image_output_name_prefix: str | None = None,
      rgb_output_name_suffix: str = '_rgb',
      depth_output_name_suffix: str = '_depth',
      segmentation_output_name_suffix: str = '_segmentation',
      xpu_device: str | None = None,
  ):
    """Initializes the MujocoCameraDevice.

    Args:
      camera_name: Name of the camera to use for rendering.
      width: Width of the rendered image.
      height: Height of the rendered image.
      render_rate_hz: Camera rendering rate in Hz, or None to render only the
        first measurement after each reset, whose frame later measurements
        return until the next reset. Must be positive if set.
      delay_ms: Latency in milliseconds by which the rendered image lags the
        current physics state. 0.0 (default) means the current state is rendered
        (no delay and no overhead). When > 0, the device records qpos at each
        substep into a small ring buffer and renders from the qpos recorded
        ``delay_ms`` ago, independent of ``render_rate_hz``. The first renders
        after a reset are clamped to the history available so far.
      scene_option: Mujoco scene options for rendering.
      render_mode: What to render (RGB, DEPTH, SEGMENTATION).
      name: Name of the entity. If None the name will be auto-generated from the
        camera name and render mode(s).
      image_output_name_prefix: Prefix of the output image name. If None the
        camera name will be used with the render mode as suffix.
      rgb_output_name_suffix: Suffix for the rgb image output name.
      depth_output_name_suffix: Suffix for the depth image output name.
      segmentation_output_name_suffix: Suffix for the segmentation image output
        name.
      xpu_device: Optional device to use for the entity. If None, the device
        from the simulation device will be used.

    Attributes:
      output_name_rgb: Name of the RGB output.
      output_name_depth: Name of the depth output.
      output_name_segmentation: Name of the segmentation output.
    """

    if render_rate_hz is not None and render_rate_hz <= 0:
      raise ValueError(
          f'render_rate_hz must be positive or None, got {render_rate_hz}'
      )

    if delay_ms < 0:
      raise ValueError(f'delay_ms must be non-negative, got {delay_ms}')

    if name is None:
      name = f'{_DEFAULT_NAME_PREFIX}_{camera_name}'
      if RenderMode.RGB in render_mode:
        name += '_rgb'
      if RenderMode.DEPTH in render_mode:
        name += '_depth'
      if RenderMode.SEGMENTATION in render_mode:
        name += '_segmentation'
    self._name = name
    self._camera_name = camera_name

    if image_output_name_prefix is None:
      output_name_prefix = camera_name
    else:
      output_name_prefix = image_output_name_prefix
    self.output_name_rgb = f'{output_name_prefix}{rgb_output_name_suffix}'
    self.output_name_depth = f'{output_name_prefix}{depth_output_name_suffix}'
    self.output_name_segmentation = (
        f'{output_name_prefix}{segmentation_output_name_suffix}'
    )

    if not (
        RenderMode.RGB in render_mode
        or RenderMode.DEPTH in render_mode
        or RenderMode.SEGMENTATION in render_mode
    ):
      raise ValueError(
          'At least one render mode (RGB, DEPTH, or SEGMENTATION) must be'
          ' selected.'
      )

    measurements_spec = {}
    if RenderMode.RGB in render_mode:
      measurements_spec[self.output_name_rgb] = specs.Array(
          shape=(height, width, 3),
          dtype=np.uint8,
      )
    if RenderMode.DEPTH in render_mode:
      measurements_spec[self.output_name_depth] = specs.Array(
          shape=(
              height,
              width,
          ),
          dtype=np.float32,
      )
    if RenderMode.SEGMENTATION in render_mode:
      measurements_spec[self.output_name_segmentation] = specs.Array(
          shape=(height, width, 2),
          dtype=np.int32,
      )
    self._measurements_spec = measurements_spec

    self._render_rate_hz = render_rate_hz
    self._width = width
    self._height = height
    self._scene_option = scene_option if scene_option else mujoco.MjvOption()
    self._render_mode = render_mode
    self._update_interval_in_substeps = 1
    self._impl: types.Impl | None = None  # Inferred in `initialize`.

    self._delay_ms = delay_ms
    self._delay_in_substeps: int = 0
    # Ring buffer of past qpos for delayed rendering. Configured in
    # `initialize()` once `simulator.clock.sim_dt` is known.
    self._buffer_capacity = 1
    self._qpos_buffer: list[np.ndarray | None] = [None]
    self._write_idx = 0
    self._total_stored = 0
    self._last_recorded_substep: int = -1

    self._last_measurements: dict[str, torch.Tensor] = {}
    self._last_rendered_substep: int = -1
    # Simulator has its own device, so override it only if we get an
    # explicit device assignment.
    self._xpu_device_arg = xpu_device
    self._xpu_device = 'cpu'

  @property
  def name(self) -> str:
    """Instance name."""
    return self._name

  @property
  def rendering_configuration(self) -> Mapping[str, Any]:
    """The rendering configuration."""

    return {
        'width': self._width,
        'height': self._height,
    }

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Initializes the entity with compiled model and data."""
    sim_dt = simulator.clock.sim_dt
    if self._render_rate_hz is not None:
      desired_period = 1.0 / self._render_rate_hz
      self._update_interval_in_substeps = max(1, round(desired_period / sim_dt))
      actual_period = self._update_interval_in_substeps * sim_dt
      if not math.isclose(
          actual_period, desired_period, rel_tol=1e-6, abs_tol=1e-9
      ):
        _logger.warning(
            'Desired render_rate_hz=%g (period %.6fs) is not an integer'
            ' multiple of sim_dt=%.6fs; discretizing to %d substeps (actual'
            ' render rate %.2f Hz).',
            self._render_rate_hz,
            desired_period,
            sim_dt,
            self._update_interval_in_substeps,
            1.0 / actual_period,
        )

    if self._delay_ms > 0:
      desired_delay_s = self._delay_ms / 1000.0
      self._delay_in_substeps = max(1, round(desired_delay_s / sim_dt))
      actual_delay_s = self._delay_in_substeps * sim_dt
      if not math.isclose(
          actual_delay_s, desired_delay_s, rel_tol=1e-6, abs_tol=1e-9
      ):
        _logger.warning(
            'Desired delay_ms=%g (%.6fs) is not an integer multiple of'
            ' sim_dt=%.6fs; discretizing to %d substeps (actual delay'
            ' %.3f ms).',
            self._delay_ms,
            desired_delay_s,
            sim_dt,
            self._delay_in_substeps,
            actual_delay_s * 1000.0,
        )
    else:
      self._delay_in_substeps = 0

    self._buffer_capacity = self._delay_in_substeps + 1
    self._qpos_buffer = [None] * self._buffer_capacity
    self._write_idx = 0
    self._total_stored = 0
    self._last_recorded_substep = -1

    mj_model = simulator.mj_model
    data = simulator.data
    self._model = mj_model
    self._renderer = mujoco.Renderer(
        self._model, height=self._height, width=self._width
    )
    # Inferred rather than configured: rendering only cares whether `data` is
    # a `mujoco.MjData` it can hand straight to the renderer, so ask the data.
    self._impl = (
        types.Impl.CPP
        if isinstance(data.obj, mujoco.MjData)
        else types.Impl.WARP
    )
    # A scratch MjData is needed whenever we cannot render straight from the
    # live data: for WARP (data.obj is not an MjData) or for delayed rendering
    # (we render from a historical qpos, not the current physics state).
    if self._impl == types.Impl.WARP or self._delay_in_substeps > 0:
      self._data = mujoco.MjData(mj_model)
    else:
      # The live `data.obj` is rendered directly, so no scratch copy is needed.
      self._data = None
    self._camera_id = self._model.camera(self._camera_name).id
    if self._xpu_device_arg is None:
      self._xpu_device = (
          str(data.qpos.device) if hasattr(data, 'qpos') else 'cpu'
      )
    else:
      self._xpu_device = self._xpu_device_arg

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Resets the processor.

    Args:
      simulator: The simulator instance.
      env_mask: Boolean mask of the environments to reset. If None, all
        environments are reset.
    """
    del simulator
    if env_mask is None or bool(env_mask[0]):
      self._last_measurements = {}
      self._last_rendered_substep = -1
      self._last_recorded_substep = -1
      self._qpos_buffer = [None] * self._buffer_capacity
      self._write_idx = 0
      self._total_stored = 0

  def substep_measurements_spec(
      self,
  ) -> Mapping[str, specs.Array]:
    """Spec for the produced substep outputs."""
    return self._measurements_spec

  def get_substep_measurements(
      self, simulator: simulator_module.Simulator
  ) -> Mapping[str, torch.Tensor]:
    """Renders the scene using MuJoCo rendering.

    Args:
      simulator: The simulator instance.

    Returns:
      A mapping containing the rendered images.

    Raises:
      ValueError: If `data.nworld` is greater than 1.
      RuntimeError: If the renderer has not been initialized.
      ValueError: If `data.obj` is not of type `mujoco.MjData` on the CPP
        no-delay path.
    """
    return self._render_with_interval(simulator)

  def _render_with_interval(
      self, simulator: simulator_module.Simulator
  ) -> Mapping[str, torch.Tensor]:
    """Renders at the configured update interval, caching between renders."""
    data = simulator.data
    if data.nworld > 1:
      raise ValueError(
          f'MujocoCameraDevice only supports nworld=1, got {data.nworld}'
      )

    global_substep = simulator.clock.global_substep
    if (
        self._delay_in_substeps > 0
        and self._last_recorded_substep != global_substep
    ):
      self._record_qpos(data)
      self._last_recorded_substep = global_substep

    if not self._last_measurements or (
        self._render_rate_hz is not None
        and self._last_rendered_substep != global_substep
        and (global_substep % self._update_interval_in_substeps) == 0
    ):
      self._last_measurements = self._render(data)
      self._last_rendered_substep = global_substep

    return self._last_measurements

  def _render(self, data: types.Data) -> dict[str, torch.Tensor]:
    """Performs the actual rendering."""
    renderer = self._renderer
    if renderer is None:
      raise RuntimeError('Renderer is not initialized.')

    renderer.update_scene(
        self._get_scene_data(data),
        camera=self._camera_id,
        scene_option=self._scene_option,
    )

    measurements = {}

    if RenderMode.RGB in self._render_mode:
      renderer.disable_depth_rendering()
      renderer.disable_segmentation_rendering()
      img = renderer.render()
      measurements[self.output_name_rgb] = (
          torch.from_numpy(img).unsqueeze(0).to(self._xpu_device)
      )

    if RenderMode.DEPTH in self._render_mode:
      renderer.disable_segmentation_rendering()
      renderer.enable_depth_rendering()
      img = renderer.render()
      measurements[self.output_name_depth] = (
          torch.from_numpy(img).unsqueeze(0).to(self._xpu_device)
      )

    if RenderMode.SEGMENTATION in self._render_mode:
      renderer.enable_segmentation_rendering()
      img = renderer.render()
      measurements[self.output_name_segmentation] = (
          torch.from_numpy(img).unsqueeze(0).to(self._xpu_device)
      )

    return measurements

  def _get_scene_data(self, data: types.Data) -> mujoco.MjData:
    """Returns the `MjData` holding the state the renderer should draw.

    There are two ways to obtain it:

    -   Use the live data directly. On CPP with no delay, `data.obj` already is
        a `mujoco.MjData` sitting at exactly the state we want to draw, so it
        is returned untouched.
    -   Update the scratch `MjData` from a `qpos`, via `mj_kinematics` and
        `mj_camlight`. The `qpos` comes from the history buffer when
        `delay_ms > 0`, or off the device under Warp.

    Args:
      data: The live simulation data.

    Returns:
      An `MjData` positioned at the state that should be rendered.

    Raises:
      ValueError: If `data.obj` is not a `mujoco.MjData` on the direct path.
    """
    if self._impl == types.Impl.CPP and self._delay_in_substeps == 0:
      mj_data = data.obj
      if not isinstance(mj_data, mujoco.MjData):
        raise ValueError(
            f'Expected data.obj to be mujoco.MjData when impl={self._impl}, '
            f'got {type(mj_data)}'
        )
      return mj_data

    # Guaranteed by `initialize`, which allocates the scratch `MjData` under
    # exactly the condition that lands us here (Warp, or a non-zero delay).
    assert self._data is not None

    if self._model.nq > 0:
      self._data.qpos[:] = (
          self._get_delayed_qpos()
          if self._delay_in_substeps > 0
          else _qpos_to_numpy(data.qpos)
      )
    mujoco.mj_kinematics(self._model, self._data)
    mujoco.mj_camlight(self._model, self._data)
    return self._data

  def _record_qpos(self, data: types.Data) -> None:
    """Stores the current qpos into the ring buffer (delayed rendering)."""
    self._qpos_buffer[self._write_idx] = _qpos_to_numpy(data.qpos)
    self._write_idx = (self._write_idx + 1) % self._buffer_capacity
    self._total_stored += 1

  def _get_delayed_qpos(self) -> np.ndarray:
    """Returns the qpos from ``_delay_in_substeps`` substeps ago.

    If not enough history has accumulated yet (e.g. the first few substeps
    after reset), the effective delay is clamped to the available history.
    """
    available = min(self._total_stored, self._buffer_capacity)
    effective_delay = min(self._delay_in_substeps, max(available - 1, 0))
    read_idx = (self._write_idx - 1 - effective_delay) % self._buffer_capacity
    qpos = self._qpos_buffer[read_idx]
    if qpos is None:
      # Safety fallback — should not happen after the first _record_qpos.
      # qpos0 (not zeros) keeps quaternions in free/ball joints normalized.
      return self._model.qpos0.copy()
    return qpos

  def substep_commands_spec(
      self,
  ) -> Mapping[str, specs.Array]:
    """No commands are consumed."""
    return {}

  def set_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      consumed_substep_commands: Mapping[str, torch.Tensor],
  ) -> None:
    """No commands are consumed."""
    del simulator, consumed_substep_commands
