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


def validate_render_mode(
    render_mode: RenderMode | None,
    enabled_mode: RenderMode | None = None,
) -> RenderMode:
  """Validates `render_mode` and optionally checks it against `enabled_mode`."""
  if render_mode is None:
    if enabled_mode is None:
      raise ValueError('Either render_mode or enabled_mode must be provided.')
    return enabled_mode

  if not render_mode:
    raise ValueError(
        'At least one render mode (RGB, DEPTH, or SEGMENTATION) must be'
        ' selected.'
    )

  if enabled_mode is not None and (render_mode & ~enabled_mode):
    raise ValueError(
        f'Requested render_mode ({render_mode}) is not enabled on this'
        f' device ({enabled_mode}).'
    )

  return render_mode


def render_camera(
    renderer: mujoco.Renderer,
    mj_data: mujoco.MjData,
    *,
    camera_id: int,
    scene_option: mujoco.MjvOption,
    render_mode: RenderMode,
    rgb_key: str,
    depth_key: str,
    segmentation_key: str,
    xpu_device: str | None = None,
) -> dict[str, torch.Tensor]:
  """Renders the requested `render_mode` passes for one camera."""
  renderer.update_scene(mj_data, camera=camera_id, scene_option=scene_option)
  measurements: dict[str, torch.Tensor] = {}

  if RenderMode.RGB in render_mode:
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    tensor = torch.from_numpy(renderer.render()).unsqueeze(0)
    measurements[rgb_key] = tensor.to(xpu_device) if xpu_device else tensor

  if RenderMode.DEPTH in render_mode:
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    tensor = torch.from_numpy(renderer.render()).unsqueeze(0)
    measurements[depth_key] = tensor.to(xpu_device) if xpu_device else tensor

  if RenderMode.SEGMENTATION in render_mode:
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    tensor = torch.from_numpy(renderer.render()).unsqueeze(0)
    measurements[segmentation_key] = (
        tensor.to(xpu_device) if xpu_device else tensor
    )

  return measurements


def _qpos_to_numpy(qpos: torch.Tensor) -> np.ndarray:
  """Copies a `(nq,)` or `(nworld, nq)` qpos tensor to a host `(nq,)` array."""
  # Currently only supports nworld=1.
  if qpos.ndim == 2:
    qpos = qpos[0]
  return qpos.detach().cpu().numpy().copy()


def get_scene_data(
    data: types.Data,
    impl: types.Impl,
    model: mujoco.MjModel,
    scratch_data: mujoco.MjData | None,
    qpos: np.ndarray | None = None,
) -> mujoco.MjData:
  """Returns an `MjData` positioned at the state that `Renderer` should draw.

  On `Impl.CPP` when `qpos` is `None`, `data.obj` is already a live
  `mujoco.MjData` and is returned untouched. Otherwise (`Impl.WARP` or an
  explicit historical `qpos`), `scratch_data` is populated and updated via
  `mj_kinematics` and `mj_camlight`.

  Args:
    data: The live simulation data.
    impl: Simulation backend implementation (`CPP` or `WARP`).
    model: Compiled `mujoco.MjModel`.
    scratch_data: Host `mujoco.MjData` workspace used when `impl != Impl.CPP` or
      `qpos` is provided.
    qpos: Optional host `(nq,)` generalized positions to render instead of
      `data.qpos` (used for delayed rendering).

  Returns:
    An `MjData` positioned at the state that should be rendered.

  Raises:
    ValueError: If `data.obj` is not a `mujoco.MjData` on the direct path.
    RuntimeError: If `scratch_data` is `None` when required.
  """
  if impl == types.Impl.CPP and qpos is None:
    mj_data = data.obj
    if not isinstance(mj_data, mujoco.MjData):
      raise ValueError(
          f'Expected data.obj to be mujoco.MjData when impl={impl}, '
          f'got {type(mj_data)}'
      )
    return mj_data

  if scratch_data is None:
    raise RuntimeError('Scratch MjData is not initialized.')

  if model.nq > 0:
    scratch_data.qpos[:] = (
        qpos if qpos is not None else _qpos_to_numpy(data.qpos)
    )
  mujoco.mj_kinematics(model, scratch_data)
  mujoco.mj_camlight(model, scratch_data)
  return scratch_data


class MujocoCameraDevice(device.Device):
  """Entity that renders the scene using MuJoCo rendering."""

  _camera_id: int = -1
  _renderer: mujoco.Renderer | None = None
  _model: mujoco.MjModel
  _data: mujoco.MjData | None = None
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
      on_demand_only: bool = False,
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
      on_demand_only: If True, the device publishes no substep measurements and
        never renders on its own (not even after a reset); frames are produced
        only by `render_now`. Requires `render_rate_hz=None`. Use this for
        cameras read exclusively through `render_now`, so that a reset does not
        render every enabled mode.
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

    if on_demand_only and render_rate_hz is not None:
      raise ValueError(
          f'on_demand_only requires render_rate_hz=None, got {render_rate_hz}'
      )

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

    validate_render_mode(render_mode)

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
    # `render_now` output keys follow the same naming, but an on-demand-only
    # device publishes nothing as a substep measurement.
    self._measurements_spec = {} if on_demand_only else measurements_spec
    self._on_demand_only = on_demand_only

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
    self._renderer = None
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

  def on_substep(self, simulator: simulator_module.Simulator) -> None:
    """Records the delayed qpos and renders on render ticks."""
    self._render_with_interval(simulator)

  def on_begin_stepping(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Records the reset qpos and renders the first frame after a reset."""
    del env_mask  # `reset()` only clears the history if it resets the world.
    self._render_with_interval(simulator)

  def get_measurements(
      self, simulator: simulator_module.Simulator
  ) -> Mapping[str, torch.Tensor]:
    """Returns the latest rendered frames, without rendering."""
    del simulator
    if self._measurements_spec and not self._last_measurements:
      raise RuntimeError(
          f'{self.name}: nothing was rendered since the last reset. Call'
          ' `DeviceLayer.begin_stepping()` after resetting.'
      )
    return self._last_measurements

  def render_now(
      self,
      simulator: simulator_module.Simulator,
      render_mode: RenderMode | None = None,
  ) -> dict[str, torch.Tensor]:
    """Renders the camera immediately without affecting device measurements.

    Side-channel call for `Impl.CPP` single-world environments that renders the
    current physics state (or the state from `delay_ms` ago when a delay is
    configured) without modifying cached substep measurements, the
    `render_rate_hz` schedule, or the delay history buffer.

    Args:
      simulator: The simulator instance.
      render_mode: Optional subset of enabled `RenderMode` flags to render.
        `None` renders all modes configured on this device.

    Returns:
      A dictionary containing the freshly rendered images for the requested
      modes.

    Raises:
      RuntimeError: If the device has not been initialized.
      ValueError: If `self._impl != types.Impl.CPP`, `data.nworld` is greater
        than 1, or `render_mode` is empty or includes modes not enabled on this
        device.
    """
    if self._impl is None:
      raise RuntimeError('Device not initialized.')

    if self._impl != types.Impl.CPP:
      raise ValueError(f'render_now only supports Impl.CPP, got {self._impl}')

    active_mode = validate_render_mode(render_mode, self._render_mode)

    data = simulator.data
    if data.nworld > 1:
      raise ValueError(
          f'MujocoCameraDevice only supports nworld=1, got {data.nworld}'
      )

    return self._render(data, render_mode=active_mode)

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

    if self._on_demand_only:
      return {}

    if not self._last_measurements or (
        self._render_rate_hz is not None
        and self._last_rendered_substep != global_substep
        and (global_substep % self._update_interval_in_substeps) == 0
    ):
      self._last_measurements = self._render(data)
      self._last_rendered_substep = global_substep

    return self._last_measurements

  def _render(
      self,
      data: types.Data,
      render_mode: RenderMode | None = None,
  ) -> dict[str, torch.Tensor]:
    """Performs the actual rendering."""
    renderer = self._renderer
    if renderer is None:
      raise RuntimeError('Renderer is not initialized.')

    active_mode = self._render_mode if render_mode is None else render_mode
    return render_camera(
        renderer,
        self._get_scene_data(data),
        camera_id=self._camera_id,
        scene_option=self._scene_option,
        render_mode=active_mode,
        rgb_key=self.output_name_rgb,
        depth_key=self.output_name_depth,
        segmentation_key=self.output_name_segmentation,
        xpu_device=self._xpu_device,
    )

  def _get_scene_data(self, data: types.Data) -> mujoco.MjData:
    """Returns the `MjData` holding the state the renderer should draw."""
    assert self._impl is not None
    delayed_qpos = (
        self._get_delayed_qpos()
        if self._delay_in_substeps > 0 and self._total_stored > 0
        else None
    )
    return get_scene_data(
        data=data,
        impl=self._impl,
        model=self._model,
        scratch_data=self._data,
        qpos=delayed_qpos,
    )

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

  def close(self) -> None:
    """Closes and releases the underlying MuJoCo renderer, if initialized."""
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None
