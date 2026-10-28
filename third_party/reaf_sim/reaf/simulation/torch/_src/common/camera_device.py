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

"""Camera entity using the MuJoCo Warp renderer."""

from collections.abc import Mapping
import logging
import math
from typing import Union

from dm_env import specs
import mujoco
import mujoco_warp as mjw
import numpy as np
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch
import warp as wp

_logger = logging.getLogger(__name__)


class CameraDevice(device.Device):
  """Entity that renders the scene using MuJoCo rendering."""

  def __init__(
      self,
      *,
      depth_scale: float | None = 1.0,
      render_rate_hz: float,
      depth_dtype: torch.dtype = torch.uint8,
      name: str | None = None,
      cam_name: Union[list[str] | str] | None = None,
      cam_resolution: (
          Union[list[tuple[int, int]] | tuple[int, int]] | None
      ) = None,
      render_rgb: Union[list[bool] | bool] = True,
      render_depth: Union[list[bool] | bool] = False,
      output_name_prefix: str | None = None,
      rgb_output_name_suffix: str = '_rgb',
      depth_output_name_suffix: str = '_depth',
      **render_context_kwargs,
  ):
    """Initializes the CameraDevice.

    Args:
      render_rate_hz: Camera rendering rate in Hz. Must be positive.
      depth_scale: Maximum depth value to scale the depth image to [0, 255]. If
        None, depth values are not normalized.
      depth_dtype: Data type of the depth measurements (defaults to
        torch.uint8).
      name: Name of the entity.
      cam_name: Name of the cameras to render. If None, all cameras will be
        rendered.
      cam_resolution: Resolutions of the cameras to render, as (width, height)
        tuples.
      render_rgb: Whether to render RGB images.
      render_depth: Whether to render depth images.
      output_name_prefix: Prefix of the output image name. If None the camera
        name will be used with the render mode as suffix.
      rgb_output_name_suffix: Suffix for the rgb image output name.
      depth_output_name_suffix: Suffix for the depth image output name.
      **render_context_kwargs: Keyword arguments to pass to the MuJoCo Warp
        render context.
    """
    if render_rate_hz <= 0:
      raise ValueError(f'render_rate_hz must be positive, got {render_rate_hz}')

    # Filter out params added by build system that RenderContext doesn't accept.
    render_context_kwargs.pop('xpu_device', None)

    self._render_rate_hz = render_rate_hz
    self._depth_scale = depth_scale
    self._depth_dtype = depth_dtype
    self._name = name or 'CameraDevice' + str(id(self))
    self._render_context_kwargs = render_context_kwargs or {}
    self._output_name_prefix = output_name_prefix or ''
    self._rgb_output_name_suffix = rgb_output_name_suffix
    self._depth_output_name_suffix = depth_output_name_suffix
    self._update_interval_in_substeps = 1

    self._raw_cam_name = cam_name
    self._raw_cam_resolution = cam_resolution
    self._raw_render_rgb = render_rgb
    self._raw_render_depth = render_depth

    self._last_measurements: dict[str, torch.Tensor] = {}
    self._last_rendered_substep: int = -1

    # These will be set in initialize().
    self._ncam: int = 0
    self._cam_resolution: list[tuple[int, int]] = []
    self._render_rgb: list[bool] = []

    self._output_names_rgb: list[str] = []
    self._output_names_depth: list[str] = []
    self._measurements_spec: dict[str, specs.Array] = {}
    self._measurement_names: dict[str, str] = {}
    self._render_graph = None

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Initializes the entity with compiled model and data."""
    sim_dt = simulator.clock.sim_dt
    desired_period = 1.0 / self._render_rate_hz
    self._update_interval_in_substeps = max(1, round(desired_period / sim_dt))
    actual_period = self._update_interval_in_substeps * sim_dt
    if not math.isclose(
        actual_period, desired_period, rel_tol=1e-6, abs_tol=1e-9
    ):
      _logger.warning(
          'Desired render_rate_hz=%g (period %.6fs) is not an integer multiple'
          ' of sim_dt=%.6fs; discretizing to %d substeps (actual render rate'
          ' %.2f Hz).',
          self._render_rate_hz,
          desired_period,
          sim_dt,
          self._update_interval_in_substeps,
          1.0 / actual_period,
      )
    mj_spec = simulator.mj_spec
    mj_model = simulator.mj_model
    data = simulator.data
    cam_names, cam_active = _cam_names_to_mask(mj_spec, self._raw_cam_name)
    ncam = len(cam_names)

    # Map user camera order to mj_spec order.
    raw_names = (
        [self._raw_cam_name]
        if isinstance(self._raw_cam_name, str)
        else list(self._raw_cam_name or cam_names)
    )
    order = [raw_names.index(name) for name in cam_names]

    # Normalize inputs to lists matching selected cameras.
    cam_resolution = self._raw_cam_resolution
    if isinstance(cam_resolution, tuple):
      cam_resolution = [cam_resolution] * ncam
    elif cam_resolution is None:
      cam_resolution = [(640, 480)] * ncam
    elif len(cam_resolution) != ncam:
      raise ValueError(
          f'cam_resolution has {len(cam_resolution)} entries but {ncam} cameras'
      )
    else:
      cam_resolution = [cam_resolution[i] for i in order]

    render_rgb = self._raw_render_rgb
    if isinstance(render_rgb, bool):
      render_rgb = [render_rgb] * ncam
    elif len(render_rgb) != ncam:
      raise ValueError(
          f'render_rgb has {len(render_rgb)} entries but {ncam} cameras'
      )
    else:
      render_rgb = [render_rgb[i] for i in order]

    render_depth = self._raw_render_depth
    if isinstance(render_depth, bool):
      render_depth = [render_depth] * ncam
    elif len(render_depth) != ncam:
      raise ValueError(
          f'render_depth has {len(render_depth)} entries but {ncam} cameras'
      )
    else:
      render_depth = [render_depth[i] for i in order]

    self._render_context_kwargs['cam_res'] = cam_resolution
    self._render_context_kwargs['render_rgb'] = render_rgb
    self._render_context_kwargs['render_depth'] = render_depth
    self._render_context_kwargs['cam_active'] = cam_active
    self._ncam = ncam
    self.cam_name = cam_names
    self._cam_resolution = cam_resolution
    self._render_depth = render_depth

    output_name_prefix = self._output_name_prefix or ''
    self._output_names_rgb = [
        f'{output_name_prefix}{name}{self._rgb_output_name_suffix}'
        for name in cam_names
    ]
    self._output_names_depth = [
        f'{output_name_prefix}{name}{self._depth_output_name_suffix}'
        for name in cam_names
    ]

    measurements_spec, measurement_names_dict = {}, {}
    for i in range(ncam):
      width, height = cam_resolution[i]
      cam_base_name = cam_names[i]

      if render_rgb[i]:
        rgb_output_name = self._output_names_rgb[i]
        measurements_spec[rgb_output_name] = specs.Array(
            (height, width, 3), dtype=np.uint8
        )
        measurement_names_dict[f'{cam_base_name}_rgb'] = rgb_output_name
      if render_depth[i]:
        depth_output_name = self._output_names_depth[i]
        measurements_spec[depth_output_name] = specs.Array(
            (height, width),
            dtype=types.TORCH_TO_NUMPY_DTYPE[self._depth_dtype],
        )
        measurement_names_dict[f'{cam_base_name}_depth'] = depth_output_name
    self._measurements_spec = measurements_spec
    self._measurement_names = measurement_names_dict

    self._nworld = nworld = data.nworld
    self._render_context_kwargs.pop('nworld', None)
    self._render_context = render_ctx = mjw.create_render_context(
        mj_model,
        nworld=nworld,
        **self._render_context_kwargs,
    )
    self._wp_device = render_ctx.rgb_data.device
    self._rgb_adr = render_ctx.rgb_adr.numpy()
    self._depth_adr = render_ctx.depth_adr.numpy()
    rc_res = render_ctx.cam_res.numpy()
    self._pixel_count = rc_res[:, 0] * rc_res[:, 1]
    self._render_rgb_list = render_ctx.render_rgb.numpy().tolist()
    self._render_depth_list = render_ctx.render_depth.numpy().tolist()

    assert self._ncam == len(self._render_rgb_list)
    assert self._ncam == len(self._render_depth_list)

    # buffers
    if any(self._render_rgb_list):
      self._rgb_unpacked = wp.array3d(
          shape=(nworld, render_ctx.rgb_data.shape[1], 3),
          dtype=wp.uint8,
          device=self._wp_device,
      )
    else:
      self._rgb_unpacked = None

    # Register graph capture callback for TorchWrapper and capture
    # initial render graph.
    simulator.model.register_graph_capture_callback(
        lambda: setattr(self, '_render_graph', None)
    )
    self.render(simulator.model, data)

  def render(
      self,
      model: types.Model | None = None,
      data: types.Data | None = None,
  ) -> None:
    """Render all cameras (called once per scene.update)."""
    if self._render_graph is None:
      with wp.ScopedDevice(self._wp_device):
        with wp.ScopedCapture() as capture:
          mjw.refit_bvh(model.obj, data.obj, self._render_context)  # pyrefly: ignore[missing-attribute]
          mjw.render(model.obj, data.obj, self._render_context)  # pyrefly: ignore[missing-attribute]
          if any(self._render_rgb_list) and self._rgb_unpacked is not None:
            wp.launch(
                _unpack_rgb_kernel,
                dim=(self._nworld, self._render_context.rgb_data.shape[1]),
                inputs=[self._render_context.rgb_data],
                outputs=[self._rgb_unpacked],
                device=self._wp_device,
            )
      self._render_graph = capture.graph

    with wp.ScopedDevice(self._wp_device):
      wp.capture_launch(self._render_graph)

  @property
  def name(self) -> str:
    return self._name

  @property
  def measurement_names(self) -> dict[str, str]:
    return self._measurement_names

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Resets the processor."""
    del simulator
    if env_mask is None:
      self._last_measurements = {}
      self._last_rendered_substep = -1
    else:
      # NOTE: Given that reset() currently happens at the end of a step, to
      # avoid calling render twice in a step, we zero out the measurements
      # instead of re-rendering.
      # TODO (shaves, btaba): Revisit whether we can reset at the beginning of
      # a step instead to leverage the existing render() call.
      for v in self._last_measurements.values():
        v[env_mask] = 0

  def substep_measurements_spec(
      self,
  ) -> Mapping[str, specs.Array]:
    """Spec for measurements generated once per step."""
    return self._measurements_spec

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Renders the scene using MuJoCo rendering."""
    global_substep = simulator.clock.global_substep
    if not self._last_measurements or (
        self._last_rendered_substep != global_substep
        and (global_substep % self._update_interval_in_substeps) == 0
    ):
      self.render(simulator.model, simulator.data)

      measurements = {}
      for i in range(self._ncam):
        if self._render_rgb_list[i] and self._rgb_unpacked is not None:
          rgb_unpacked_torch = wp.to_torch(
              self._rgb_unpacked  # pyrefly: ignore[bad-argument-type]
          )
          start = int(self._rgb_adr[i])
          size = int(self._pixel_count[i])
          width, height = self._cam_resolution[i]
          rgb_flat = rgb_unpacked_torch[:, start : start + size]
          measurements[self._output_names_rgb[i]] = rgb_flat.reshape(
              self._nworld,
              height,
              width,
              3,
          )

        if self._render_depth_list[i]:
          depth_torch = wp.to_torch(self._render_context.depth_data)
          start_d = int(self._depth_adr[i])
          size = int(self._pixel_count[i])
          width, height = self._cam_resolution[i]
          depth_flat = depth_torch[:, start_d : start_d + size]
          depth = depth_flat.reshape(
              self._nworld,
              height,
              width,
          )
          if self._depth_scale is not None:
            depth = (
                torch.clip(depth / max(self._depth_scale, 1e-6), 0.0, 1.0)
                * 255.0
            )
          if depth.dtype != self._depth_dtype:
            depth = depth.to(self._depth_dtype)
          measurements[self._output_names_depth[i]] = depth

      self._last_measurements = measurements
      self._last_rendered_substep = global_substep

    return self._last_measurements


@wp.kernel
def _unpack_rgb_kernel(
    packed: wp.array2d(dtype=wp.uint32),  # pyrefly: ignore[invalid-annotation]
    rgb: wp.array3d(dtype=wp.uint8),  # pyrefly: ignore[invalid-annotation]
):
  """Unpack packed uint32 RGB into uint8 buffer for all cameras."""
  wid, pid = wp.tid()  # type: ignore
  b = wp.uint8(packed[wid, pid] & wp.uint32(0xFF))
  g = wp.uint8((packed[wid, pid] >> wp.uint32(8)) & wp.uint32(0xFF))
  r = wp.uint8((packed[wid, pid] >> wp.uint32(16)) & wp.uint32(0xFF))
  rgb[wid, pid, 0] = r
  rgb[wid, pid, 1] = g
  rgb[wid, pid, 2] = b


def _cam_names_to_mask(
    mj_spec: mujoco.MjSpec,
    cam_name: Union[list[str] | str] | None,
) -> tuple[list[str], list[bool]]:
  """Converts camera name(s) to selected names and active mask in mj_spec order.

  Args:
    mj_spec: MuJoCo spec containing camera definitions.
    cam_name: Camera name(s) to activate. None means all cameras.

  Returns:
    selected_names: List of selected camera names in mj_spec order.
    cam_active: Boolean mask of length ncam (True for selected cameras).
  """
  all_names = [cam.name for cam in mj_spec.cameras]
  if cam_name is None:
    raw_names = all_names
  elif isinstance(cam_name, str):
    raw_names = [cam_name]
  else:
    raw_names = list(cam_name)

  for name in raw_names:
    if name not in all_names:
      raise ValueError(f"Camera '{name}' not found in model.")

  if len(raw_names) != len(set(raw_names)):
    raise ValueError(f'Duplicate camera names found in cam_name: {raw_names}')

  name_set = set(raw_names)
  selected_names = [n for n in all_names if n in name_set]
  cam_active = [n in name_set for n in all_names]
  return selected_names, cam_active
