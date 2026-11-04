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

"""Camera entity using Python mujoco.Renderer directly (no FFI).

As opposed to mujoco_camera_device.py, this entity loops over the render calls
in a single entity rather than split across multiple entities, which
improves raw throughput of rendering with multiple cameras.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import logging
import math

from dm_env import specs
import mujoco
import mujoco_warp as mjw
import numpy as np
from reaf.simulation.torch._src.common import mujoco_camera_device
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch

_logger = logging.getLogger(__name__)


class BatchedMujocoCameraDevice(device.Device):
  """Renders multiple cameras using Python mujoco.Renderer.

  Creates its own render context with singleton pattern per environment.
  """

  def __init__(
      self,
      *,
      name: str = '',
      camera_configs: Sequence[dict],
      render_rate_hz: float | None,
      width: int = 128,
      height: int = 128,
      scene_option: mujoco.MjvOption | None = None,
  ):
    """Initialize camera entity.

    Args:
      name: Entity name.
      camera_configs: List of dicts with 'camera_name' and 'output_name' keys,
        e.g. [{'camera_name': 'front', 'output_name': 'front'}, {'camera_name':
        'wrist', 'output_name': 'wrist'}]. Each entry renders the named camera
        and produces '{output_name}_rgb' in the measurements dict.
      render_rate_hz: Camera rendering rate in Hz, or None to render only the
        first measurement after each reset. Must be positive if set.
      width: Image width.
      height: Image height.
      scene_option: MuJoCo visualization options.
    """
    if render_rate_hz is not None and render_rate_hz <= 0:
      raise ValueError(
          f'render_rate_hz must be positive or None, got {render_rate_hz}'
      )

    self._name = name
    self._camera_configs = list(camera_configs)
    self._render_rate_hz = render_rate_hz
    self._width = width
    self._height = height
    self._update_interval_in_substeps = 1
    self._scene_option = scene_option or mujoco.MjvOption()

    # Build specs for all cameras
    self._measurements_spec: dict[str, specs.Array] = {}
    for cfg in camera_configs:
      output_name = cfg['output_name']
      self._measurements_spec[f'{output_name}_rgb'] = specs.Array(
          shape=(height, width, 3), dtype=np.uint8
      )

    self._last_measurements: dict[str, torch.Tensor] = {}
    self._last_rendered_substep: int = -1

    # Initialized in initialize()
    self._mj_model: mujoco.MjModel | None = None
    self._mj_data: mujoco.MjData | None = None
    self._renderer: mujoco.Renderer | None = None
    self._camera_ids: list[int] = []
    self._xpu_device: str = 'cpu'
    self._impl: types.Impl | None = None

  @property
  def name(self) -> str:
    return self._name

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    """Initialize entity with compiled model - creates renderer."""
    if self._render_rate_hz is not None:
      sim_dt = simulator.clock.sim_dt
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
    self._mj_model = simulator.mj_model
    self._xpu_device = simulator.xpu_device
    if isinstance(simulator.model.obj, mjw.Model):
      self._impl = types.Impl.WARP
    else:
      self._impl = types.Impl.CPP

    # Create renderer for this entity
    self._renderer = mujoco.Renderer(
        simulator.mj_model, height=self._height, width=self._width
    )

    # For WARP impl, create host-side MjData for kinematics
    if self._impl == types.Impl.WARP:
      self._mj_data = mujoco.MjData(self._mj_model)

    # Get camera IDs from compiled model
    self._camera_ids = []
    for cfg in self._camera_configs:
      cam_id = self._mj_model.camera(cfg['camera_name']).id
      self._camera_ids.append(cam_id)

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator
    if env_mask is None or bool(env_mask[0]):
      self._last_measurements = {}
      self._last_rendered_substep = -1

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return self._measurements_spec

  def get_substep_measurements(
      self,
      simulator: simulator_module.Simulator,
  ) -> Mapping[str, torch.Tensor]:
    """Render all cameras using Python mujoco.Renderer."""
    if self._renderer is None:
      raise RuntimeError('Device not initialized')

    global_substep = simulator.clock.global_substep
    if not self._last_measurements or (
        self._render_rate_hz is not None
        and self._last_rendered_substep != global_substep
        and (global_substep % self._update_interval_in_substeps) == 0
    ):
      self._last_measurements = self._render(simulator)
      self._last_rendered_substep = global_substep

    return self._last_measurements

  def render_now(
      self,
      simulator: simulator_module.Simulator,
      output_names: Sequence[str] | None = None,
  ) -> dict[str, torch.Tensor]:
    """Renders cameras immediately without affecting device measurements.

    Side-channel call for `Impl.CPP` single-world environments that renders from
    the current simulator state and returns only the requested cameras, without
    modifying cached substep measurements or the `render_rate_hz` schedule.

    Args:
      simulator: The simulator instance.
      output_names: Optional subset of camera `output_name`s to render. `None`
        renders all cameras configured on this device.

    Returns:
      A dictionary mapping `{output_name}_rgb` to the freshly rendered tensor
      for each requested camera.

    Raises:
      RuntimeError: If the device has not been initialized.
      ValueError: If `self._impl != types.Impl.CPP`, `data.nworld` is greater
        than 1, or `output_names` is a bare string, empty, or contains an
        unknown camera.
    """
    if self._renderer is None or self._impl is None:
      raise RuntimeError('Device not initialized')

    if self._impl != types.Impl.CPP:
      raise ValueError(f'render_now only supports Impl.CPP, got {self._impl}')

    target_names: set[str] | None = None
    if output_names is not None:
      # In Python typing, `str` satisfies `Sequence[str]`.
      if isinstance(output_names, str) or not output_names:
        raise ValueError(
            'output_names must be a non-empty sequence of camera output names.'
        )

      known = {cfg['output_name'] for cfg in self._camera_configs}
      unknown = set(output_names) - known
      if unknown:
        raise ValueError(
            f'Unknown camera output names {sorted(unknown)}; expected a subset'
            f' of {sorted(known)}.'
        )
      target_names = set(output_names)

    return self._render(simulator, target_names)

  def _render(
      self,
      simulator: simulator_module.Simulator,
      output_names: set[str] | None = None,
  ) -> dict[str, torch.Tensor]:
    """Renders `output_names` (all cameras if `None`) from the current state."""
    renderer = self._renderer
    if renderer is None or self._impl is None or self._mj_model is None:
      raise RuntimeError('Device not initialized')

    data = simulator.data
    if data.nworld > 1:
      raise ValueError(
          f'BatchedMujocoCameraDevice only supports nworld=1, got {data.nworld}'
      )

    mj_data = mujoco_camera_device.get_scene_data(
        data=data,
        impl=self._impl,
        model=self._mj_model,
        scratch_data=self._mj_data,
    )

    measurements = {}
    for cfg, cam_id in zip(self._camera_configs, self._camera_ids):
      output_name = cfg['output_name']
      if output_names is not None and output_name not in output_names:
        continue

      renderer.update_scene(
          mj_data, camera=cam_id, scene_option=self._scene_option
      )
      img = renderer.render()
      tensor = torch.from_numpy(img).unsqueeze(0)
      if self._impl == types.Impl.WARP:
        tensor = tensor.to(self._xpu_device)
      measurements[f'{output_name}_rgb'] = tensor

    return measurements

  def substep_commands_spec(self) -> Mapping[str, specs.Array]:
    return {}

  def set_substep_commands(
      self,
      simulator: simulator_module.Simulator,
      substep_commands: Mapping[str, torch.Tensor],
  ) -> None:
    del simulator, substep_commands
