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
"""Physics Simulator.

Encapsulates MuJoCo / MuJoCo-Warp simulation state and stepping.
"""

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Union

import mujoco
import mujoco_warp as mjw
from reaf.simulation.torch._src.core import clock as clock_module
from reaf.simulation.torch._src.core import protocols
from reaf.simulation.torch._src.core import torch_wrapper
from reaf.simulation.torch._src.core import types
import torch
import warp as wp


class Simulator(protocols.Simulator):
  """MuJoCo / MuJoCo-Warp physics simulator.

  Encapsulates:
    - MuJoCo MjSpec and compiled MjModel.
    - Raw and Torch-wrapped Model and Data tensors.
    - Physics stepping, forward kinematics, and state resets.
    - CUDA graph capture management for Warp backend.
  """

  def __init__(
      self,
      *,
      mj_spec: mujoco.MjSpec,
      nworld: int = 1,
      xpu_device: str = "cpu",
      impl: Union[types.Impl, str] = types.Impl.CPP,
      nconmax: Optional[int] = None,
      njmax: Optional[int] = None,
      nccdmax: Optional[int] = None,
      nvmax: Optional[int] = None,
  ):
    impl = types.Impl(impl)

    if nvmax is not None and impl != types.Impl.WARP:
      raise ValueError(
          f"nvmax is only supported for types.Impl.WARP, got {impl}."
      )

    if impl == types.Impl.CPP:
      if nworld > 1:
        raise NotImplementedError(
            "Multiple environments are not supported for types.Impl.CPP."
        )
      if xpu_device != "cpu":
        raise NotImplementedError("types.Impl.CPP is only supported on CPU.")

    self._nworld = nworld
    self._nconmax = nconmax
    self._njmax = njmax
    self._nccdmax = nccdmax
    self._nvmax = nvmax
    self._xpu_device = xpu_device
    self._impl = impl
    self._mj_spec = mj_spec

    xpu_device_ = wp.get_device(self._xpu_device)
    self._use_cuda_graph = xpu_device_.is_cuda and wp.is_mempool_enabled(
        xpu_device_
    )
    self._graph_captures: dict[str, wp.Graph] = {}
    self._set_const_graph_captures: dict[
        torch_wrapper.SetConstLevel, wp.Graph
    ] = {}

    self._model: Any = None
    self._data: Any = None
    self._mj_data = None
    self._mj_model = None
    self._model_torch = None
    self._data_torch = None
    self._clock: Any = None

    self._reset_bool = torch.zeros(
        self._nworld, dtype=torch.bool, device=self._xpu_device
    )
    self._reset_wp_array: Any = wp.from_torch(self._reset_bool, dtype=wp.bool)

    # Compile the MuJoCo model and data upon initialization.
    self.compile()

  def compile(self) -> None:
    """Compiles MjSpec to MjModel/MjData and allocates backend models."""
    self._mj_model = self._mj_spec.compile()
    self._mj_data = mujoco.MjData(self._mj_model)

    mujoco.mj_forward(self._mj_model, self._mj_data)

    # Put the model and data on device.
    if self._impl == types.Impl.CPP:
      self._model = self._mj_model
      self._data = self._mj_data
    elif self._impl == types.Impl.WARP:
      with wp.ScopedDevice(self._xpu_device):
        self._model = mjw.put_model(self._mj_model)
        self._data = mjw.put_data(
            self._mj_model,
            self._mj_data,
            nworld=self._nworld,
            nconmax=self._nconmax,
            njmax=self._njmax,
            nccdmax=self._nccdmax,
            nvmax=self._nvmax,
        )
    else:
      raise ValueError(f"Unsupported backend: {self._impl}")

    self._mj_model = self._mj_model
    self._graph_captures.clear()
    self._set_const_graph_captures.clear()
    self._model_torch = torch_wrapper.TorchWrapper(
        self._model, nworld=self._nworld
    )
    self._model_torch.register_graph_capture_callback(
        self._graph_captures.clear
    )
    self._model_torch.register_graph_capture_callback(
        self._set_const_graph_captures.clear
    )
    self._data_torch = torch_wrapper.TorchWrapper(
        self._data, nworld=self._nworld
    )

    sim_dt = float(self._mj_model.opt.timestep)
    self._clock = clock_module.Clock(
        sim_dt=sim_dt,
        control_dt=sim_dt,
        n_substeps=1,
        nworld=self._nworld,
        xpu_device=self._xpu_device,
    )

  @property
  def clock(self) -> clock_module.Clock:
    assert self._clock is not None
    return self._clock

  def set_clock(self, clock: clock_module.Clock) -> None:
    if clock is None:
      raise ValueError("Clock must not be None.")
    self._clock = clock

  @property
  def model(self) -> types.Model:
    assert self._model_torch is not None
    return self._model_torch

  @property
  def data(self) -> types.Data:
    assert self._data_torch is not None
    return self._data_torch

  @property
  def warp_model(self) -> mjw.Model:
    assert self._impl == types.Impl.WARP
    assert self._model is not None
    return self._model

  @property
  def warp_data(self) -> mjw.Data:
    assert self._impl == types.Impl.WARP
    assert self._data is not None
    return self._data

  @property
  def mj_spec(self) -> mujoco.MjSpec:
    return self._mj_spec

  @property
  def mj_model(self) -> mujoco.MjModel:
    assert self._mj_model is not None
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    assert self._mj_data is not None
    return self._mj_data

  @property
  def impl(self) -> types.Impl:
    return self._impl

  @property
  def nworld(self) -> int:
    return self._nworld

  @property
  def xpu_device(self) -> str:
    return self._xpu_device

  @property
  def nvmax(self) -> Optional[int]:
    return self._nvmax

  def forward(self) -> None:
    """Computes forward kinematics and dynamics."""
    assert self._model is not None and self._data is not None
    if self._impl == types.Impl.CPP:
      mujoco.mj_forward(self._model, self._data)
    elif self._impl == types.Impl.WARP:
      self._maybe_capture_graphs()
      with wp.ScopedDevice(self._xpu_device):
        if self._use_cuda_graph:
          wp.capture_launch(self._graph_captures["forward"])
        else:
          mjw.forward(self._model, self._data)

  def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
    """Resets simulation data for specified environments."""
    assert self._model is not None and self._data is not None

    self._maybe_capture_graphs()
    if self._impl == types.Impl.CPP:
      mujoco.mj_resetData(self._model, self._data)
    else:
      if env_mask is None:
        self._reset_bool[:] = True
      else:
        self._reset_bool[:] = env_mask
      assert self._reset_bool.data_ptr() == self._reset_wp_array.ptr
      with wp.ScopedDevice(self._xpu_device):
        if self._use_cuda_graph:
          wp.capture_launch(self._graph_captures["reset_data"])
        else:
          mjw.reset_data(self._model, self._data, self._reset_wp_array)

    self._clock.reset(env_mask)

  def step(self) -> None:
    """Executes a single low-level physics step."""
    assert self._model is not None and self._data is not None
    if self._impl == types.Impl.CPP:
      mujoco.mj_step(self._model, self._data)
    elif self._impl == types.Impl.WARP:
      self._maybe_capture_graphs()
      with wp.ScopedDevice(self._xpu_device):
        if self._use_cuda_graph:
          wp.capture_launch(self._graph_captures["step"])
        else:
          mjw.step(self._model, self._data)

    self._clock.tick_substep()

  def step_n(self, n_substeps: int) -> None:
    for _ in range(n_substeps):
      self.step()

  def _maybe_capture_graphs(self) -> None:
    """Captures CUDA graphs for Warp operations if supported."""
    assert self._model is not None and self._data is not None
    if self._graph_captures or not self._use_cuda_graph:
      return

    with wp.ScopedDevice(self._xpu_device):
      with wp.ScopedCapture(device=self._xpu_device) as capture:
        mjw.step(self._model, self._data)
      self._graph_captures["step"] = capture.graph

      with wp.ScopedCapture(device=self._xpu_device) as capture:
        mjw.forward(self._model, self._data)
      self._graph_captures["forward"] = capture.graph

      with wp.ScopedCapture(device=self._xpu_device) as capture:
        mjw.reset_data(self._model, self._data, self._reset_wp_array)
      self._graph_captures["reset_data"] = capture.graph

  def _maybe_capture_set_const_graph(
      self, level: torch_wrapper.SetConstLevel
  ) -> None:
    """Captures a CUDA graph for `mjw.set_const` at `level` if supported."""
    assert self._model is not None and self._data is not None
    if level in self._set_const_graph_captures or not self._use_cuda_graph:
      return

    with wp.ScopedDevice(self._xpu_device):
      with wp.ScopedCapture(device=self._xpu_device) as capture:
        mjw.set_const(self._model, self._data, groups=int(level), restore=False)
      self._set_const_graph_captures[level] = capture.graph

  def set_const(self) -> torch_wrapper.SetConstLevel:
    """Recomputes derived model constants after primary model fields changed.

    Flushes the `SetConstLevel` tracked by `model` (see
    `TorchWrapper.flush_set_const`) and runs the matching `mj_setConst` stages
    for all worlds. Only in-place writes through `self.model` (`TorchWrapper`)
    are tracked; direct modifications to the base models (`mj_model`,
    `warp_model`, `._tensor`, or `._array`) are not tracked. This is a no-op if
    no tracked model field was written since the last call. The Warp backend
    always runs with `restore=False`, so derived data fields (e.g. `xpos`, `qM`)
    are left at `qpos0` for all worlds; callers must call `forward()` before
    reading data.

    Returns:
      The `SetConstLevel` that was run.
    """
    assert self._model is not None and self._data is not None
    assert self._model_torch is not None
    level = self._model_torch.flush_set_const()
    if level == torch_wrapper.SetConstLevel.NONE:
      return level
    if self._impl == types.Impl.CPP:
      qpos = self._data.qpos.copy()
      mujoco.mj_setConst(self._model, self._data)
      self._data.qpos[:] = qpos
      return level
    self._maybe_capture_set_const_graph(level)
    with wp.ScopedDevice(self._xpu_device):
      if self._use_cuda_graph:
        wp.capture_launch(self._set_const_graph_captures[level])
      else:
        mjw.set_const(self._model, self._data, groups=int(level), restore=False)
    return level
