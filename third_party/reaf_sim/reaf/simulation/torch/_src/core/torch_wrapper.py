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
"""Wraps a mujoco warp simulation to expose the arrays as torch tensors.

When model fields are mutated through the `TorchTensor` wrapper on `TorchModel`,
the wrapper tracks the writes and marks the required `SetConstLevel` on
`model.pending_set_const`. The owning `Environment` should flush
these pending constants via `Simulator.set_const()` before a forward call.
"""

import contextlib
import dataclasses
import enum
import functools
from typing import Any
from typing import Callable
from typing import Generic
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

import mujoco
from mujoco.introspect import structs as mj_structs
import mujoco_warp as mjw
import numpy as np
from reaf.simulation.torch._src.core import mujoco_types
import torch
import warp as wp


class SetConstLevel(enum.IntFlag):
  """Determines which mj_setConst levels must run after modifying the model.

  Levels are defined as bitmask flags to support independent compilation stages.
  """

  NONE = mjw.SetConstGroup.NONE  # No constants need updating.
  FIXED = mjw.SetConstGroup.FIXED  # Configuration-independent constants.
  KIN0 = mjw.SetConstGroup.KIN0  # Kinematic constants at qpos0.
  INERT0 = mjw.SetConstGroup.INERT0  # Inertial constants at qpos0.
  FULL = mjw.SetConstGroup.ALL  # All constants.


T = TypeVar("T")


# pylint: disable=protected-access
def _set_const_output_fields(group: str) -> frozenset[str]:
  """Returns batched Model output fields for a set_const group."""
  idx = mujoco.MjModel._setconst_groups.index(group)
  return frozenset(
      name
      for name, access in mujoco.MjModel._setconst_fields.items()
      if (access[idx] & mujoco.mjSC_OUT)
      and isinstance(ann := mjw.Model.__annotations__.get(name), wp.array)
      and ann.shape[0] in ("*", "nworld")
  )


def _set_const_input_levels() -> dict[str, SetConstLevel]:
  """Returns the mapping from Model input fields to SetConstLevel."""
  # Direct mjSC_IN bits per group suffice for mjwarp: FIXED only writes
  # body_subtreemass and geom bounds (no body_sameframe or ten_J sparsity), and
  # the inputs whose FIXED outputs are read by KIN0/INERT0 in mjwarp
  # (body_mass -> body_subtreemass, geom_size -> cylinder/sphere tendon wrap)
  # already carry direct mjSC_IN bits in KIN0 and INERT0. Taking a group-level
  # closure over FIXED in the C table would over-promote FIXED-only inputs.
  groups = (
      (mujoco.MjModel._setconst_groups.index("fixed"), SetConstLevel.FIXED),
      (mujoco.MjModel._setconst_groups.index("kin0"), SetConstLevel.KIN0),
      (mujoco.MjModel._setconst_groups.index("inert0"), SetConstLevel.INERT0),
  )
  levels: dict[str, SetConstLevel] = {}
  for name in mujoco.MjModel._setconst_inputs:
    access = mujoco.MjModel._setconst_fields[name]
    level = SetConstLevel.NONE
    for idx, group_level in groups:
      if access[idx] & mujoco.mjSC_IN:
        level |= group_level
    if level != SetConstLevel.NONE:
      levels[name] = level
  return levels


# pylint: enable=protected-access


# Model fields that are the outputs of mjw.set_const(..., groups=FIXED).
SET_CONST_FIXED_OUTPUT_FIELDS: frozenset[str] = _set_const_output_fields(
    "fixed"
)
# Model fields that are the outputs of mjw.set_const(..., groups=KIN0).
SET_CONST_KIN0_OUTPUT_FIELDS: frozenset[str] = _set_const_output_fields("kin0")
# Model fields that are the outputs of mjw.set_const(..., groups=INERT0).
SET_CONST_INERT0_OUTPUT_FIELDS: frozenset[str] = _set_const_output_fields(
    "inert0"
)
# Model fields that are the outputs of mjw.set_const.
SET_CONST_OUTPUT_FIELDS: frozenset[str] = (
    SET_CONST_FIXED_OUTPUT_FIELDS
    | SET_CONST_KIN0_OUTPUT_FIELDS
    | SET_CONST_INERT0_OUTPUT_FIELDS
)

SET_CONST_INPUT_LEVELS: dict[str, SetConstLevel] = _set_const_input_levels()


def _has_geom_tendon_wrap(obj: object) -> bool:
  """Returns True if the model has sphere or cylinder tendon wrap objects."""
  if isinstance(obj, mjw.Model):
    return obj.wrap_geom_adr.size > 0
  if isinstance(obj, mujoco.MjModel):
    return bool(
        np.isin(
            obj.wrap_type,
            (
                int(mujoco.mjtWrap.mjWRAP_SPHERE),
                int(mujoco.mjtWrap.mjWRAP_CYLINDER),
            ),
        ).any()
    )
  return True


def _get_set_const_level(field: str, obj: object) -> SetConstLevel:
  """Returns the SetConstLevel required after modifying a model field."""
  level = SET_CONST_INPUT_LEVELS.get(field, SetConstLevel.NONE)
  if field == "geom_size" and not _has_geom_tendon_wrap(obj):
    level &= ~(SetConstLevel.KIN0 | SetConstLevel.INERT0)
  return level


def _create_set_const_callback(
    wrapper: "TorchWrapper[Any]", field_name: str
) -> Callable[[], None] | None:
  """Returns an on_write callback that tracks defaults and set_const for model fields."""
  # pylint: disable=protected-access
  if wrapper._obj_type not in (MjObjType.MJW_MODEL, MjObjType.MJ_MODEL):
    return None
  level = _get_set_const_level(field_name, wrapper._obj)

  def _record():
    # Clone the field once before its first write so restore_defaults() can
    # reset it in-place without re-cloning on subsequent writes or resets.
    if field_name not in wrapper._defaults:
      wrapper._defaults[field_name] = getattr(
          wrapper, field_name
      )._tensor.clone()
    if level != SetConstLevel.NONE:
      object.__setattr__(
          wrapper, "_pending_set_const", wrapper._pending_set_const | level
      )

  # pylint: enable=protected-access
  return _record


# pylint: disable=protected-access
def _add_torch_operators(cls):
  """Adds pytorch operators to a class."""
  two_sided_ops = ("add", "sub", "mul", "div", "pow")
  one_sided_ops = ("eq", "ne", "lt", "le", "gt", "ge")

  for op in one_sided_ops + two_sided_ops:
    torch_fn = getattr(torch, op)
    setattr(cls, f"__{op}__", lambda s, o, func=torch_fn: func(s._tensor, o))
  for op in two_sided_ops:
    torch_fn = getattr(torch, op)
    setattr(cls, f"__r{op}__", lambda s, o, func=torch_fn: func(o, s._tensor))

  setattr(cls, "__pos__", lambda s: +s._tensor)
  setattr(cls, "__neg__", lambda s: -s._tensor)
  setattr(cls, "__abs__", lambda s: torch.abs(s._tensor))
  setattr(cls, "__truediv__", lambda s, o: s._tensor / o)
  setattr(cls, "__rtruediv__", lambda s, o: o / s._tensor)
  return cls


# pylint: enable=protected-access


class NoSetAttr:

  def __setattr__(self, name: str, value: Any) -> None:
    raise AttributeError(
        "This wrapper is read-only so that Warp array pointers are kept"
        " immutable for usage in CUDA graphs. Use `x[...] = value` instead."
    )


def _wp_to_torch(array: wp.array) -> torch.Tensor:
  """Converts a Warp array to a PyTorch tensor."""
  if array.size == 0 and array.device is not None and array.device.is_cpu:
    # handle empty arrays on CPU where wp.to_torch fails
    scalar_type = getattr(array.dtype, "_wp_scalar_type_", array.dtype)
    torch_dtype = wp.dtype_to_torch(scalar_type)
    dtype_shape = getattr(array.dtype, "_shape_", ())
    return torch.empty(
        array.shape + dtype_shape, dtype=torch_dtype, device="cpu"
    )
  return wp.to_torch(array)


@_add_torch_operators
class TorchTensor(NoSetAttr):
  """Wraps a warp/numpy array to expose it as a torch.Tensor.

  If `on_write` is set, it is called before explicit write operations. This is
  typically used to determine when to call `set_const` after modifying Model
  fields.

  Tracked operation on TorchTensor:

  - Subscript assignment: `model.<field>[idx] = val` via `__setitem__`
  - Bound attribute assignment: `bound.set("field", val)` (uses `__setitem__`)

  Write operations that bypass the wrapper are NOT tracked:

  - Direct writes to underlying storage: `t._tensor[idx] = val` or
    `t._array[idx] = val`
  - In-place ops on views from `t[idx]`, e.g. `t[idx] += val`
  - In-place tensor methods: `t.copy_()`, `t.add_()`, `t.mul_()`, `t.zero_()`,
    etc.
  - In-place torch module functions: `torch.add_(t, ...)`,
    `torch.clamp_(t, ...)`
  - Functions writing to `out=` destinations: `torch.add(..., out=t)`, etc.
  - Direct mutation of raw Warp / C++ models: `sim.warp_model.<field>`,
    `sim.mj_model.<field>`.

  Access to underlying storage is private (`_tensor` and `_array`). Do not
  access or mutate `_tensor` or `_array` directly.
  """

  def __init__(
      self,
      array: wp.array | np.ndarray,
      add_batch_dim: bool = False,
      view_shape: tuple[int, ...] | None = None,
      nworld: int | None = None,
      on_write: Callable[[], None] | None = None,
  ):
    if isinstance(array, np.ndarray):
      # Use torch.from_numpy directly for zero-copy.
      tensor = torch.from_numpy(array)
    else:
      tensor = _wp_to_torch(array)

    if add_batch_dim:
      tensor = tensor.unsqueeze(0)

    if view_shape is not None:
      tensor = tensor.view(view_shape)

    object.__setattr__(self, "_array", array)
    object.__setattr__(self, "_torch_tensor", tensor)
    object.__setattr__(self, "_nworld", nworld)
    object.__setattr__(self, "_on_write", on_write)

  @property
  def _tensor(self) -> torch.Tensor:
    return self._torch_tensor

  def _get_torch_stream(self) -> Optional[torch.cuda.Stream]:
    """Creates a torch stream from warp device stream."""
    if isinstance(self._array, np.ndarray) or self._array.device.is_cpu:
      return None
    stream = wp.get_stream(self._array.device)
    return torch.cuda.ExternalStream(
        stream.cuda_stream, device=self._array.device.ordinal
    )

  @property
  def device(self) -> torch.device:
    return self._tensor.device

  def __repr__(self) -> str:
    return repr(self._tensor)

  def __getitem__(self, idx: Any) -> Any:
    return self._tensor[idx]

  def __setitem__(self, idx: Any, value: Any):
    if (
        isinstance(self._array, np.ndarray)
        and isinstance(value, torch.Tensor)
        and value.dtype != self._tensor.dtype
    ):
      # Convert dtype to match the destination only when wrapping numpy arrays.
      value = value.to(self._tensor.dtype)

    if isinstance(self._array, np.ndarray) or self._array.device.is_cpu:
      self._maybe_call_on_write()
      self._tensor[idx] = value
      return

    if (
        _array_has_batched_attr(self._array)
        and self._nworld is not None
        and self._array.shape[0] != self._nworld
    ):
      raise AssertionError(
          "Underlying Warp array is a batched field but does "
          "not have leading dimension of size nworld."
      )

    with torch.cuda.stream(self._get_torch_stream()):
      self._maybe_call_on_write()
      self._tensor[idx] = value

  def _maybe_call_on_write(self) -> None:
    """Calls the `on_write` callback if it exists."""
    if self._on_write is not None:
      self._on_write()

  def _no_augmented_assignment(self, other: Any) -> NoReturn:
    del other
    raise TypeError(
        f"{type(self).__name__} does not support augmented assignment"
        " (`x += v`); use `x[...] = x + v`."
    )

  __iadd__ = __isub__ = __imul__ = __itruediv__ = __ipow__ = (
      _no_augmented_assignment
  )

  def __getattr__(self, name: str) -> Any:
    return getattr(self._tensor, name)

  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    """Allows torch functions to be called on TorchTensor."""
    if kwargs is None:
      kwargs = {}

    if not any(issubclass(t, cls) for t in types):
      return NotImplementedError(
          f"TorchTensor.__torch_function__ called with types {types}"
      )

    def _get_tensor(x: Any) -> Any:
      if isinstance(x, cls):
        return x._tensor  # pylint: disable=protected-access
      elif isinstance(x, (list, tuple)):
        return type(x)(_get_tensor(y) for y in x)
      else:
        return x

    args = tuple(_get_tensor(arg) for arg in args)
    kwargs = {k: _get_tensor(v) for k, v in kwargs.items()}
    return func(*args, **kwargs)

  if TYPE_CHECKING:
    # fmt: off
    def __add__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __sub__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __mul__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __div__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __pow__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __radd__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rsub__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rmul__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rdiv__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rpow__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __eq__(self, other: Any) -> "TorchTensor": raise NotImplementedError  # pyrefly: ignore[bad-override]
    def __ne__(self, other: Any) -> "TorchTensor": raise NotImplementedError  # pyrefly: ignore[bad-override]
    def __lt__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __le__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __gt__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __ge__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __pos__(self) -> "TorchTensor": raise NotImplementedError
    def __neg__(self) -> "TorchTensor": raise NotImplementedError
    def __abs__(self) -> "TorchTensor": raise NotImplementedError
    def __truediv__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rtruediv__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    # fmt: on


@_add_torch_operators
class ScalarTensor(TorchTensor):
  """Wraps a scalar field and propagates reads/writes back to the parent object."""

  def __init__(
      self,
      parent: Any,
      field_name: str,
      add_batch_dim: bool = False,
  ):
    value = getattr(parent, field_name)
    if not isinstance(value, (int, float)):
      raise TypeError(f"Expected int or float, got {type(value)}")
    dtype = np.float64 if isinstance(value, float) else np.int32
    array = np.array(value, dtype=dtype)
    super().__init__(array, add_batch_dim=add_batch_dim)
    object.__setattr__(self, "_parent", parent)
    object.__setattr__(self, "_field_name", field_name)

  @property
  def _tensor(self) -> torch.Tensor:
    self._torch_tensor[...] = getattr(self._parent, self._field_name)
    return self._torch_tensor

  def __setitem__(self, idx: Any, value: Any):
    super().__setitem__(idx, value)
    setattr(
        self._parent, self._field_name, self._torch_tensor.flatten()[0].item()
    )

  def __int__(self) -> int:
    return int(self.item())

  def __index__(self) -> int:
    return int(self.item())

  def __float__(self) -> float:
    return float(self.item())

  if TYPE_CHECKING:
    # fmt: off
    def __add__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __sub__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __mul__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __div__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __pow__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __radd__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rsub__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rmul__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rdiv__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rpow__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __eq__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __ne__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __lt__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __le__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __gt__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __ge__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __pos__(self) -> "TorchTensor": raise NotImplementedError
    def __neg__(self) -> "TorchTensor": raise NotImplementedError
    def __abs__(self) -> "TorchTensor": raise NotImplementedError
    def __truediv__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    def __rtruediv__(self, other: Any) -> "TorchTensor": raise NotImplementedError
    # fmt: on


class MjObjType(enum.Enum):
  """Enum for different types of Mujoco objects."""

  MJW_MODEL = "MJW_MODEL"
  MJW_DATA = "MJW_DATA"
  MJ_DATA = "MJ_DATA"
  MJ_MODEL = "MJ_MODEL"
  MJ_OPTION = "MJ_OPTION"
  MJW_OPTION = "MJW_OPTION"
  OTHER = "OTHER"

  @classmethod
  def from_obj(cls, obj: Any) -> "MjObjType":
    obj_type_map = {
        mjw.Model: MjObjType.MJW_MODEL,
        mjw.Data: MjObjType.MJW_DATA,
        mjw.Option: MjObjType.MJW_OPTION,
        mujoco.MjData: MjObjType.MJ_DATA,
        mujoco.MjModel: MjObjType.MJ_MODEL,
        mujoco.MjOption: MjObjType.MJ_OPTION,
    }
    for cls_, obj_type in obj_type_map.items():
      if isinstance(obj, cls_):
        return obj_type
    return cls.OTHER


class JointType(enum.IntEnum):
  FREE = mujoco.mjtJoint.mjJNT_FREE  # pyrefly: ignore[bad-assignment]
  BALL = mujoco.mjtJoint.mjJNT_BALL  # pyrefly: ignore[bad-assignment]
  SLIDE = mujoco.mjtJoint.mjJNT_SLIDE  # pyrefly: ignore[bad-assignment]
  HINGE = mujoco.mjtJoint.mjJNT_HINGE  # pyrefly: ignore[bad-assignment]

  def dof_width(self) -> int:
    return {0: 6, 1: 3, 2: 1, 3: 1}[self.value]

  def qpos_width(self) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[self.value]


class TorchWrapper(Generic[T], NoSetAttr):
  """Wraps objects that contain Warp/Numpy arrays and exposes them as tensors."""

  def __init__(self, obj: T, nworld: int | None = None) -> None:
    if isinstance(obj, mjw.Data):
      nworld = obj.nworld
    if isinstance(obj, mjw.Model) and nworld is None:
      raise ValueError("nworld must be provided when wrapping a Model object.")
    if isinstance(obj, mujoco.MjData) or isinstance(obj, mujoco.MjModel):
      if nworld is not None and nworld != 1:
        raise ValueError("nworld must be 1 when wrapping MjData or MjModel.")
      nworld = 1
      if isinstance(obj, mujoco.MjModel):
        if obj.nu != obj.nactuator or obj.nu != obj.nout:
          raise NotImplementedError(
              f"MIMO actuators (nu={obj.nu}, nactuator={obj.nactuator}, "
              f"nout={obj.nout}) are not supported by the Torch wrapper yet."
          )

    object.__setattr__(self, "_obj", obj)
    object.__setattr__(self, "_obj_type", MjObjType.from_obj(obj))
    object.__setattr__(self, "_tensor_cache", {})
    object.__setattr__(self, "_nworld", nworld or 1)
    object.__setattr__(self, "_bound_cache", {})
    object.__setattr__(self, "_defaults", {})
    object.__setattr__(self, "_pending_set_const", SetConstLevel.NONE)

    if isinstance(obj, mjw.Model):
      # Force trigger lazy tiling in __getattr__ for all setconst output fields.
      # This avoids race conditions when calling setconst methods in MuJoCo Warp
      # since mjw.Model fields are nworld==1 by default.
      for name in SET_CONST_OUTPUT_FIELDS:
        getattr(self, name, None)

  @property
  def obj(self) -> T:
    return self._obj

  @property
  def nworld(self) -> int:
    return self._nworld

  def __getattr__(self, name: str) -> Any:
    if name in self._tensor_cache:
      return self._tensor_cache[name]

    _check_mjfield_support(self._obj_type, name)
    val = getattr(self._obj, name)

    if (
        isinstance(val, wp.array)
        and _array_has_batched_attr(val)
        and val.shape[0] != self._nworld
        and isinstance(self._obj, (mjw.Model, mjw.Option))
    ):
      val = _tile_field(val, self._nworld)
      setattr(self._obj, name, val)
      graph_recapture_fn = getattr(self, "_graph_recapture_callback", None)
      if graph_recapture_fn is not None:
        graph_recapture_fn()

    if isinstance(val, wp.array):
      torch_val = TorchTensor(
          val,
          nworld=self._nworld,
          on_write=_create_set_const_callback(self, name),
      )
      self._tensor_cache[name] = torch_val
      return torch_val

    if isinstance(val, (int, float)):
      # handle scalar values that should be batched (e.g., time on MjData).
      add_batch_dim = _should_add_batch_dim(self._obj_type, name)
      scalar_val = ScalarTensor(self._obj, name, add_batch_dim=add_batch_dim)
      self._tensor_cache[name] = scalar_val
      return scalar_val

    if isinstance(val, np.ndarray):
      add_batch_dim = _should_add_batch_dim(self._obj_type, name)
      view_shape = _get_view_shape(
          self._obj_type, name, val, add_batch_dim, self._nworld
      )
      torch_val = TorchTensor(
          val,
          add_batch_dim=add_batch_dim,
          view_shape=view_shape,
          nworld=self._nworld,
          on_write=_create_set_const_callback(self, name),
      )
      self._tensor_cache[name] = torch_val
      return torch_val

    if _obj_has_array(val):
      wrapped_val = TorchWrapper(val, self._nworld)
      self._tensor_cache[name] = wrapped_val
      return wrapped_val

    self._tensor_cache[name] = val
    return val

  def __repr__(self) -> str:
    return f"TorchWrapper({repr(self._obj)})"

  @property
  def pending_set_const(self) -> SetConstLevel:
    """The `SetConstLevel` recorded since the last `flush_set_const()`."""
    return self._pending_set_const

  def flush_set_const(self) -> SetConstLevel:
    """Returns the pending `SetConstLevel` and resets it to `NONE`."""
    if self._obj_type not in (MjObjType.MJW_MODEL, MjObjType.MJ_MODEL):
      raise TypeError(
          "set_const tracking is only supported for Model objects, got"
          f" '{type(self._obj).__name__}'."
      )
    level = self._pending_set_const
    object.__setattr__(self, "_pending_set_const", SetConstLevel.NONE)
    return level

  def clear_defaults(self) -> None:
    """Clears saved defaults so current model values become the new defaults."""
    if self._obj_type not in (MjObjType.MJW_MODEL, MjObjType.MJ_MODEL):
      raise TypeError(
          "Default tracking is only supported for Model objects, got"
          f" '{type(self._obj).__name__}'."
      )
    self._defaults.clear()

  def restore_defaults(self, env_mask: torch.Tensor) -> None:
    """Restores modified model fields to their defaults in-place for `env_mask`."""
    if self._obj_type not in (MjObjType.MJW_MODEL, MjObjType.MJ_MODEL):
      raise TypeError(
          "Default tracking is only supported for Model objects, got"
          f" '{type(self._obj).__name__}'."
      )
    model_fields = _union_mj_fields()["Model"]
    for field, saved in self._defaults.items():
      tensor = getattr(self, field)
      target = tensor._tensor  # pylint: disable=protected-access
      stream = tensor._get_torch_stream()  # pylint: disable=protected-access
      # Avoid `torch.cuda.stream(None)`, which raises on MPS (macOS) devices.
      ctx = (
          torch.cuda.stream(stream)
          if stream is not None
          else contextlib.nullcontext()
      )
      with ctx:
        if model_fields[field].is_batch:
          mask = env_mask.view(-1, *([1] * (saved.ndim - 1)))
          torch.where(mask, saved, target, out=target)
        else:
          target.copy_(saved)
      level = _get_set_const_level(field, self._obj)
      if level != SetConstLevel.NONE:
        object.__setattr__(
            self, "_pending_set_const", self._pending_set_const | level
        )

  @overload
  def bind(
      self,
      specs: Any,
      /,
  ) -> "BoundModelAccessor":
    ...

  @overload
  def bind(
      self,
      model: Any,
      specs: Any,
      /,
  ) -> "BoundDataAccessor":
    ...

  def bind(self, *args) -> Union["BoundDataAccessor", "BoundModelAccessor"]:
    """Binds a Data or Model object."""
    if isinstance(self._obj, (mjw.Data, mujoco.MjData)):
      # Usage: data.bind(model, specs)
      if len(args) != 2:
        raise TypeError(
            "bind() on a Data object takes 2 positional arguments (model,"
            f" specs), but {len(args)} were given."
        )
      model, specs = args
      if not isinstance(model, TorchWrapper) or not isinstance(
          model.obj, (mjw.Model, mujoco.MjModel)
      ):
        raise TypeError(
            "bind on a Data object requires a model wrapper as the first"
            f" argument, got {model}."
        )
      specs = (specs,) if not isinstance(specs, (list, tuple)) else tuple(specs)
      if not specs:
        raise ValueError("specs must not be empty.")

      # TODO: is this cache key sufficient for all possible specs?
      # TODO: add model signature to the key
      key = ("data", type(specs[0]), tuple(s.id for s in specs))
      if key in self._bound_cache:
        return self._bound_cache[key]

      self._bound_cache[key] = BoundDataAccessor(self, model, specs)
      return self._bound_cache[key]

    if isinstance(self._obj, (mjw.Model, mujoco.MjModel)):
      # Usage: model.bind(spec)
      if len(args) != 1:
        raise TypeError(
            "bind() on a Model object takes 1 positional argument (specs), but"
            f" {len(args)} were given."
        )
      (specs,) = args
      specs = (specs,) if not isinstance(specs, (list, tuple)) else tuple(specs)
      if not specs:
        raise ValueError("specs must not be empty.")

      # TODO: is this cache key sufficient for all possible specs?
      # TODO: add model signature to the key
      key = ("model", type(specs[0]), tuple(s.id for s in specs))
      if key in self._bound_cache:
        return self._bound_cache[key]

      self._bound_cache[key] = BoundModelAccessor(self, specs)
      return self._bound_cache[key]

    raise TypeError(f"bind() is not supported for wrapped object '{self._obj}'")

  def bind_pattern(
      self,
      spec: mujoco.MjSpec,
      obj_type: mujoco.mjtObj,
      pattern: str | tuple[str, ...],
  ) -> "BoundModelAccessor":
    """Binds model fields by name pattern.

    Args:
      spec: MjSpec used for name resolution.
      obj_type: mujoco.mjtObj (mjOBJ_GEOM, mjOBJ_BODY, etc).
      pattern: Name pattern (exact, glob, or regex). Tuple for multiple.

    Returns:
      BoundModelAccessor for the matched entities.

    Raises:
      ValueError: If no names match or obj_type unsupported.
      TypeError: If called on non-Model object.
    """
    if not isinstance(self._obj, (mjw.Model, mujoco.MjModel)):
      raise TypeError("bind_pattern() only supported for Model objects")

    key = ("model_pattern", obj_type, pattern)
    if key in self._bound_cache:
      return self._bound_cache[key]

    names = mujoco_types.resolve_names(spec, obj_type, pattern)
    specs = [
        mujoco_types.get_spec_by_obj_type(spec, obj_type, n) for n in names
    ]
    self._bound_cache[key] = BoundModelAccessor(self, specs)
    return self._bound_cache[key]

  def register_graph_capture_callback(
      self, callback: Callable[[], None]
  ) -> None:
    """Registers a CUDA graph capture callback if TorchWrapper invalidates memory."""
    # If TorchWrapper modifies an underlying Warp array, a downstream graph
    # capture will become invalid. This callback mechanism triggers a downstream
    # graph capture after a graph invalidating TorchWrapper modification.
    existing = getattr(self, "_graph_recapture_callback", None)
    if existing is None:
      object.__setattr__(self, "_graph_recapture_callback", callback)
      return

    new_callback = lambda: (existing(), callback())
    object.__setattr__(self, "_graph_recapture_callback", new_callback)


class BoundDataAccessor:
  """Binds a Data object."""

  def __init__(
      self,
      data_wrapper: "TorchWrapper",
      model_wrapper: "TorchWrapper",
      specs: Union[Sequence[mujoco.MjStruct], mujoco.MjStruct],
  ):
    self.data = data_wrapper
    self.model = model_wrapper
    self.prefix = ""

    ids = []
    for spec in specs:
      # TODO: check model signature against spec signature
      if spec.id < 0:
        raise ValueError(f"Invalid spec ID: {spec.id}.")

      if spec.id < 0:
        raise KeyError(f"invalid id: {spec.id}")
      elif isinstance(spec, mujoco.MjsBody):
        pass
      elif isinstance(spec, mujoco.MjsJoint):
        self.prefix = "jnt_"
      elif isinstance(spec, mujoco.MjsGeom):
        self.prefix = "geom_"
      elif isinstance(spec, mujoco.MjsSite):
        self.prefix = "site_"
      elif isinstance(spec, mujoco.MjsLight):
        self.prefix = "light_"
      elif isinstance(spec, mujoco.MjsCamera):
        self.prefix = "cam_"
      elif isinstance(spec, mujoco.MjsTendon):
        self.prefix = "ten_"
      elif isinstance(spec, mujoco.MjsActuator):
        self.prefix = "actuator_"
      elif isinstance(spec, mujoco.MjsSensor):
        self.prefix = "sensor_"
      elif isinstance(spec, mujoco.MjsEquality):
        self.prefix = "eq_"
      else:
        raise ValueError("invalid spec type")
      ids.append(spec.id)

    if len(ids) == 1:
      ids = ids[0]

    self.device = data_wrapper.qpos.device
    self.id = torch.tensor(ids, dtype=torch.int, device=self.device)
    self._index_cache = {}

  def __getname(self, name: str):  # pylint: disable=unused-private-member
    """Get the name of the attribute and check if the type is correct."""
    if name == "sensordata":
      if self.prefix == "sensor_":
        return name
      raise AttributeError("sensordata is not available for this type")
    if name == "ctrl":
      if self.prefix == "actuator_":
        return name
      raise AttributeError("ctrl is not available for this type")
    if (
        name == "qpos"
        or name == "qvel"
        or name == "qacc"
        or name.startswith("qfrc_")
    ):
      if self.prefix == "jnt_":
        return name
      raise AttributeError(
          "qpos, qvel, qacc, qfrc are not available for this type"
      )
    return self.prefix + name

  def _get_indices(self, name: str) -> torch.Tensor:
    """Get the indices for the bound object(s) across all worlds."""
    if name in ("sensordata", "qpos", "qvel", "qacc") or (
        name.startswith("qfrc_")
    ):
      adr: Any = 0
      num: Any = 0
      if name == "sensordata":
        adr = self.model.sensor_adr[self.id]
        num = self.model.sensor_dim[self.id]
      elif name == "qpos":
        adr = self.model.jnt_qposadr[self.id]
        typ = self.model.jnt_type[self.id]
        num = sum((typ == jt) * jt.qpos_width() for jt in JointType)
      elif name == "qvel" or name == "qacc" or name.startswith("qfrc_"):
        adr = self.model.jnt_dofadr[self.id]
        typ = self.model.jnt_type[self.id]
        num = sum((typ == jt) * jt.dof_width() for jt in JointType)

      if self.id.shape:
        idx = []
        for a, n in zip(adr, num):
          idx.extend(a + j for j in range(n))
        return torch.tensor(idx, dtype=torch.int, device=self.device)

      if num > 1:
        return torch.arange(adr, adr + num, dtype=torch.int, device=self.device)
      return adr

    if name in ("mocap_pos", "mocap_quat"):
      return self.model.body_mocapid[self.id]

    return self.id

  def _get_indices_cached(self, name: str) -> torch.Tensor:
    if name in self._index_cache:
      return self._index_cache[name]
    idx = self._index_cache[name] = self._get_indices(name)
    return idx

  def __getattr__(self, name: str) -> torch.Tensor:
    """Gets data for the bound object(s) across all worlds."""
    target_tensor = getattr(self.data, self.__getname(name))
    indices = self._get_indices_cached(name)
    # NB: the returned tensor will not mutate the underlying data.
    return target_tensor[:, indices]

  def set(
      self,
      name: str,
      value: torch.Tensor,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Sets data for the bound object(s) across all worlds."""
    target_tensor = getattr(self.data, self.__getname(name))
    indices = self._get_indices_cached(name)
    if env_mask is None:
      target_tensor[:, indices] = value
    else:
      curr = target_tensor[:, indices]
      mask = env_mask.view(-1, *([1] * (curr.ndim - 1)))
      target_tensor[:, indices] = torch.where(mask, value, curr)


class BoundModelAccessor:
  """Binds a Model object."""

  def __init__(
      self,
      model_wrapper: "TorchWrapper",
      specs: Union[Sequence[mujoco.MjStruct], mujoco.MjStruct],
  ):
    self.model = model_wrapper
    self.prefix = ""

    ids = []
    for spec in specs:
      # TODO: check model signature against spec signature
      if spec.id < 0:
        raise ValueError(f"Invalid spec ID: {spec.id}.")

      if spec.id < 0:
        raise KeyError(f"invalid id: {spec.id}")
      elif isinstance(spec, mujoco.MjsBody):
        self.prefix = "body_"
      elif isinstance(spec, mujoco.MjsJoint):
        self.prefix = "jnt_"
      elif isinstance(spec, mujoco.MjsGeom):
        self.prefix = "geom_"
      elif isinstance(spec, mujoco.MjsSite):
        self.prefix = "site_"
      elif isinstance(spec, mujoco.MjsLight):
        self.prefix = "light_"
      elif isinstance(spec, mujoco.MjsCamera):
        self.prefix = "cam_"
      elif isinstance(spec, mujoco.MjsMesh):
        self.prefix = "mesh_"
      elif isinstance(spec, mujoco.MjsHField):
        self.prefix = "hfield_"
      elif isinstance(spec, mujoco.MjsPair):
        self.prefix = "pair_"
      elif isinstance(spec, mujoco.MjsTendon):
        self.prefix = "tendon_"
      elif isinstance(spec, mujoco.MjsActuator):
        self.prefix = "actuator_"
      elif isinstance(spec, mujoco.MjsSensor):
        self.prefix = "sensor_"
      elif isinstance(spec, mujoco.MjsNumeric):
        self.prefix = "numeric_"
      elif isinstance(spec, mujoco.MjsText):
        self.prefix = "text_"
      elif isinstance(spec, mujoco.MjsTuple):
        self.prefix = "tuple_"
      elif isinstance(spec, mujoco.MjsKey):
        self.prefix = "key_"
      elif isinstance(spec, mujoco.MjsEquality):
        self.prefix = "eq_"
      elif isinstance(spec, mujoco.MjsExclude):
        self.prefix = "exclude_"
      elif isinstance(spec, mujoco.MjsSkin):
        self.prefix = "skin_"
      elif isinstance(spec, mujoco.MjsMaterial):
        self.prefix = "material_"
      else:
        raise ValueError("invalid spec type")
      ids.append(spec.id)

    if len(ids) == 1:
      ids = ids[0]

    self.device = model_wrapper.geom_pos.device
    self.id = torch.tensor(ids, dtype=torch.int, device=self.device)

    # Pre-compute and cache the mapped DOF indices for joint bound accessors.
    if self.prefix == "jnt_":
      ids_1d = self.id.unsqueeze(0) if self.id.ndim == 0 else self.id
      mask_dof = ids_1d.unsqueeze(1) == model_wrapper.dof_jntid.unsqueeze(0)
      self.dof_id = mask_dof.nonzero(as_tuple=True)[1].to(self.id.dtype)

  def __getname(self, name: str) -> str:  # pylint: disable=unused-private-member
    candidate = self.prefix + name
    fields = _union_mj_fields()["Model"]
    if candidate in fields:
      # return prefix + name if it is a valid Model field
      return candidate
    if name in fields:
      # return name if it is a valid Model field (e.g. dof_* on a jnt_ prefix)
      return name
    raise AttributeError(f"'{name}' is not a recognized model field.")

  def _get_indices(self, name: str) -> torch.Tensor:
    """Returns cached indices for the bound accessor."""
    if name.startswith("dof_") and self.prefix == "jnt_":
      return self.dof_id
    return self.id

  def __getattr__(self, name: str) -> torch.Tensor:
    """Gets model field for the bound object(s) across all worlds."""
    name = self.__getname(name)
    target_tensor = getattr(self.model, name)
    indices = self._get_indices(name)
    # NB: single-entity binds return a view that aliases the model tensor.
    if _union_mj_fields()["Model"][name].is_batch:
      return target_tensor[:, indices]
    return target_tensor[indices]

  def set(
      self,
      name: str,
      value: torch.Tensor,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    """Sets model field for the bound object(s) across all worlds."""
    name = self.__getname(name)
    target_tensor = getattr(self.model, name)
    indices = self._get_indices(name)
    if not _union_mj_fields()["Model"][name].is_batch:
      target_tensor[indices] = value
      return

    if env_mask is None:
      target_tensor[:, indices] = value
    else:
      curr = target_tensor[:, indices]
      mask = env_mask.view(-1, *([1] * (curr.ndim - 1)))
      target_tensor[:, indices] = torch.where(mask, value, curr)


def _tile_field(wp_array: wp.array, nworld: int) -> wp.array:
  """Tiles fields that are not batched to nworld.

  Args:
    wp_array: warp array to tile.
    nworld: the number of worlds to tile the field to.

  Returns:
    The tiled warp array.
  """
  src = _wp_to_torch(wp_array)
  dst = src.repeat((nworld,) + (1,) * (src.ndim - 1)).contiguous()
  dst_wp_array: Any = wp.from_torch(dst, dtype=wp_array.dtype)
  dst_wp_array._is_batched = True  # pylint: disable=protected-access
  return dst_wp_array


def _obj_has_array(obj: Any) -> bool:
  """Checks recursively if an object contains arrays."""
  if isinstance(obj, wp.array) or isinstance(obj, np.ndarray):
    return True

  if isinstance(obj, (mujoco.MjOption, mjw.Option)):
    return True

  for attr in vars(obj):
    if _obj_has_array(getattr(obj, attr)):
      return True

  return False


def _array_has_batched_attr(array: wp.array | np.ndarray) -> bool:
  """Checks if a Warp or numpy array has an is_batched=True attribute."""
  if isinstance(array, np.ndarray):
    return False
  return hasattr(array, "_is_batched") and array._is_batched  # pylint: disable=protected-access


@dataclasses.dataclass(frozen=True)
class MjField:

  name: str
  is_batch: bool  # whether the field is batched in MuJoCo Warp
  shape: tuple[str, ...]  # the shape of the field in MuJoCo Warp


@functools.lru_cache(maxsize=1)
def _union_mj_fields() -> dict[str, dict[str, MjField]]:
  """Returns the union of fields in MuJoCo and MuJoCo Warp."""
  mj_model_fields = mj_structs.STRUCTS["mjModel"].fields
  mj_data_fields = mj_structs.STRUCTS["mjData"].fields
  mj_opt_fields = mj_structs.STRUCTS["mjOption"].fields
  mjw_model_annotations = mjw.Model.__annotations__
  mjw_data_annotations = mjw.Data.__annotations__
  mjw_opt_annotations = mjw.Option.__annotations__

  fields = {}
  for obj_name, mj_fields, mjw_annotations in (
      ("Model", mj_model_fields, mjw_model_annotations),
      ("Data", mj_data_fields, mjw_data_annotations),
      ("Option", mj_opt_fields, mjw_opt_annotations),
  ):
    fields[obj_name] = {}
    for mj_field in mj_fields:
      if mj_field.name in ("contact", "efc", "stat"):  # pyrefly: ignore[missing-attribute]
        continue
      name = mj_field.name
      if name not in mjw_annotations:
        continue

      # get the MJWarp shape.
      shape = (None,)
      if hasattr(mjw_annotations[name], "shape"):
        shape = mjw_annotations[name].shape
        dtype = getattr(mjw_annotations[name], "dtype", None)
        if dtype is not None and hasattr(dtype, "_shape_"):
          shape = shape + dtype._shape_  # pylint: disable=protected-access

      is_batch = shape[0] in ("*", "nworld")

      if not _shape_annotations_compatible(mj_field, shape, is_batch):  # pyrefly: ignore[bad-argument-type]
        continue

      fields[obj_name][name] = MjField(
          name=name, is_batch=is_batch, shape=shape  # pyrefly: ignore[bad-argument-type]
      )

  return fields


def _shape_annotations_compatible(
    mj_field: mj_structs.StructFieldDecl,
    mjw_shape: list[str | int],
    is_batch: bool,
) -> bool:
  """Checks that MJ and MJW shape annotations are compatible."""
  mjw_shape = mjw_shape[is_batch:]  # ignore mjw batch dim
  has_arr_extent = hasattr(mj_field, "array_extent")
  if has_arr_extent and hasattr(mj_field.type, "extents"):
    return mj_field.type.extents == mjw_shape

  if has_arr_extent and mj_field.array_extent is None:
    return True

  mj_shape = [
      getattr(mujoco, a, a) if isinstance(a, str) else a
      for a in mj_field.array_extent  # pyrefly: ignore[not-iterable]
  ]
  mjw_shape_int = [i for i in mjw_shape if isinstance(i, int)]
  mj_shape_int = [i for i in mj_shape if isinstance(i, int)]
  if np.prod(mjw_shape_int) != np.prod(mj_shape_int):
    return False

  # Normalize actuator dimension names: the MuJoCo MIMO refactoring split "nu"
  # into "nactuator" (objects), "nu" (controls), and "nout" (force outputs).
  # Allow "nactuator" and "nout" to match "nu" for backward/forward compatibility.
  act_dim_aliases = {"nactuator": "nu", "nout": "nu"}
  mjw_shape_str = [
      act_dim_aliases.get(i, i) for i in mjw_shape if not isinstance(i, int)
  ]
  mj_shape_str = [
      act_dim_aliases.get(i, i) for i in mj_shape if not isinstance(i, int)
  ]
  if mjw_shape_str != mj_shape_str:
    return False

  return True


def _check_mjfield_support(obj_type: MjObjType, name: str) -> None:
  """Checks if the given name is a valid field accessor for the given object type."""
  if obj_type in (MjObjType.MJ_DATA, MjObjType.MJW_DATA):
    if name not in _union_mj_fields()["Data"]:
      raise AttributeError(
          f"{name} is not a supported field accessor for mjData/mjWarpData."
      )
    return
  if obj_type in (MjObjType.MJ_MODEL, MjObjType.MJW_MODEL):
    if name not in _union_mj_fields()["Model"]:
      raise AttributeError(
          f"{name} is not a supported field accessor for mjModel/mjWarpModel."
      )
    return
  if obj_type in (MjObjType.MJ_OPTION, MjObjType.MJW_OPTION):
    if name not in _union_mj_fields()["Option"]:
      raise AttributeError(
          f"{name} is not a supported field accessor for mjOption/mjWarpOption."
      )
    return


def _should_add_batch_dim(obj_type: MjObjType, name: str) -> bool:
  """Returns True if a batch dimension should be added for this field."""
  if obj_type not in (
      MjObjType.MJ_DATA,
      MjObjType.MJ_MODEL,
      MjObjType.MJ_OPTION,
  ):
    return False  # MuJoCo Warp fields already have batch dimensions added.
  obj_type_to_fields_key = {
      MjObjType.MJ_DATA: "Data",
      MjObjType.MJ_MODEL: "Model",
      MjObjType.MJ_OPTION: "Option",
  }
  fields_key = obj_type_to_fields_key[obj_type]
  field = _union_mj_fields()[fields_key][name]
  return field.is_batch


def _get_view_shape(
    obj_type: MjObjType,
    name: str,
    np_val: np.ndarray,
    add_batch_dim: bool,
    nworld: int,
) -> tuple[int, ...] | None:
  """Computes view shape if numpy array ndim doesn't match expected ndim."""
  if obj_type not in (
      MjObjType.MJ_DATA,
      MjObjType.MJ_MODEL,
      MjObjType.MJ_OPTION,
  ):
    return None  # We use MuJoCo Warp shapes by default.
  fields_key = {
      MjObjType.MJ_MODEL: "Model",
      MjObjType.MJ_DATA: "Data",
      MjObjType.MJ_OPTION: "Option",
  }[obj_type]

  field: MjField = _union_mj_fields()[fields_key][name]
  expected_mjw_ndim = len(field.shape)
  actual_ndim = len(np_val.shape) + (1 if add_batch_dim else 0)

  if expected_mjw_ndim == actual_ndim:
    return None

  view_shape, idx = [], 0
  for mjw_dim in field.shape:
    if mjw_dim in ("*", "nworld"):
      view_shape.append(nworld)
    elif isinstance(mjw_dim, int):
      if idx < len(np_val.shape) and np_val.shape[idx] == mjw_dim:
        idx += 1  # only advance numpy index if it matches mjw dim.
      view_shape.append(mjw_dim)
    else:
      view_shape.append(np_val.shape[idx])
      idx += 1

  if np.prod(view_shape) != np.prod(np_val.shape):
    # Could not find an MJW compatible view shape for the field. We may
    # want to raise an AssertionError at some point.
    return None

  return tuple(view_shape)
