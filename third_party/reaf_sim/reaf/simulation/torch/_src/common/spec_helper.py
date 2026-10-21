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
"""Helper to manage and prefix specification names for environment measurements."""

from collections.abc import Mapping
import dataclasses
from typing import Generic
from typing import TypeVar

from dm_env import specs
import torch

# TODO(reaf): we fork because of nworld -- is there a better way?

SpecTemplate = TypeVar("SpecTemplate", bound="dataclasses.DataclassInstance")


# TODO(taylorhowell): remove this
class SpecHelper(Generic[SpecTemplate]):
  """Helper to manage and prefix specification names for environment measurements/commands."""

  # TODO(silvia): use a protocol?

  def __init__(
      self,
      template_type: type[SpecTemplate],
      spec_definition: Mapping[str, specs.Array],
      prefix: str,
      separator: str,
      reshape_to_spec: bool = False,
      cast_to_spec: bool = False,
      nworld: int = 1,
  ):
    """Initializes the SpecHelper."""
    self._template_type = template_type
    template_fields = dataclasses.fields(self._template_type)
    self._template_field_names = set([field.name for field in template_fields])

    # Always prepend nworld dimension to all specs
    spec_definition_with_nworld = {}
    for key, spec_value in spec_definition.items():
      new_shape = (nworld,) + spec_value.shape
      spec_definition_with_nworld[key] = specs.Array(
          shape=new_shape, dtype=spec_value.dtype
      )
    self._spec_definition = spec_definition_with_nworld

    spec_definition_field_names = set(spec_definition.keys())
    if self._template_field_names != spec_definition_field_names:
      raise ValueError(
          "The fields of the template and the spec definition must match. "
          f"Template fields: {self._template_field_names}, "
          f"Spec definition fields: {spec_definition_field_names}"
      )

    field_name_to_prefixed_name = {}
    self._spec_dict = {}
    for field_name in self._template_field_names:
      spec = self._spec_definition[field_name]
      if not isinstance(spec, specs.Array):
        raise TypeError(
            f"Field {field_name} is not a specs.Array. Got {type(spec)}."
        )
      if spec.name is not None:
        base_name = spec.name
      else:
        base_name = field_name
      prefixed_name = prefix + separator + base_name
      field_name_to_prefixed_name[field_name] = prefixed_name
      self._spec_dict[prefixed_name] = spec

    self._names = self._template_type(**field_name_to_prefixed_name)

    self._reshape_to_spec = reshape_to_spec
    self._cast_to_spec = cast_to_spec

  @property
  def spec_dict(self) -> Mapping[str, specs.Array]:
    """Returns a mapping of full prefixed measurement names to array specs."""
    return self._spec_dict

  @property
  def names_dataclass(self) -> SpecTemplate:
    """Returns the SpecTemplate instance with fields containing the full prefixed measurement names."""
    return self._names

  def make_dict_of_arrays(
      self,
      **kwargs: torch.Tensor,
  ) -> Mapping[str, torch.Tensor]:
    """Returns a mapping of full prefixed measurement names to array values."""
    arg_keys = set(kwargs.keys())
    if arg_keys != self._template_field_names:
      raise ValueError(
          "Arguments to make_dict_of_arrays must match the spec. "
          f"Kwargs keys: {arg_keys}, "
          f"Spec keys: {self._template_field_names}"
      )

    result = {}
    for arg_key in arg_keys:
      prefixed_name = getattr(self._names, arg_key)
      spec = self._spec_dict[prefixed_name]
      array = kwargs[arg_key]
      if self._reshape_to_spec:
        array = torch.reshape(array, spec.shape)
      if self._cast_to_spec:
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        torch_dtype = dtype_map.get(str(spec.dtype), torch.float32)
        array = array.to(dtype=torch_dtype)
      result[prefixed_name] = array
    return result
