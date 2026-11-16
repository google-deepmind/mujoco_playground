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
"""Tests for TaskLayer.validate_spec."""

from collections.abc import Iterable
from collections.abc import Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import commands_processor as core_commands_processor
from reaf.simulation.torch._src.core import feature_producer as feature_producer_module
from reaf.simulation.torch._src.core import task_layer as task_layer_module
from reaf.simulation.torch._src.core import termination_checker as core_termination_checker
import torch


def _spec(keys: Iterable[str]) -> dict[str, gdmr_types.UnboundedArraySpec]:
  return {
      k: gdmr_types.UnboundedArraySpec(shape=(), dtype=np.float32) for k in keys
  }


def _processor(
    name: str, consumed: Iterable[str], produced: Iterable[str]
) -> core_commands_processor.CommandsProcessor:
  proc = mock.create_autospec(
      core_commands_processor.CommandsProcessor, instance=True
  )
  proc.name = name
  proc.consumed_commands_spec.return_value = _spec(consumed)
  proc.produced_commands_keys.return_value = set(produced)
  return proc


def _producer(
    name: str, required: Iterable[str], produced: Iterable[str]
) -> feature_producer_module.FeatureProducer:
  prod = mock.create_autospec(
      feature_producer_module.FeatureProducer, instance=True
  )
  prod.name = name
  prod.required_features_keys.return_value = set(required)
  prod.produced_features_spec.return_value = _spec(produced)
  return prod


def _task_layer(
    commands_processors: Sequence[
        core_commands_processor.CommandsProcessor
    ] = (),
    features_producers: Sequence[feature_producer_module.FeatureProducer] = (),
) -> task_layer_module.TaskLayer:
  return task_layer_module.TaskLayer(
      commands_processors=commands_processors,
      features_producers=features_producers,
      termination_checkers=(),
      xpu_device="cpu",
  )


_DEV_CMDS = _spec(["dev/c1", "dev/c2", "dev/c3"])
_DEV_MEAS = _spec(["dev/m1", "dev/m2"])


class TaskLayerValidateSpecTest(parameterized.TestCase):

  def test_valid_empty_and_none_specs(self):
    _task_layer().validate_spec()
    _task_layer().validate_spec(
        device_layer_commands_spec=_DEV_CMDS,
        device_layer_measurements_spec=_DEV_MEAS,
    )

  def test_valid_chained_processors_and_producers(self):
    layer = _task_layer(
        commands_processors=(
            _processor("p0", ["pol/c"], ["p1/c", "dev/c3"]),
            _processor("p1", ["p1/c"], ["dev/c1", "dev/c2"]),
        ),
        features_producers=(
            _producer("f1", ["dev/m1"], ["f1/out"]),
            _producer("f2", ["dev/m2", "f1/out"], ["f2/out"]),
        ),
    )
    layer.validate_spec(
        device_layer_commands_spec=_DEV_CMDS,
        device_layer_measurements_spec=_DEV_MEAS,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="unconsumed_command",
          processors=(_processor("p0", [], ["dev/missing"]),),
          producers=(),
          error=r"commands processor 'p0' produces keys \['dev/missing'\]",
      ),
      dict(
          testcase_name="processors_in_wrong_order",
          processors=(
              _processor("p_late", ["p_early/c"], ["dev/c1"]),
              _processor("p_early", ["pol/c"], ["p_early/c"]),
          ),
          producers=(),
          error=r"commands processor 'p_early' produces keys \['p_early/c'\]",
      ),
      dict(
          testcase_name="command_already_claimed_by_later_processor",
          processors=(
              _processor("p0", [], ["dev/c1"]),
              _processor("p1", ["p1/c"], ["dev/c1"]),
          ),
          producers=(),
          error=r"commands processor 'p0' produces keys \['dev/c1'\]",
      ),
      dict(
          testcase_name="missing_measurement",
          processors=(),
          producers=(_producer("f1", ["dev/missing"], ["f1/out"]),),
          error=r"features producer 'f1' requires keys \['dev/missing'\]",
      ),
      dict(
          testcase_name="producers_in_wrong_order",
          processors=(),
          producers=(
              _producer("f_late", ["f_early/out"], ["f_late/out"]),
              _producer("f_early", ["dev/m1"], ["f_early/out"]),
          ),
          error=r"features producer 'f_late' requires keys \['f_early/out'\]",
      ),
      dict(
          testcase_name="duplicate_producer_key",
          processors=(),
          producers=(
              _producer("f1", ["dev/m1"], ["dup"]),
              _producer("f2", ["dev/m1"], ["dup"]),
          ),
          error=r"features producer 'f2' produces keys \['dup'\]",
      ),
      dict(
          testcase_name="overwrites_device_measurement",
          processors=(),
          producers=(_producer("f1", [], ["dev/m1"]),),
          error=r"features producer 'f1' produces keys \['dev/m1'\]",
      ),
  )
  def test_invalid_spec_raises_value_error(
      self,
      processors: Sequence[core_commands_processor.CommandsProcessor],
      producers: Sequence[feature_producer_module.FeatureProducer],
      error: str,
  ):
    layer = _task_layer(processors, producers)
    with self.assertRaisesRegex(ValueError, error):
      layer.validate_spec(
          device_layer_commands_spec=_DEV_CMDS,
          device_layer_measurements_spec=_DEV_MEAS,
      )

  def test_check_for_termination_updates_in_place(self):
    checker = mock.create_autospec(
        core_termination_checker.TerminationChecker, instance=True
    )
    checker.name = "c0"
    expected = torch.tensor(
        [
            core_termination_checker.TerminationResult.TERMINATE,
            core_termination_checker.TerminationResult.DO_NOT_TERMINATE,
        ],
        dtype=torch.int32,
    )
    checker.check_termination.return_value = expected
    layer = task_layer_module.TaskLayer(
        commands_processors=(),
        features_producers=(),
        termination_checkers=(checker,),
        xpu_device="cpu",
        nworld=2,
    )
    buf_ptr = layer._termination_buffer.data_ptr()
    out = layer.check_for_termination({})
    self.assertEqual(out.data_ptr(), buf_ptr)
    self.assertEqual(layer._termination_buffer.data_ptr(), buf_ptr)
    torch.testing.assert_close(out, expected)


if __name__ == "__main__":
  absltest.main()
