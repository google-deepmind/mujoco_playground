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

"""reaf-sim vectorized simulation framework."""

from reaf.simulation.torch._src.builders import flat_scene
from reaf.simulation.torch._src.common import action_history
from reaf.simulation.torch._src.common import actuator_device
from reaf.simulation.torch._src.common import articulated_device
from reaf.simulation.torch._src.common import batched_camera_device
from reaf.simulation.torch._src.common import camera_device
from reaf.simulation.torch._src.common import clock_device
from reaf.simulation.torch._src.common import contact_sensor_device
from reaf.simulation.torch._src.common import default_discount_provider
from reaf.simulation.torch._src.common import delay_buffer
from reaf.simulation.torch._src.common import free_device
from reaf.simulation.torch._src.common import functional_device
from reaf.simulation.torch._src.common import functional_metric
from reaf.simulation.torch._src.common import functional_observation
from reaf.simulation.torch._src.common import functional_reset
from reaf.simulation.torch._src.common import functional_reward
from reaf.simulation.torch._src.common import functional_termination
from reaf.simulation.torch._src.common import inside_site_device
from reaf.simulation.torch._src.common import math
from reaf.simulation.torch._src.common import model_randomization
from reaf.simulation.torch._src.common import mujoco_camera_device
from reaf.simulation.torch._src.common import multi_asset_device
from reaf.simulation.torch._src.common import noise
from reaf.simulation.torch._src.common import processors
from reaf.simulation.torch._src.common import resetter
from reaf.simulation.torch._src.common import site_device
from reaf.simulation.torch._src.common import spec_helper
from reaf.simulation.torch._src.common import tactile_device
from reaf.simulation.torch._src.common import tensors
from reaf.simulation.torch._src.common import timeout_termination_checker
from reaf.simulation.torch._src.common import zero_reward_provider
from reaf.simulation.torch._src.common.action_history import ActionHistory
from reaf.simulation.torch._src.common.actuator_device import ActuatorDevice
from reaf.simulation.torch._src.common.articulated_device import ArticulatedDevice
from reaf.simulation.torch._src.common.batched_camera_device import BatchedMujocoCameraDevice
from reaf.simulation.torch._src.common.camera_device import CameraDevice
from reaf.simulation.torch._src.common.clock_device import ClockDevice
from reaf.simulation.torch._src.common.contact_sensor_device import ContactField
from reaf.simulation.torch._src.common.contact_sensor_device import ContactSensorDevice
from reaf.simulation.torch._src.common.contact_sensor_device import ObjType
from reaf.simulation.torch._src.common.contact_sensor_device import ReduceMode
from reaf.simulation.torch._src.common.default_discount_provider import DefaultDiscountProvider
from reaf.simulation.torch._src.common.delay_buffer import DelayBuffer
from reaf.simulation.torch._src.common.free_device import FreeDevice
from reaf.simulation.torch._src.common.free_device import MocapDevice
from reaf.simulation.torch._src.common.functional_device import joint
from reaf.simulation.torch._src.common.functional_device import joint_pos
from reaf.simulation.torch._src.common.functional_device import joint_vel
from reaf.simulation.torch._src.common.functional_device import MeasurementDevice
from reaf.simulation.torch._src.common.functional_metric import FunctionalMetricProvider
from reaf.simulation.torch._src.common.functional_metric import MetricReduceMode
from reaf.simulation.torch._src.common.functional_observation import FunctionalObservationAdapter
from reaf.simulation.torch._src.common.functional_reset import FunctionalEnvironmentReset
from reaf.simulation.torch._src.common.functional_reset import TYPE_KEY
from reaf.simulation.torch._src.common.functional_reset import TYPE_RESET
from reaf.simulation.torch._src.common.functional_reset import TYPE_STARTUP
from reaf.simulation.torch._src.common.functional_reward import FunctionalRewardProvider
from reaf.simulation.torch._src.common.functional_reward import PassthroughRewardProvider
from reaf.simulation.torch._src.common.functional_termination import FunctionalTerminationChecker
from reaf.simulation.torch._src.common.functional_termination import TerminationResult
from reaf.simulation.torch._src.common.inside_site_device import InsideSiteDevice
from reaf.simulation.torch._src.common.model_randomization import randomize_actuator_gains
from reaf.simulation.torch._src.common.model_randomization import randomize_field
from reaf.simulation.torch._src.common.model_randomization import randomize_quat
from reaf.simulation.torch._src.common.mujoco_camera_device import MujocoCameraDevice
from reaf.simulation.torch._src.common.mujoco_camera_device import RenderMode
from reaf.simulation.torch._src.common.multi_asset_device import MultiAsset
from reaf.simulation.torch._src.common.multi_asset_device import MultiAssetDevice
from reaf.simulation.torch._src.common.processors import DeltaActions
from reaf.simulation.torch._src.common.processors import StochasticSubstepCommands
from reaf.simulation.torch._src.common.processors import StochasticSubstepMeasurements
from reaf.simulation.torch._src.common.processors import TransformActions
from reaf.simulation.torch._src.common.resetter import CompositeResetter
from reaf.simulation.torch._src.common.resetter import DomainRandomizationResetter
from reaf.simulation.torch._src.common.resetter import NanTerminationResetter
from reaf.simulation.torch._src.common.resetter import TerminationResetter
from reaf.simulation.torch._src.common.site_device import SiteDevice
from reaf.simulation.torch._src.common.spec_helper import SpecHelper
from reaf.simulation.torch._src.common.spec_helper import SpecTemplate
from reaf.simulation.torch._src.common.tactile_device import TactileDevice
from reaf.simulation.torch._src.common.timeout_termination_checker import TimeoutTerminationChecker
from reaf.simulation.torch._src.common.zero_reward_provider import ZeroRewardProvider
from reaf.simulation.torch._src.core import clock
from reaf.simulation.torch._src.core import device
from reaf.simulation.torch._src.core import device_layer
from reaf.simulation.torch._src.core import discount_provider
from reaf.simulation.torch._src.core import environment
from reaf.simulation.torch._src.core import feature_producer
from reaf.simulation.torch._src.core import metric_provider
from reaf.simulation.torch._src.core import mujoco_types
from reaf.simulation.torch._src.core import observation_space_adapter
from reaf.simulation.torch._src.core import profile
from reaf.simulation.torch._src.core import protocols
from reaf.simulation.torch._src.core import reward_provider
from reaf.simulation.torch._src.core import simulator
from reaf.simulation.torch._src.core import substep_commands_processor
from reaf.simulation.torch._src.core import substep_measurements_processor
from reaf.simulation.torch._src.core import task_layer
from reaf.simulation.torch._src.core import termination_checker
from reaf.simulation.torch._src.core import torch_wrapper
from reaf.simulation.torch._src.core import types
from reaf.simulation.torch._src.core.clock import Clock
from reaf.simulation.torch._src.core.device import Device
from reaf.simulation.torch._src.core.device_layer import DeviceLayer
from reaf.simulation.torch._src.core.discount_provider import DiscountProvider
from reaf.simulation.torch._src.core.environment import AutoResetMode
from reaf.simulation.torch._src.core.environment import EndOfEpisodeHandler
from reaf.simulation.torch._src.core.environment import Environment
from reaf.simulation.torch._src.core.environment import EnvironmentReset
from reaf.simulation.torch._src.core.feature_producer import FeatureProducer
from reaf.simulation.torch._src.core.metric_provider import MetricProvider
from reaf.simulation.torch._src.core.mujoco_types import get_spec_by_obj_type
from reaf.simulation.torch._src.core.mujoco_types import match_names
from reaf.simulation.torch._src.core.mujoco_types import OBJ_TYPE_TO_ACCESSOR
from reaf.simulation.torch._src.core.mujoco_types import PREFIX_TO_OBJ_TYPE
from reaf.simulation.torch._src.core.mujoco_types import resolve_names
from reaf.simulation.torch._src.core.mujoco_types import resolve_values
from reaf.simulation.torch._src.core.mujoco_types import soft_joint_limits
from reaf.simulation.torch._src.core.mujoco_types import SPEC_TYPE_INFO
from reaf.simulation.torch._src.core.observation_space_adapter import ObservationSpaceAdapter
from reaf.simulation.torch._src.core.protocols import TorchDeviceLayer
from reaf.simulation.torch._src.core.reward_provider import RewardProvider
from reaf.simulation.torch._src.core.reward_provider import RewardSpec
from reaf.simulation.torch._src.core.reward_provider import RewardValue
from reaf.simulation.torch._src.core.simulator import Simulator
from reaf.simulation.torch._src.core.substep_commands_processor import SubstepCommandsProcessor
from reaf.simulation.torch._src.core.substep_measurements_processor import SubstepMeasurementsProcessor
from reaf.simulation.torch._src.core.task_layer import TaskLayer
from reaf.simulation.torch._src.core.termination_checker import TerminationChecker
from reaf.simulation.torch._src.core.torch_wrapper import JointType
from reaf.simulation.torch._src.core.torch_wrapper import MjField
from reaf.simulation.torch._src.core.torch_wrapper import MjObjType
from reaf.simulation.torch._src.core.torch_wrapper import SetConstLevel
from reaf.simulation.torch._src.core.torch_wrapper import TorchTensor
from reaf.simulation.torch._src.core.torch_wrapper import TorchWrapper
from reaf.simulation.torch._src.core.types import ArrayType
from reaf.simulation.torch._src.core.types import Data
from reaf.simulation.torch._src.core.types import Impl
from reaf.simulation.torch._src.core.types import Model
from reaf.simulation.torch._src.core.types import NUMPY_TO_TORCH_DTYPE
from reaf.simulation.torch._src.core.types import Term
from reaf.simulation.torch._src.core.types import TORCH_TO_NUMPY_DTYPE
