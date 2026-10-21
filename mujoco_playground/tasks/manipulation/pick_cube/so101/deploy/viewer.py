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

"""An interactive view of a running deployment, in simulation or on hardware."""

from collections.abc import Iterable
from collections.abc import Sequence

from gdm_robotics.interfaces import episodic_logger as gdmr_logger
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
from mujoco_playground.robots.so101.real import constant
from mujoco_playground.robots.so101.sim import constant as sim_constant
from mujoco_playground.tasks.manipulation.pick_cube import task as pick_task
from mujoco_playground.tasks.manipulation.pick_cube.so101 import task as so101_task
from mujoco_playground.tasks.manipulation.pick_cube.so101.deploy import env as pick_cube_env
import numpy as np
from reaf.simulation import torch as reaf_sim
from reaf.simulation.torch import viewer as reaf_viewer
from typing_extensions import override

# The goal marker, as `pick_cube` attaches it. Its geom sits in a group MuJoCo
# hides by default; the viewer moves it into a visible group so the human can
# see what they are dragging.
_TARGET_BODY = f"{pick_task.TARGET}/{pick_task.TARGET}"
# Geom group 2 is enabled by default in MuJoCo (target defaults to group 4).
_TARGET_GEOM_GROUP = 2

_CAMERA_DISTANCE = 0.8
_CAMERA_AZIMUTH = 135.0
_CAMERA_ELEVATION = -25.0


def configure_viewer_spec(
    spec: mujoco.MjSpec,
    lookat: Sequence[float] | np.ndarray | None = None,
) -> None:
  """Shows the mocap target and frames the default free camera on it."""
  if lookat is None:
    elem = spec.body(_TARGET_BODY)
    lookat = np.zeros(3)
    while elem is not None:
      lookat = lookat + elem.pos
      elem = elem.frame
  spec.geom(_TARGET_BODY).group = _TARGET_GEOM_GROUP
  spec.stat.center = lookat
  spec.stat.extent = _CAMERA_DISTANCE / 1.5
  spec.visual.global_.azimuth = _CAMERA_AZIMUTH
  spec.visual.global_.elevation = _CAMERA_ELEVATION


class RealSimulator(reaf_sim.Simulator):
  """Poses the arm from joint measurements, and moves the target device."""

  def __init__(self, device_layer: reaf_sim.TorchDeviceLayer):
    cfg = so101_task.so101_pick_cube_pixels_env(impl="cpp")
    deploy_cfg = reaf_sim.flat_scene.make_deployment_config(
        cfg, observation_groups=pick_cube_env.OBSERVATION_GROUPS
    )
    spec = reaf_sim.flat_scene.build_scene(deploy_cfg)
    # Hides the simulation cube since it's not tracked in real mode.
    spec.delete(spec.body("cube/cube"))

    self._device_layer = device_layer
    self._target_device = device_layer.get_device(pick_task.TARGET)
    configure_viewer_spec(spec, lookat=self._target_device.target_pos)
    super().__init__(mj_spec=spec, impl="cpp", xpu_device="cpu")

    self._qpos_adr = np.array([
        self.mj_model.joint(f"{sim_constant.ATTACHMENT_PREFIX}{j}").qposadr[0]
        for j in constant.JOINTS
    ])
    self._mocap_id = int(
        self.mj_model.body_mocapid[self.mj_model.body(_TARGET_BODY).id]
    )
    self.mj_data.mocap_pos[self._mocap_id] = self._target_device.target_pos

  @property
  @override
  def data(self) -> reaf_sim.Data:
    joint_pos = self._device_layer.get_measurements().get(
        f"{constant.ROBOT_STATE}/{constant.JOINT_POS}"
    )
    if joint_pos is not None:
      self.mj_data.qpos[self._qpos_adr] = joint_pos[0].cpu().numpy()
    self._target_device.target_pos = self.mj_data.mocap_pos[self._mocap_id]
    return super().data


def launch(
    env: reaf_sim.Environment,
    policy: gdmr_policy.Policy[gdmr_types.PolicyDataType],
    *,
    loggers: Iterable[gdmr_logger.EpisodicLogger] = (),
    gfx: str = "web",
) -> None:
  """Launches the deployment viewer for a simulated or hardware environment."""
  sim = env.device_layer.sim
  if sim is not None:
    configure_viewer_spec(sim.mj_spec)
    step_duration = env.device_layer.control_timestep
  else:
    sim = RealSimulator(env.device_layer)
    step_duration = None
  loggers = tuple(loggers)
  reaf_viewer.launch(
      env,
      policy=policy,
      simulator=sim,
      step_duration=step_duration,
      reward_plot_keys=(),
      loggers=loggers,
      gfx=gfx,
  )
  for logger in loggers:
    logger.write()
