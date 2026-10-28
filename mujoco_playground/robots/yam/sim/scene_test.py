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
"""Tests for the YAM scene."""

from absl.testing import absltest
import mujoco
from mujoco_playground.robots.yam.real import constant
from mujoco_playground.robots.yam.sim import constant as sim_constant
from mujoco_playground.robots.yam.sim import scene


def _dof(model: mujoco.MjModel, joint: str) -> int:
  return model.joint(joint).dofadr[0]


class SceneTest(absltest.TestCase):

  def test_constants_cover_every_actuator(self) -> None:
    for table in (
        sim_constant.ARMATURE,
        sim_constant.FRICTIONLOSS,
        sim_constant.STIFFNESS,
        sim_constant.DAMPING,
    ):
      self.assertEqual(tuple(table), constant.JOINTS)

  def test_yam_applies_the_identified_dynamics(self) -> None:
    model = scene.yam().compile()

    for name in constant.JOINTS:
      with self.subTest(name):
        actuator = model.actuator(name)
        self.assertAlmostEqual(
            actuator.gainprm[0], sim_constant.STIFFNESS[name]
        )
        self.assertAlmostEqual(
            actuator.biasprm[1], -sim_constant.STIFFNESS[name]
        )
        self.assertAlmostEqual(actuator.biasprm[2], -sim_constant.DAMPING[name])
        joints = (
            sim_constant.FINGER_JOINTS
            if name == sim_constant.GRIPPER_ACTUATOR
            else (name,)
        )
        for joint in joints:
          dof = _dof(model, joint)
          self.assertAlmostEqual(
              model.dof_armature[dof], sim_constant.ARMATURE[name]
          )
          self.assertAlmostEqual(
              model.dof_frictionloss[dof], sim_constant.FRICTIONLOSS[name]
          )

  def test_yam_menagerie_keeps_menageries_values(self) -> None:
    model = scene.yam_menagerie().compile()

    # Menagerie's kp are the I2RT SDK's gains.
    for joint, kp in zip(constant.ARM_JOINTS, (80, 80, 80, 40, 10, 10)):
      self.assertAlmostEqual(model.actuator(joint).gainprm[0], kp)
    self.assertAlmostEqual(model.dof_armature[_dof(model, "joint1")], 0.032)

  def test_yam_holds_its_home_pose(self) -> None:
    model = scene.yam().compile()
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    home = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home)

    for _ in range(1000):  # 2 s.
      mujoco.mj_step(model, data)

    arm = [model.joint(j).qposadr[0] for j in constant.ARM_JOINTS]
    for i, q in zip(arm, model.key_qpos[home][arm]):
      self.assertAlmostEqual(data.qpos[i], q, delta=0.05)


if __name__ == "__main__":
  absltest.main()
