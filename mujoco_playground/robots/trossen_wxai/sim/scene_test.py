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

from absl.testing import absltest
from mujoco_playground import asset
from mujoco_playground.robots.trossen_wxai.sim import constant
from mujoco_playground.robots.trossen_wxai.sim import scene


class SceneTest(absltest.TestCase):

  def test_plane(self) -> None:
    spec = scene.trossen_wxai_biarm_plane()
    self.assertIsNotNone(spec)
    model = spec.compile()
    self.assertGreater(model.nu, 0)

  def test_table(self) -> None:
    spec = scene.trossen_wxai_biarm_table()
    self.assertIsNotNone(spec)
    model = spec.compile()
    self.assertGreater(model.nu, 0)

  def test_constants_cover_every_actuator(self) -> None:
    for table in (
        constant.STIFFNESS,
        constant.DAMPING,
        constant.FORCERANGE,
        constant.CTRLRANGE,
    ):
      self.assertEqual(tuple(table), tuple(constant.JOINTS))

  def test_scene_applies_identified_dynamics(self) -> None:
    model = scene.trossen_wxai_biarm_table().compile()
    xml_model = (
        asset.get(
            "github:google-deepmind/mujoco_menagerie/trossen_wxai/"
            "wxai_follower_biarm.xml"
        )
        .spec()
        .compile()
    )

    for joint in constant.JOINTS:
      with self.subTest(joint):
        name = f"right/{joint}"
        actuator = model.actuator(name)
        jnt = model.joint(name)
        dof = jnt.dofadr[0]
        kp = constant.STIFFNESS[joint]
        self.assertAlmostEqual(float(actuator.gainprm[0]), kp)
        self.assertAlmostEqual(float(actuator.biasprm[1]), -kp)
        self.assertAlmostEqual(float(actuator.biasprm[2]), 0.0)
        self.assertAlmostEqual(
            float(model.dof_damping[dof]), constant.DAMPING[joint]
        )
        expected_forcerange = min(
            float(xml_model.actuator(name).forcerange[1]),
            float(model.jnt_actfrcrange[jnt.id, 1]),
        )
        self.assertAlmostEqual(
            float(actuator.forcerange[0]), -expected_forcerange
        )
        self.assertAlmostEqual(
            float(actuator.forcerange[1]), expected_forcerange
        )


if __name__ == "__main__":
  absltest.main()
