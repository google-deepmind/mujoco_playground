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
from absl.testing import parameterized
import mujoco
import numpy as np
from reaf.simulation.torch._src.core import mujoco_types

_XML = """
<mujoco model="testbed">
  <worldbody>
    <body name="box">
      <joint name="slide" type="slide"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1"/>
      <site name="tip"/>
      <light name="spot" pos="0 0 1"/>
      <camera name="eye"/>
    </body>
    <body name="ball" pos="1 0 0">
      <joint name="free_jnt" type="slide"/>
      <geom name="ball_geom" type="sphere" size="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="slide"/>
    <motor name="motor2" joint="free_jnt"/>
  </actuator>
  <sensor>
    <jointpos name="jpos" joint="slide"/>
  </sensor>
  <tendon>
    <fixed name="ten"><joint joint="slide" coef="1"/></fixed>
  </tendon>
  <equality>
    <joint name="eq" joint1="slide"/>
  </equality>
</mujoco>
"""


class GetSpecByObjTypeTest(parameterized.TestCase):
  """Tests that get_spec_by_obj_type returns the correct spec object."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.spec = mujoco.MjSpec.from_string(_XML)

  @parameterized.parameters(
      (mujoco.mjtObj.mjOBJ_BODY, "box"),
      (mujoco.mjtObj.mjOBJ_JOINT, "slide"),
      (mujoco.mjtObj.mjOBJ_GEOM, "box_geom"),
      (mujoco.mjtObj.mjOBJ_SITE, "tip"),
      (mujoco.mjtObj.mjOBJ_LIGHT, "spot"),
      (mujoco.mjtObj.mjOBJ_CAMERA, "eye"),
      (mujoco.mjtObj.mjOBJ_ACTUATOR, "motor"),
      (mujoco.mjtObj.mjOBJ_SENSOR, "jpos"),
      (mujoco.mjtObj.mjOBJ_TENDON, "ten"),
      (mujoco.mjtObj.mjOBJ_EQUALITY, "eq"),
  )
  def test_accessor_returns_named_object(self, obj_type, name):
    result = mujoco_types.get_spec_by_obj_type(self.spec, obj_type, name)
    self.assertEqual(result.name, name)


class ResolveNamesTest(absltest.TestCase):
  """Tests for resolve_names with str and tuple patterns."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.spec = mujoco.MjSpec.from_string(_XML)

  def test_exact_str(self):
    result = mujoco_types.resolve_names(
        self.spec, mujoco.mjtObj.mjOBJ_BODY, "box"
    )
    self.assertEqual(result, ["box"])

  def test_regex_str(self):
    result = mujoco_types.resolve_names(
        self.spec, mujoco.mjtObj.mjOBJ_GEOM, "box.*"
    )
    self.assertIn("box_geom", result)

  def test_tuple_of_names(self):
    result = mujoco_types.resolve_names(
        self.spec, mujoco.mjtObj.mjOBJ_ACTUATOR, ("motor", "motor2")
    )
    self.assertLen(result, 2)
    self.assertIn("motor", result)
    self.assertIn("motor2", result)

  def test_no_match_raises(self):
    with self.assertRaises(ValueError):
      mujoco_types.resolve_names(
          self.spec, mujoco.mjtObj.mjOBJ_BODY, "nonexistent"
      )


_NAMES = (
    "left_hip_joint",
    "left_knee_joint",
    "right_hip_joint",
    "right_knee_joint",
    "waist_joint",
)


class ResolveValuesTest(absltest.TestCase):

  def test_patterns_and_default(self):
    values = mujoco_types.resolve_values(
        {".*_knee_joint": 0.5, "left_hip_joint": -0.1},
        _NAMES,
        default=9.0,
    )
    self.assertEqual(values, [-0.1, 0.5, 9.0, 0.5, 9.0])

  def test_requires_full_match(self):
    values = mujoco_types.resolve_values({"knee": 1.0}, _NAMES)
    self.assertEqual(values, [0.0] * len(_NAMES))

  def test_multiple_matches_raises(self):
    with self.assertRaisesRegex(ValueError, "left_knee_joint"):
      mujoco_types.resolve_values(
          {".*_knee_joint": 1.0, "left_.*": 2.0}, _NAMES
      )


class MatchNamesTest(absltest.TestCase):

  def test_preserves_name_order(self):
    self.assertEqual(
        mujoco_types.match_names(("waist_joint", "left_.*"), _NAMES),
        ["left_hip_joint", "left_knee_joint", "waist_joint"],
    )

  def test_requires_full_match(self):
    self.assertEqual(mujoco_types.match_names(("hip",), _NAMES), [])


class SoftJointLimitsTest(absltest.TestCase):

  def test_shrinks_range_about_midpoint(self):
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body>
              <joint name="a" type="hinge" range="-1 3"/>
              <geom size="0.1"/>
              <body>
                <joint name="b" type="hinge" range="0 2"/>
                <geom size="0.1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
    """)
    limits = mujoco_types.soft_joint_limits(model, ("b", "a"), factor=0.5)
    np.testing.assert_allclose(limits, [[0.5, 1.5], [0.0, 2.0]])


if __name__ == "__main__":
  absltest.main()
