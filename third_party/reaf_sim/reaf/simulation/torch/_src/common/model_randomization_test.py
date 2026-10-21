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

"""Tests for model_randomization utilities."""

from absl.testing import absltest
import mujoco
import mujoco_warp as mjw
from reaf.simulation.torch._src.common import model_randomization
from reaf.simulation.torch._src.common import noise
from reaf.simulation.torch._src.core import torch_wrapper
import torch
import warp as wp

XML = """
<mujoco model="randomization_test">
  <worldbody>
    <body name="floor">
      <geom name="floor" type="plane" size="1 1 0.1" friction="0.5 0.1 0.01"/>
    </body>
    <body name="box1" pos="0 0 1">
      <freejoint/>
      <geom name="box1_geom" type="box" size="0.1 0.1 0.1" friction="0.8 0.2 0.02"/>
    </body>
    <body name="box2" pos="0 0 2">
      <freejoint/>
      <geom name="box2_geom" type="box" size="0.1 0.1 0.1" friction="0.6 0.15 0.015"/>
    </body>
    <body name="robot_link1" pos="1 0 1">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom name="robot_geom1" type="capsule" size="0.05 0.2" friction="0.7 0.1 0.01"/>
      <body name="robot_link2" pos="0.4 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1"/>
        <geom name="robot_geom2" type="capsule" size="0.05 0.2" friction="0.7 0.1 0.01"/>
      </body>
    </body>
    <camera name="wrist_cam" pos="0 -1 1" fovy="61"/>
  </worldbody>
  <actuator>
    <general name="test_actuator" joint="joint1" biastype="affine" gainprm="10" biasprm="0 -10 0.1"/>
  </actuator>
</mujoco>
"""


_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class RandomizeFieldTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    wp.init()
    wp.set_device(_DEVICE)
    wp.config.kernel_cache_dir = "/tmp/kernel_cache"
    self.nworld = 4
    self.spec = mujoco.MjSpec.from_string(XML)
    self.mj_model = self.spec.compile()
    self.mjw_model = mjw.put_model(self.mj_model)
    self.mj_data = mujoco.MjData(self.mj_model)
    self.mjw_data = mjw.put_data(
        self.mj_model, self.mj_data, nworld=self.nworld
    )
    self.model = torch_wrapper.TorchWrapper(self.mjw_model, nworld=self.nworld)
    self.device = self.model.geom_friction.device

  def test_randomize_geom_friction_all_axes(self):
    """Test randomizing all friction axes for matched geoms."""
    original = self.model.geom_friction[:].clone()

    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="geom_friction",
        pattern="box.*",
        noise=noise.Noise(dist=noise.Uniform(low=0.1, high=0.9), op="replace"),
    )
    self.assertEqual(
        self.model.pending_set_const, torch_wrapper.SetConstLevel.NONE
    )

    new = self.model.geom_friction[:]
    # Match box1_geom and box2_geom (IDs 1 and 2).
    box_ids = [
        self.mj_model.geom("box1_geom").id,
        self.mj_model.geom("box2_geom").id,
    ]
    for geom_id in box_ids:
      for axis in range(3):
        # Values should be changed.
        self.assertFalse(
            torch.allclose(original[:, geom_id, axis], new[:, geom_id, axis])
        )
        # Values should be in range.
        self.assertTrue((new[:, geom_id, axis] >= 0.1).all())
        self.assertTrue((new[:, geom_id, axis] <= 0.9).all())

    # Floor geom should be unchanged.
    floor_id = self.mj_model.geom("floor").id
    torch.testing.assert_close(original[:, floor_id], new[:, floor_id])

  def test_randomize_geom_friction_single_axis(self):
    """Test randomizing only slide friction (axis 0)."""
    original = self.model.geom_friction[:].clone()

    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="geom_friction",
        pattern="robot_geom.*",
        noise={
            0: noise.Noise(dist=noise.Uniform(low=0.3, high=1.5), op="replace")
        },
    )
    self.assertEqual(
        self.model.pending_set_const, torch_wrapper.SetConstLevel.NONE
    )

    new = self.model.geom_friction[:]
    robot_ids = [
        self.mj_model.geom("robot_geom1").id,
        self.mj_model.geom("robot_geom2").id,
    ]
    for geom_id in robot_ids:
      # Slide friction (axis 0) should be randomized.
      self.assertTrue((new[:, geom_id, 0] >= 0.3).all())
      self.assertTrue((new[:, geom_id, 0] <= 1.5).all())
      # Spin and roll friction (axes 1, 2) should be unchanged.
      torch.testing.assert_close(original[:, geom_id, 1:], new[:, geom_id, 1:])

  def test_randomize_body_mass(self):
    """Test randomizing body mass with scale operation."""
    original = self.model.body_mass[:].clone()

    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="body_mass",
        pattern="box.*",
        noise=noise.Noise(dist=noise.Uniform(low=0.8, high=1.2), op="scale"),
    )
    self.assertEqual(
        self.model.pending_set_const, torch_wrapper.SetConstLevel.FULL
    )

    new = self.model.body_mass[:]
    box_ids = [self.mj_model.body("box1").id, self.mj_model.body("box2").id]
    for body_id in box_ids:
      # Mass should be scaled within 0.8-1.2x.
      ratio = new[:, body_id] / original[:, body_id]
      self.assertTrue((ratio >= 0.8).all())
      self.assertTrue((ratio <= 1.2).all())

  def test_randomize_cam_fovy(self):
    """Test randomizing camera fovy, the field the renderer reads.

    A camera that declares fovy (rather than sensorsize/focal) is rendered
    from cam_fovy alone, so this is the only FOV field worth randomizing for
    such a camera. Assert the values both land in range and actually differ
    across worlds, so a term that silently fails to bind is caught.
    """
    cam_id = self.mj_model.camera("wrist_cam").id
    original = self.model.cam_fovy[:].clone()

    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="cam_fovy",
        pattern="wrist_cam",
        noise=noise.Noise(dist=noise.Uniform(low=0.9, high=1.1), op="scale"),
    )
    self.assertEqual(
        self.model.pending_set_const, torch_wrapper.SetConstLevel.NONE
    )

    new = self.model.cam_fovy[:, cam_id]
    ratio = new / original[:, cam_id]
    self.assertTrue((ratio >= 0.9).all())
    self.assertTrue((ratio <= 1.1).all())
    # Ensure randomization actually changed values from original.
    self.assertFalse(torch.allclose(new, original[:, cam_id]))
    # Every world must get its own draw.
    self.assertLen(torch.unique(new), self.nworld)

  def test_randomize_geom_size_recomputes_bounds(self):
    """Test that randomizing geom_size analytically recomputes rbound and aabb."""
    original_size = self.model.geom_size[:].clone()

    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="geom_size",
        pattern="box.*",
        noise=noise.Noise(dist=noise.Uniform(low=1.5, high=1.5), op="scale"),
    )
    self.assertEqual(
        self.model.pending_set_const, torch_wrapper.SetConstLevel.FIXED
    )

    new_size = self.model.geom_size[:]
    new_rbound = self.model.geom_rbound[:]
    new_aabb = self.model.geom_aabb[:]

    box_ids = [
        self.mj_model.geom("box1_geom").id,
        self.mj_model.geom("box2_geom").id,
    ]

    for geom_id in box_ids:
      # Size should be exactly 1.5x scaled
      torch.testing.assert_close(
          new_size[:, geom_id], original_size[:, geom_id] * 1.5
      )

      # Box geometry bounds recomputation:
      # rbound = sqrt(s0^2 + s1^2 + s2^2)
      s = new_size[:, geom_id]
      expected_rbound = torch.sqrt((s * s).sum(dim=-1))
      torch.testing.assert_close(new_rbound[:, geom_id], expected_rbound)

      # aabb index 1 (half-size) should match size
      torch.testing.assert_close(new_aabb[:, geom_id, 1], s)

  def test_randomize_single_body_mass_scales_inertia(self):
    """Test that randomizing body_mass on a single body scales body_inertia."""
    box1_id = self.mj_model.body("box1").id
    orig_mass = self.model.body_mass[:, box1_id].clone()
    orig_inertia = self.model.body_inertia[:, box1_id].clone()

    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="body_mass",
        pattern="box1",
        noise=noise.Noise(dist=noise.Uniform(low=2.0, high=2.0), op="scale"),
    )
    torch.testing.assert_close(
        self.model.body_mass[:, box1_id], orig_mass * 2.0
    )
    torch.testing.assert_close(
        self.model.body_inertia[:, box1_id],
        orig_inertia * 2.0,
        rtol=1e-5,
        atol=1e-5,
    )

  def test_randomize_body_mass_preserves_massless_bodies(self):
    """Massless bodies matched by a pattern stay at zero mass and inertia."""
    floor_id = self.mj_model.body("floor").id
    box1_id = self.mj_model.body("box1").id
    orig_box1_mass = self.model.body_mass[:, box1_id].clone()

    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="body_mass",
        pattern=".*",
        noise=noise.Noise(dist=noise.Uniform(low=0.5, high=0.5), op="add"),
    )
    torch.testing.assert_close(
        self.model.body_mass[:, floor_id],
        torch.zeros_like(self.model.body_mass[:, floor_id]),
    )
    torch.testing.assert_close(
        self.model.body_inertia[:, floor_id],
        torch.zeros_like(self.model.body_inertia[:, floor_id]),
    )
    torch.testing.assert_close(
        self.model.body_mass[:, box1_id], orig_box1_mass + 0.5
    )

  def test_randomize_field_with_custom_post_processor(self):
    """Test that passing a custom post-processor list works perfectly."""
    called_args = []

    def mock_processor(model, spec, bound, env_mask, current, result):
      # Actively apply a post-processing side-effect on a derived field
      bound.set("rbound", result[..., 0] * 2.0, env_mask=env_mask)
      called_args.append((model, spec, bound, env_mask, current, result))

    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="geom_friction",
        pattern="box1_geom",
        noise=noise.Noise(
            dist=noise.Uniform(low=0.15, high=0.15), op="replace"
        ),
        custom_post_processors=mock_processor,
    )
    self.assertEqual(
        self.model.pending_set_const, torch_wrapper.SetConstLevel.NONE
    )

    # Verify that the custom processor was called exactly once.
    self.assertEqual(len(called_args), 1)
    model, spec, _, env_mask, current, _ = called_args[0]
    self.assertEqual(model, self.model)
    self.assertEqual(spec, self.spec)
    self.assertIsNone(env_mask)

    # The custom post-processor should have updated rbound from result.
    geom_id = self.mj_model.geom("box1_geom").id
    torch.testing.assert_close(
        self.model.geom_rbound[:, geom_id],
        torch.full_like(current[..., 0], 0.3),
    )

  def test_randomize_with_env_mask(self):
    """Test randomizing only a subset of environments."""
    original = self.model.geom_friction[:].clone()
    env_mask = torch.tensor(
        [False, True, False, True], dtype=torch.bool, device=self.device
    )

    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="geom_friction",
        pattern="floor",
        noise=noise.Noise(dist=noise.Uniform(low=0.1, high=0.2), op="replace"),
        env_mask=env_mask,
    )

    new = self.model.geom_friction[:]
    floor_id = self.mj_model.geom("floor").id

    # Envs 0 and 2 should be unchanged.
    torch.testing.assert_close(original[0, floor_id], new[0, floor_id])
    torch.testing.assert_close(original[2, floor_id], new[2, floor_id])

    # Envs 1 and 3 should be randomized.
    self.assertFalse(torch.allclose(original[1, floor_id], new[1, floor_id]))
    self.assertFalse(torch.allclose(original[3, floor_id], new[3, floor_id]))

  def test_unknown_field_raises(self):
    """Test that unknown field prefix raises ValueError."""
    with self.assertRaisesRegex(ValueError, "Unknown field prefix"):
      model_randomization.randomize_field(
          self.model,  # pyrefly: ignore[bad-argument-type]
          self.spec,
          self.device,
          field="invalid_field",
          pattern=".*",
          noise=noise.Noise(
              dist=noise.Uniform(low=0.1, high=0.9), op="replace"
          ),
      )

  def test_caching(self):
    """Test that bind_pattern is cached across calls."""
    self.assertEqual(len(self.model._bound_cache), 0)

    # First call.
    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="geom_friction",
        pattern="box.*",
        noise=noise.Noise(dist=noise.Uniform(low=0.1, high=0.9), op="replace"),
    )
    cache_size_1 = len(self.model._bound_cache)

    # Second call with same pattern - should reuse cache.
    model_randomization.randomize_field(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        field="geom_friction",
        pattern="box.*",
        noise=noise.Noise(dist=noise.Uniform(low=0.2, high=0.8), op="replace"),
    )
    cache_size_2 = len(self.model._bound_cache)

    self.assertEqual(cache_size_1, cache_size_2)

  def test_randomize_body_mass_dict_invalid(self):
    """Test that dict ranges with invalid keys for scalar fields raise ValueError."""
    with self.assertRaisesRegex(
        ValueError, "does not support per-axis noise dict"
    ):
      model_randomization.randomize_field(
          self.model,  # pyrefly: ignore[bad-argument-type]
          self.spec,
          self.device,
          field="body_mass",
          pattern="box.*",
          noise={
              1: noise.Noise(dist=noise.Uniform(low=0.8, high=1.2), op="scale")
          },
      )

  def test_randomize_actuator_gains(self):
    """Test randomizing actuator gains."""
    original_gainprm = self.model.actuator_gainprm[:].clone()
    original_biasprm = self.model.actuator_biasprm[:].clone()

    model_randomization.randomize_actuator_gains(
        self.model,  # pyrefly: ignore[bad-argument-type]
        self.spec,
        self.device,
        pattern="test_actuator",
        kp_noise=noise.Noise(dist=noise.Uniform(low=0.8, high=1.2), op="scale"),
        kd_noise=noise.Noise(dist=noise.Uniform(low=0.5, high=1.5), op="scale"),
    )
    self.assertEqual(
        self.model.pending_set_const, torch_wrapper.SetConstLevel.NONE
    )

    new_gainprm = self.model.actuator_gainprm[:]
    new_biasprm = self.model.actuator_biasprm[:]

    actuator_id = self.mj_model.actuator("test_actuator").id

    # kp (gainprm[0]) should be randomized.
    self.assertFalse(
        torch.allclose(
            original_gainprm[:, actuator_id, 0],
            new_gainprm[:, actuator_id, 0],
        )
    )
    # biasprm[1] should be -kp.
    torch.testing.assert_close(
        new_biasprm[:, actuator_id, 1],
        -new_gainprm[:, actuator_id, 0],
    )

    # kd (calculated from biasprm[2] which is -kd) should be randomized.
    self.assertFalse(
        torch.allclose(
            original_biasprm[:, actuator_id, 2],
            new_biasprm[:, actuator_id, 2],
        )
    )


if __name__ == "__main__":
  absltest.main()
