# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
import numpy as np
from reaf.simulation.torch._src.common import mujoco_camera_device
from reaf.simulation.torch._src.core import device_layer
from reaf.simulation.torch._src.core import simulator as simulator_module
from reaf.simulation.torch._src.core import types
import torch

# Simple cartpole model
_MODEL_XML = """
<mujoco>
  <option timestep="0.1"/>
  <visual>
    <global offwidth="320" offheight="240"/>
  </visual>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <body pos="0 0 0">
      <joint type="slide" axis="1 0 0"/>
      <geom type="box" size="0.5 0.5 0.1" rgba="1 0 0 1"/>
    </body>
    <camera name="fixed" pos="0 -2 2" xyaxes="1 0 0 0 1 1"/>
  </worldbody>
</mujoco>
"""


class MujocoCameraDeviceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._mj_spec = mujoco.MjSpec.from_string(_MODEL_XML)
    self._width = 320
    self._height = 240
    self._entities_to_close = []

  def tearDown(self) -> None:
    for entity in self._entities_to_close:
      entity.close()
    super().tearDown()

  def _create_entity(self, render_mode, render_rate_hz=10):
    # No impl here: the entity infers it from the model it is initialized with.
    entity = mujoco_camera_device.MujocoCameraDevice(
        camera_name="fixed",
        width=self._width,
        height=self._height,
        render_mode=render_mode,
        render_rate_hz=render_rate_hz,
    )
    self._entities_to_close.append(entity)
    return entity

  def _create_named_entity(
      self,
      prefix,
      delay_ms=0,
      render_rate_hz=10,
  ):
    entity = mujoco_camera_device.MujocoCameraDevice(
        camera_name="fixed",
        width=self._width,
        height=self._height,
        render_mode=mujoco_camera_device.RenderMode.RGB,
        render_rate_hz=render_rate_hz,
        delay_ms=delay_ms,
        image_output_name_prefix=prefix,
    )
    self._entities_to_close.append(entity)
    return entity

  def _create_device(self, entity, impl=types.Impl.CPP, nworld=1):
    return self._create_device_layer([entity], impl=impl, nworld=nworld)

  def _create_device_layer(self, entities, impl=types.Impl.CPP, nworld=1):
    device = (
        "cuda:0"
        if impl == types.Impl.WARP and torch.cuda.is_available()
        else "cpu"
    )
    sim = simulator_module.Simulator(
        mj_spec=self._mj_spec,
        nworld=nworld,
        xpu_device=device,
        impl=impl,
    )
    return device_layer.DeviceLayer(
        simulator=sim,
        control_timestep=0.1,
        devices=entities,
    )

  def _run_on_substep(self, device):
    """Runs the substep hooks of the devices without stepping physics."""
    for dev in device.devices:
      dev.on_substep(device.sim)

  @parameterized.product(
      impl=[
          ("cpp", types.Impl.CPP),
          ("warp", types.Impl.WARP),
      ],
      render_mode_spec=[
          ("rgb", mujoco_camera_device.RenderMode.RGB, ["fixed_rgb"]),
          ("depth", mujoco_camera_device.RenderMode.DEPTH, ["fixed_depth"]),
          (
              "segmentation",
              mujoco_camera_device.RenderMode.SEGMENTATION,
              ["fixed_segmentation"],
          ),
          (
              "rgb_depth",
              mujoco_camera_device.RenderMode.RGB
              | mujoco_camera_device.RenderMode.DEPTH,
              ["fixed_rgb", "fixed_depth"],
          ),
      ],
  )
  def test_rendering_returns_correct_shape(self, impl, render_mode_spec):
    _, impl = impl
    _, render_mode, expected_keys = render_mode_spec
    entity = self._create_entity(render_mode)

    # 1 environment (CPP only supports 1)
    nworld = 1
    device = self._create_device(entity, impl=impl, nworld=nworld)

    device.begin_stepping()
    measurements = device.get_measurements()

    for key in expected_keys:
      with self.subTest(key):
        self.assertIn(key, measurements)
        if "rgb" in key:
          self.assertEqual(
              measurements[key].shape, (nworld, self._height, self._width, 3)
          )
          self.assertEqual(measurements[key].dtype, torch.uint8)
        elif "depth" in key:
          self.assertEqual(
              measurements[key].shape, (nworld, self._height, self._width)
          )
          self.assertEqual(measurements[key].dtype, torch.float32)
        elif "segmentation" in key:
          self.assertEqual(
              measurements[key].shape, (nworld, self._height, self._width, 2)
          )
          self.assertEqual(measurements[key].dtype, torch.int32)

    if impl == types.Impl.WARP:
      with self.assertRaisesRegex(ValueError, "only supports Impl.CPP"):
        entity.render_now(device.sim)

  @parameterized.product(
      impl=[
          ("cpp", types.Impl.CPP),
          ("warp", types.Impl.WARP),
      ],
      render_mode_spec=[
          ("rgb", mujoco_camera_device.RenderMode.RGB, "fixed_rgb"),
          ("depth", mujoco_camera_device.RenderMode.DEPTH, "fixed_depth"),
          (
              "segmentation",
              mujoco_camera_device.RenderMode.SEGMENTATION,
              "fixed_segmentation",
          ),
      ],
  )
  def test_update_interval(self, impl, render_mode_spec):
    _, impl = impl
    _, render_mode, measurement_key = render_mode_spec
    # Timestep is 0.1s (10 Hz). 5 Hz -> interval 2.
    entity = self._create_entity(render_mode, render_rate_hz=5)

    nworld = 1
    device = self._create_device(entity, impl=impl, nworld=nworld)

    # Step 0: Should render
    device.begin_stepping()
    m0 = {k: v.clone() for k, v in device.get_measurements().items()}
    self.assertIn(measurement_key, m0)

    # Step 1: Advance physics and verify cached measurement is returned.
    device.sim.data.qpos[0] = 1.0  # Move cart
    device.sim.forward()
    device.step({})
    m1 = device.get_measurements()
    self.assertTrue(torch.equal(m0[measurement_key], m1[measurement_key]))

    # Step 2: Advance physics again to reach interval 2 -> should render again with new qpos
    device.step({})
    m2 = device.get_measurements()
    side = (
        entity.render_now(device.sim)[measurement_key]
        if impl == types.Impl.CPP
        else None
    )
    self.assertFalse(torch.equal(m0[measurement_key], m2[measurement_key]))
    if side is not None:
      self.assertTrue(torch.equal(m2[measurement_key], side))

  @parameterized.product(
      impl=[types.Impl.CPP, types.Impl.WARP],
      qpos_shape=["1d", "2d"],
  )
  def test_get_measurements_qpos_shapes(self, impl, qpos_shape):
    # This test ensures that MujocoCameraDevice can handle both 1D and 2D
    # qpos tensors. For WARP, this catches mutations in the qpos shape
    # handling logic.
    entity = self._create_entity(mujoco_camera_device.RenderMode.RGB)

    # Model with 2 DOFs to ensure scalar indexing fails if bug exists.
    xml = """
    <mujoco>
      <worldbody>
        <body pos="0 0 0">
          <joint name="j1" type="slide" axis="1 0 0"/>
          <joint name="j2" type="slide" axis="0 1 0"/>
          <geom type="sphere" size="0.1"/>
        </body>
        <camera name="fixed" pos="0 -2 2" xyaxes="1 0 0 0 1 1"/>
      </worldbody>
    </mujoco>
    """
    spec = mujoco.MjSpec.from_string(xml)
    model_obj = spec.compile()

    # Create real data objects for the respective implementations.
    if impl == types.Impl.CPP:
      mj_data = mujoco.MjData(model_obj)
      data = types.Data(mj_data, nworld=1)
      if qpos_shape == "1d":
        # TorchWrapper for MjData always provides 2D qpos, so we mock it here
        # to test the entity's robustness if it were ever passed 1D data.
        data = mock.create_autospec(types.Data, instance=True)
        data.nworld = 1
        data.qpos = torch.tensor([0.1, 0.2])
        data.obj = mj_data
    else:
      # For WARP, we mock the Data object to provide the desired qpos shape.
      data = mock.create_autospec(types.Data, instance=True)
      data.nworld = 1
      if qpos_shape == "1d":
        data.qpos = torch.tensor([0.1, 0.2])
      else:
        data.qpos = torch.tensor([[0.1, 0.2]])
    renderer = mujoco.Renderer(model_obj)
    self.addCleanup(renderer.close)

    sim = simulator_module.Simulator(
        mj_spec=spec,
        nworld=1,
        impl=impl,
    )
    with mock.patch.object(
        type(sim), "data", new_callable=mock.PropertyMock, return_value=data
    ):
      entity.initialize(sim)
      # Should not raise
      entity.on_begin_stepping(sim)

    if impl == types.Impl.WARP:
      # Verify qpos was copied correctly (checking internal state since it's a
      # unit test for the WARP copy logic).
      np.testing.assert_allclose(entity._data.qpos, [0.1, 0.2])

  def test_entity_initializes_own_renderer(self):
    entity1 = self._create_entity(mujoco_camera_device.RenderMode.RGB)
    entity2 = self._create_entity(mujoco_camera_device.RenderMode.RGB)

    device = self._create_device(entity1, impl=types.Impl.CPP)
    # SimulationDevice protocol no longer has 'renderer', and the class
    # doesn't have it either.
    self.assertFalse(hasattr(device, "renderer"))

    entity2.initialize(device.sim)

    self.assertIsInstance(entity1._renderer, mujoco.Renderer)
    self.assertIsInstance(entity2._renderer, mujoco.Renderer)
    self.assertNotEqual(entity1._renderer, entity2._renderer)

  def test_render_rate_hz(self):
    # Timestep is 0.1s (10 Hz). 5 Hz -> interval 2.
    entity = self._create_entity(
        mujoco_camera_device.RenderMode.RGB, render_rate_hz=5
    )
    device = self._create_device(entity)
    self.assertEqual(entity._update_interval_in_substeps, 2)
    del device

  def test_render_rate_hz_validation(self):
    with self.assertRaises(ValueError):
      mujoco_camera_device.MujocoCameraDevice(
          camera_name="fixed",
          width=self._width,
          height=self._height,
          render_rate_hz=0.0,
      )
    with self.assertRaises(ValueError):
      mujoco_camera_device.MujocoCameraDevice(
          camera_name="fixed",
          width=self._width,
          height=self._height,
          render_rate_hz=-10.0,
      )
    # None renders only after resets.
    mujoco_camera_device.MujocoCameraDevice(
        camera_name="fixed",
        width=self._width,
        height=self._height,
        render_rate_hz=None,
    )

  def test_none_render_rate_renders_only_after_reset(self):
    entity = self._create_entity(
        mujoco_camera_device.RenderMode.RGB, render_rate_hz=None
    )
    device = self._create_device(entity)

    # The first measurement after a reset renders.
    device.begin_stepping()
    first = device.get_measurements()["fixed_rgb"].clone()

    # Later measurements return the same frame, even after the cart moved.
    device.sim.data.qpos[0] = 1.0
    device.sim.forward()
    device.step({})
    self.assertTrue(torch.equal(device.get_measurements()["fixed_rgb"], first))

    # After a reset, the next measurement renders the current state. `reset`
    # restores qpos to the model default, so move the cart again first.
    device.reset()
    device.sim.data.qpos[0] = 1.0
    device.sim.forward()
    device.begin_stepping()
    self.assertFalse(torch.equal(device.get_measurements()["fixed_rgb"], first))

  def test_render_now_renders_current_state_without_side_effects(self):
    rgb_depth = (
        mujoco_camera_device.RenderMode.RGB
        | mujoco_camera_device.RenderMode.DEPTH
    )
    entity = self._create_entity(rgb_depth, render_rate_hz=None)
    # An on-demand-only device on the same camera publishes no measurements and
    # never renders on reset or during reads.
    on_demand = mujoco_camera_device.MujocoCameraDevice(
        camera_name="fixed",
        width=self._width,
        height=self._height,
        render_rate_hz=None,
        render_mode=rgb_depth,
        on_demand_only=True,
        name="on_demand_camera",
        image_output_name_prefix="on_demand",
    )
    self._entities_to_close.append(on_demand)
    self.assertEqual(on_demand.substep_measurements_spec(), {})
    device = self._create_device_layer([entity, on_demand])
    self.assertNotIn("on_demand_rgb", device.measurements_spec())

    uninitialized = mujoco_camera_device.MujocoCameraDevice(
        camera_name="fixed",
        width=self._width,
        height=self._height,
        render_rate_hz=None,
    )
    with self.assertRaisesRegex(RuntimeError, "Device not initialized"):
      uninitialized.render_now(device.sim)

    # Side-channel subset render before the first measurement leaves cache
    # empty.
    pre_step = entity.render_now(
        device.sim, render_mode=mujoco_camera_device.RenderMode.RGB
    )
    self.assertEqual(set(pre_step.keys()), {"fixed_rgb"})
    self.assertEqual(
        pre_step["fixed_rgb"].shape, (1, self._height, self._width, 3)
    )
    self.assertEqual(entity._last_measurements, {})

    assert on_demand._renderer is not None
    with mock.patch.object(
        on_demand._renderer, "render", wraps=on_demand._renderer.render
    ) as on_demand_spy:
      device.reset()
      device.begin_stepping()
      self.assertEqual(on_demand_spy.call_count, 0)
    self.assertNotIn("on_demand_rgb", device.get_measurements())
    first_rgb = device.get_measurements()["fixed_rgb"].clone()
    first_depth = device.get_measurements()["fixed_depth"].clone()

    # Without `render_rate_hz`, later measurements keep the reset frames.
    device.sim.data.qpos[0] = 1.0
    device.sim.forward()
    device.step({})
    self.assertTrue(
        torch.equal(device.get_measurements()["fixed_rgb"], first_rgb)
    )

    # `render_now` returns the current state without mutating measurements, and
    # `render_mode` runs only the requested pass.
    rendered = entity.render_now(device.sim)
    self.assertEqual(set(rendered.keys()), {"fixed_rgb", "fixed_depth"})
    self.assertFalse(torch.equal(rendered["fixed_rgb"], first_rgb))
    rendered_on_demand = on_demand.render_now(device.sim)
    self.assertEqual(
        set(rendered_on_demand.keys()), {"on_demand_rgb", "on_demand_depth"}
    )
    self.assertTrue(
        torch.equal(rendered_on_demand["on_demand_rgb"], rendered["fixed_rgb"])
    )

    assert entity._renderer is not None
    with mock.patch.object(
        entity._renderer, "render", wraps=entity._renderer.render
    ) as render_spy:
      rgb_only = entity.render_now(
          device.sim, render_mode=mujoco_camera_device.RenderMode.RGB
      )
      self.assertEqual(render_spy.call_count, 1)
    self.assertEqual(set(rgb_only.keys()), {"fixed_rgb"})
    self.assertTrue(torch.equal(rgb_only["fixed_rgb"], rendered["fixed_rgb"]))
    device.update_measurements()
    self.assertTrue(
        torch.equal(device.get_measurements()["fixed_rgb"], first_rgb)
    )
    self.assertTrue(
        torch.equal(device.get_measurements()["fixed_depth"], first_depth)
    )

    with self.assertRaisesRegex(ValueError, "not enabled on this device"):
      entity.render_now(
          device.sim,
          render_mode=mujoco_camera_device.RenderMode.SEGMENTATION,
      )
    with self.assertRaisesRegex(ValueError, "At least one render mode"):
      entity.render_now(
          device.sim, render_mode=mujoco_camera_device.RenderMode(0)
      )

  def test_delay_ms_validation(self):
    with self.assertRaisesRegex(ValueError, "on_demand_only requires"):
      mujoco_camera_device.MujocoCameraDevice(
          camera_name="fixed",
          width=self._width,
          height=self._height,
          render_rate_hz=10.0,
          on_demand_only=True,
      )
    with self.assertRaises(ValueError):
      mujoco_camera_device.MujocoCameraDevice(
          camera_name="fixed",
          width=self._width,
          height=self._height,
          render_rate_hz=10.0,
          render_mode=mujoco_camera_device.RenderMode.RGB,
          delay_ms=-1.0,
      )

  def test_discretization_warnings(self):
    # Timestep is 0.1s (100 ms).
    # render_rate_hz=3 (period 0.333s) and delay_ms=150 (0.15s) do not evenly
    # divide into sim_dt=0.1s and should both emit warnings.
    entity = self._create_named_entity(
        "warn_cam",
        delay_ms=150,
        render_rate_hz=3,
    )
    with self.assertLogs(mujoco_camera_device.__name__, level="WARNING") as cm:
      self._create_device(entity)
    log_text = "\n".join(cm.output)
    self.assertIn("render_rate_hz=3", log_text)
    self.assertIn("delay_ms=150", log_text)

  @parameterized.product(
      impl=[
          ("cpp", types.Impl.CPP),
          ("warp", types.Impl.WARP),
      ],
  )
  def test_delayed_rendering_matches_baseline_history(self, impl):
    """Delayed[t] renders the state from t - delay_in_substeps.

    A no-delay baseline and a delayed camera view the same joint, so rendering
    from the same qpos must produce pixel-identical images.

    Args:
      impl: The (name, impl) pair identifying the simulation backend.
    """
    _, impl = impl
    # Timestep is 0.1s (100 ms). 200 ms -> 2 substeps delay.
    delay_substeps = 2
    baseline = self._create_named_entity(
        "base",
        delay_ms=0,
        render_rate_hz=10,
    )
    delayed = self._create_named_entity(
        "delayed",
        delay_ms=200,
        render_rate_hz=10,
    )
    device = self._create_device_layer([baseline, delayed], impl=impl)

    if impl == types.Impl.CPP:
      # Before any substep is recorded, `render_now` renders live state without
      # mutating `_qpos_buffer`.
      device.sim.data.qpos[0] = 0.25
      device.sim.forward()
      self.assertTrue(
          torch.equal(
              delayed.render_now(device.sim)["delayed_rgb"],
              baseline.render_now(device.sim)["base_rgb"],
          )
      )
      self.assertEqual(delayed._total_stored, 0)
      self.assertEqual(delayed._last_recorded_substep, -1)

    total_substeps = 6
    imgs_base = []
    imgs_delayed = []
    for t in range(total_substeps):
      # Move the cart to a distinct pose each substep.
      device.sim.data.qpos[0] = 0.3 * t
      device.sim.forward()
      self._run_on_substep(device)
      device.update_measurements()
      m = device.get_measurements()
      imgs_base.append(m["base_rgb"].clone())
      imgs_delayed.append(m["delayed_rgb"].clone())
      device.clock.tick_substep()

    # Guard against an invisible scene: if every frame came back uniform (e.g.
    # all black), the equality assertions below would pass vacuously.
    for t, img in enumerate(imgs_base):
      self.assertGreater(
          int(img.max()) - int(img.min()),
          0,
          f"baseline[{t}] is a uniform image; the scene may not be rendering",
      )

    # Motion produces distinct frames, so the newest frames must differ.
    self.assertFalse(torch.equal(imgs_delayed[-1], imgs_base[-1]))

    # Until `delay_substeps` frames of history exist, the delay is clamped to
    # what is available, so the early frames replay the oldest recorded state.
    for t in range(delay_substeps):
      self.assertTrue(
          torch.equal(imgs_delayed[t], imgs_base[0]),
          f"delayed[{t}] should be clamped to baseline[0]",
      )

    # Core invariant: delayed[t] == baseline[t - delay_substeps].
    for t in range(delay_substeps, total_substeps):
      self.assertTrue(
          torch.equal(imgs_delayed[t], imgs_base[t - delay_substeps]),
          f"delayed[{t}] should match baseline[{t - delay_substeps}]",
      )

    if impl == types.Impl.CPP:
      stored_before = delayed._total_stored
      write_before = delayed._write_idx
      delayed_now = delayed.render_now(device.sim)["delayed_rgb"]
      self.assertEqual(delayed._total_stored, stored_before)
      self.assertEqual(delayed._write_idx, write_before)
      self.assertTrue(
          torch.equal(
              delayed_now, imgs_base[total_substeps - 1 - delay_substeps]
          )
      )

  @parameterized.product(
      impl=[
          ("cpp", types.Impl.CPP),
          ("warp", types.Impl.WARP),
      ],
  )
  def test_delayed_rendering_is_independent_of_render_rate(self, impl):
    """Verifies delay_ms is relative to physical time (substeps), not render ticks."""
    _, impl = impl
    # With sim timestep 0.1s (100 ms):
    # - base_10hz renders every substep (10 Hz) with 0 ms delay.
    # - delayed_5hz renders every 2 substeps (5 Hz) with 100 ms (1 substep) delay.
    # On substep 2, delayed_5hz must render the pose from substep 1 (100 ms ago),
    # NOT substep 0 (1 render tick / 200 ms ago).
    base_10hz = self._create_named_entity(
        "base",
        delay_ms=0,
        render_rate_hz=10,
    )
    delayed_5hz = self._create_named_entity(
        "delayed",
        delay_ms=100,
        render_rate_hz=5,
    )
    device = self._create_device_layer([base_10hz, delayed_5hz], impl=impl)

    # Substep 0: pose = 0.0. Both render pose 0.0.
    device.sim.data.qpos[0] = 0.0
    device.sim.forward()
    self._run_on_substep(device)
    device.update_measurements()
    m0 = {k: v.clone() for k, v in device.get_measurements().items()}
    self.assertTrue(torch.equal(m0["delayed_rgb"], m0["base_rgb"]))
    device.clock.tick_substep()

    # Substep 1: pose = 1.0. base_10hz renders pose 1.0; delayed_5hz records
    # pose 1.0 into its buffer and returns its cached frame from substep 0.
    device.sim.data.qpos[0] = 1.0
    device.sim.forward()
    self._run_on_substep(device)
    device.update_measurements()
    m1 = {k: v.clone() for k, v in device.get_measurements().items()}
    self.assertTrue(torch.equal(m1["delayed_rgb"], m0["base_rgb"]))
    self.assertFalse(torch.equal(m1["base_rgb"], m0["base_rgb"]))
    device.clock.tick_substep()

    # Substep 2: pose = 2.0. delayed_5hz renders at 5 Hz cadence, using the
    # qpos from 100 ms ago (substep 1, pose 1.0, which equals m1["base_rgb"]).
    device.sim.data.qpos[0] = 2.0
    device.sim.forward()
    self._run_on_substep(device)
    device.update_measurements()
    m2 = {k: v.clone() for k, v in device.get_measurements().items()}
    self.assertTrue(torch.equal(m2["delayed_rgb"], m1["base_rgb"]))
    self.assertFalse(torch.equal(m2["delayed_rgb"], m0["base_rgb"]))

  @parameterized.product(
      impl=[types.Impl.CPP, types.Impl.WARP],
  )
  def test_reset_clears_delay_history(self, impl):
    """The first render after a reset is undelayed.

    Args:
      impl: The simulation backend to run against.
    """
    delay_substeps = 2
    baseline = self._create_named_entity("base", delay_ms=0)
    delayed = self._create_named_entity("delayed", delay_ms=200)
    device = self._create_device_layer([baseline, delayed], impl=impl)

    # Accumulate enough history that the delay is fully engaged.
    for t in range(delay_substeps + 2):
      device.sim.data.qpos[0] = 0.3 * t
      device.sim.forward()
      self._run_on_substep(device)
      device.update_measurements()
      device.clock.tick_substep()
    measurements = device.get_measurements()
    self.assertFalse(
        torch.equal(measurements["delayed_rgb"], measurements["base_rgb"])
    )

    device.reset()

    # With the history cleared the delay clamps to zero again, so the delayed
    # camera must agree with the undelayed baseline. `reset` also restores qpos
    # to the model default, hence the explicit pose below.
    device.sim.data.qpos[0] = 2.0
    device.sim.forward()
    device.begin_stepping()
    measurements = device.get_measurements()
    self.assertTrue(
        torch.equal(measurements["delayed_rgb"], measurements["base_rgb"])
    )

  def test_reset_mid_interval_rerenders(self):
    # Timestep is 0.1s (10 Hz). 5 Hz -> interval 2 substeps.
    entity = self._create_entity(
        mujoco_camera_device.RenderMode.RGB, render_rate_hz=5
    )
    device = self._create_device(entity, impl=types.Impl.CPP, nworld=1)

    # Render at substep 0, then step once to substep 1 (non-multiple of 2).
    device.begin_stepping()
    device.step({})
    self.assertEqual(device.clock.global_substep, 1)

    # Reset with env_mask=[True] while global_substep is 1 (odd).
    # _last_measurements is cleared on reset, so begin_stepping() must render a
    # fresh frame even though this is not a render tick.
    env_mask = torch.tensor([True])
    device.reset(env_mask=env_mask)
    device.sim.data.qpos[0] = 1.5
    device.sim.forward()
    device.begin_stepping(env_mask)
    measurements = device.get_measurements()
    self.assertIn("fixed_rgb", measurements)
    self.assertEqual(
        measurements["fixed_rgb"].shape, (1, self._height, self._width, 3)
    )

  def test_close_is_idempotent_before_and_after_initialize(self) -> None:
    entity = self._create_entity(mujoco_camera_device.RenderMode.RGB)
    self.assertIsNone(entity._renderer)
    entity.close()
    self.assertIsNone(entity._renderer)

    device = self._create_device(entity, impl=types.Impl.CPP, nworld=1)
    device.begin_stepping()
    self.assertIsNotNone(entity._renderer)
    assert entity._renderer is not None

    with mock.patch.object(
        entity._renderer, "close", wraps=entity._renderer.close
    ) as mock_close:
      entity.close()
      self.assertIsNone(entity._renderer)
      entity.close()
      self.assertIsNone(entity._renderer)

    mock_close.assert_called_once()
    device.reset()
    with self.assertRaisesRegex(RuntimeError, "Renderer is not initialized"):
      device.begin_stepping()


if __name__ == "__main__":
  absltest.main()
