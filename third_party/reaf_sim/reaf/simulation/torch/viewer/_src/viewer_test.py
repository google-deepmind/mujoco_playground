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

from unittest import mock

from absl.testing import absltest
import dm_env
from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch import simulator as simulator_module
from reaf.simulation.torch import types
from reaf.simulation.torch.viewer._src import sim_backend
from reaf.simulation.torch.viewer._src import viewer
import torch


class ViewerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.spec = mujoco.MjSpec()
    body = self.spec.worldbody.add_body(name='box')
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0.1, 0.1])
    goal = self.spec.worldbody.add_body(name='goal', mocap=True)
    goal.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0.05, 0.05])
    self.sim = simulator_module.Simulator(
        mj_spec=self.spec, nworld=1, impl=types.Impl.CPP, xpu_device='cpu'
    )

  def _make_backend(
      self,
      sim: simulator_module.Simulator | None = None,
      xpu_device: str = 'cpu',
      world_id: int = 0,
  ) -> sim_backend.TorchSimBackend:
    action_spec = {'action': specs.Array(shape=(4,), dtype=np.float32)}
    policy = viewer._ZeroActionPolicy(action_spec, xpu_device=xpu_device)
    return sim_backend.TorchSimBackend(
        environment=mock.MagicMock(),
        policy=policy,
        simulator=sim or self.sim,
        xpu_device=xpu_device,
        world_id=world_id,
    )

  def test_zero_action_policy(self):
    action_spec = {
        'action': specs.Array(shape=(4,), dtype=np.float32),
    }
    policy = viewer._ZeroActionPolicy(action_spec, xpu_device='cpu')
    self.assertEqual(policy.initial_state(), ())
    assert isinstance(policy._action, dict)
    self.assertIn('action', policy._action)
    self.assertEqual(policy._action['action'].shape, (4,))
    self.assertTrue(torch.all(policy._action['action'] == 0))

    (action, policy_extra), state = policy.step(dm_env.restart({}))
    self.assertIn('action', action)
    self.assertEqual(policy_extra, {})
    self.assertEqual(state, ())

  def test_sync_to_viewer(self):
    model = self.spec.compile()
    data = mujoco.MjData(model)
    backend = self._make_backend()
    self.sim.model.opt.gravity[0] = torch.tensor([0.0, 0.0, -5.0])
    backend.sync_to_viewer(model, data)
    np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, -5.0], atol=1e-6)
    self.assertEqual(len(model.geom_size), len(self.sim.mj_model.geom_size))

  def test_sync_to_viewer_tracking_and_geom_dr(self):
    xpu_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.spec.worldbody.add_camera(
        name='track_cam',
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
        targetbody='box',
        pos=[0.0, -1.0, 0.5],
    )
    self.spec.worldbody.add_light(
        name='track_light',
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
        targetbody='box',
        pos=[0.0, 0.0, 2.0],
    )
    self.spec.worldbody.add_site(name='unused_site', pos=[0.0, 0.0, 0.0])
    sim = simulator_module.Simulator(
        mj_spec=self.spec, nworld=2, impl=types.Impl.WARP, xpu_device=xpu_device
    )
    sim.model.geom_pos[1, 0] = torch.tensor([0.2, -0.1, 0.3], device=xpu_device)
    sim.model.geom_quat[1, 0] = torch.tensor(
        [0.5, 0.5, 0.5, 0.5], device=xpu_device
    )
    sim.model.cam_pos[1, 0] = torch.tensor([0.4, -1.5, 0.8], device=xpu_device)
    sim.model.light_pos[1, 0] = torch.tensor(
        [-0.3, 0.2, 2.5], device=xpu_device
    )
    sim.set_const()
    sim.forward()

    model = self.spec.compile()
    data = mujoco.MjData(model)
    self._make_backend(sim, xpu_device, world_id=1).sync_to_viewer(model, data)

    self.assertEqual(sim.warp_model.site_pos.shape[0], 1)
    for field in (
        'geom_pos',
        'geom_quat',
        'cam_pos0',
        'cam_mat0',
        'light_pos0',
    ):
      np.testing.assert_allclose(
          getattr(model, field)[0],
          getattr(sim.model, field)[1, 0].cpu().numpy().ravel(),
          atol=1e-6,
      )
    for field in (
        'geom_xpos',
        'geom_xmat',
        'cam_xpos',
        'cam_xmat',
        'light_xpos',
    ):
      np.testing.assert_allclose(
          getattr(data, field)[0].ravel(),
          getattr(sim.data, field)[1, 0].cpu().numpy().ravel(),
          atol=1e-5,
      )

  def test_apply_mocap(self):
    backend = self._make_backend()
    backend.apply_mocap(
        np.array([[0.1, 0.2, 0.3]]), np.array([[0.0, 1.0, 0.0, 0.0]])
    )
    np.testing.assert_allclose(
        self.sim.data.mocap_pos[0].cpu().numpy(), [[0.1, 0.2, 0.3]]
    )
    np.testing.assert_allclose(
        self.sim.data.mocap_quat[0].cpu().numpy(), [[0.0, 1.0, 0.0, 0.0]]
    )
    # The pose reaches the CPU mirror on the next sync.
    model = self.spec.compile()
    data = mujoco.MjData(model)
    backend.sync_to_viewer(model, data)
    np.testing.assert_allclose(data.mocap_pos, [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(data.mocap_quat, [[0.0, 1.0, 0.0, 0.0]])


if __name__ == '__main__':
  absltest.main()
