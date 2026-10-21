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

"""Tests for Sim config and helper utilities."""

from absl.testing import absltest
import mujoco
from reaf.simulation.torch._src.builders import config


class ConfigTest(absltest.TestCase):

  def test_apply_sim_config(self):
    mj_spec = mujoco.MjSpec()
    sim = config.Sim(
        iterations=12,
        ls_iterations=18,
        impratio=2.0,
        solver=mujoco.mjtSolver.mjSOL_CG,
        jacobian=mujoco.mjtJacobian.mjJAC_SPARSE,
        cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
    )
    result = config.apply_sim_config(mj_spec, sim)
    self.assertIs(result, mj_spec)
    self.assertEqual(mj_spec.option.iterations, 12)
    self.assertEqual(mj_spec.option.ls_iterations, 18)
    self.assertAlmostEqual(mj_spec.option.impratio, 2.0)
    self.assertEqual(mj_spec.option.solver, mujoco.mjtSolver.mjSOL_CG)
    self.assertEqual(mj_spec.option.jacobian, mujoco.mjtJacobian.mjJAC_SPARSE)
    self.assertEqual(mj_spec.option.cone, mujoco.mjtCone.mjCONE_ELLIPTIC)

  def test_apply_sim_config_preserves_unspecified_options(self):
    mj_spec = mujoco.MjSpec()
    mj_spec.option.iterations = 77
    mj_spec.option.impratio = 3.3
    sim = config.Sim()
    result = config.apply_sim_config(mj_spec, sim)
    self.assertIs(result, mj_spec)
    self.assertEqual(mj_spec.option.iterations, 77)
    self.assertAlmostEqual(mj_spec.option.impratio, 3.3)


if __name__ == "__main__":
  absltest.main()
