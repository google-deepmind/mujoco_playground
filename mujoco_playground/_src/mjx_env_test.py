# Copyright 2026 DeepMind Technologies Limited
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

import jax.numpy as jp
import mujoco
import numpy as np

from mujoco_playground._src import mjx_env


def _model():
  return mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <body>
            <joint name="hinge" type="hinge"/>
            <geom type="capsule" size="0.02 0.1"/>
          </body>
        </worldbody>
        <actuator><motor joint="hinge"/></actuator>
      </mujoco>
      """
  )


def test_step_can_return_substeps():
  mj_model = _model()
  model = mjx_env.put_model(mj_model)
  data = mjx_env.make_data(mj_model)
  action = jp.array([0.5])

  final_data, substeps = mjx_env.step(
      model, data, action, n_substeps=3, return_substeps=True
  )
  default_final = mjx_env.step(model, data, action, n_substeps=3)

  assert substeps.time.shape == (3,)
  np.testing.assert_allclose(substeps.time, jp.array([0.01, 0.02, 0.03]))
  np.testing.assert_allclose(final_data.qpos, substeps.qpos[-1])
  np.testing.assert_allclose(default_final.qpos, final_data.qpos)
