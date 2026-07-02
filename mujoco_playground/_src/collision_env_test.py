"""Tests for collision behavior in Playground environments."""

import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import smooth
from mujoco_playground import registry
import numpy as np


ENV_NAMES = ('HopperHop',)
ROLLOUT_MODES = (
    (False, 'hard'),
    (True, 'hard'),
    (True, 'c2'),
)
SOFT_MODES = ('hard', 'c2')
NUM_STEPS = 2


def _require_soft_collision_support() -> None:
  model = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <body pos="0 0 0.2">
            <freejoint/>
            <geom type="sphere" size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
      """)
  mx = mjx.put_model(model)
  if not hasattr(mx.opt, 'col_soft_enable') or not hasattr(
      mx.opt, 'softjax_mode'
  ):
    raise AssertionError(
        'Vendored diff-mjx is not active. Expected mjx option fields '
        '`col_soft_enable` and `softjax_mode`.'
    )


def _clone_env(env_name: str):
  return registry.load(env_name, config_overrides={'impl': 'jax'})


def _configured_model(env, col_soft_enable: bool, softjax_mode: str):
  return env.mjx_model.replace(
      opt=env.mjx_model.opt.replace(
          col_soft_enable=col_soft_enable,
          softjax_mode=softjax_mode,
      )
  )


def _configured_env(env_name: str, col_soft_enable: bool, softjax_mode: str):
  env = _clone_env(env_name)
  env._mjx_model = _configured_model(env, col_soft_enable, softjax_mode)
  return env


def _contact_dist(contact_data):
  return contact_data._impl.contact.dist


def _contact_loss(env, data, qpos):
  dx = data.replace(qpos=qpos)
  dx = smooth.kinematics(env.mjx_model, dx)
  dx = smooth.com_pos(env.mjx_model, dx)
  dx = collision_driver.collision(env.mjx_model, dx)
  return jp.sum(_contact_dist(dx))


def _rollout_states(env, seed: int = 0):
  state = env.reset(jax.random.PRNGKey(seed))
  action = jp.zeros(env.action_size, dtype=state.data.qpos.dtype)
  states = []
  for _ in range(NUM_STEPS):
    state = env.step(state, action)
    states.append(state)
  return states


def _rollout(env, seed: int = 0):
  states = _rollout_states(env, seed=seed)
  final_state = states[-1]
  rewards = np.asarray([np.asarray(state.reward) for state in states])
  return final_state, rewards


def _gradient_metrics(env_name: str, col_soft_enable: bool, softjax_mode: str):
  env = _configured_env(env_name, col_soft_enable, softjax_mode)
  states = _rollout_states(env)
  grad_fn = jax.jit(
      jax.grad(lambda qpos, data: _contact_loss(env, data, qpos), argnums=0)
  )
  grads = []
  finite_by_step = []
  zero_by_step = []
  norms = []

  for state in states:
    data = state.data
    grad = np.asarray(grad_fn(data.qpos, data))
    finite = bool(np.isfinite(grad).all())
    zero = bool(np.allclose(grad, 0.0)) if finite else False
    norm = float(
        np.linalg.norm(np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0))
    )
    grads.append(grad)
    finite_by_step.append(finite)
    zero_by_step.append(zero)
    norms.append(norm)

  return {
      'grads': grads,
      'finite': all(finite_by_step),
      'zero': all(zero_by_step) if finite_by_step else False,
      'any_nonzero': any(not is_zero for is_zero in zero_by_step)
      if zero_by_step
      else False,
      'norm': max(norms, default=0.0),
      'norms': norms,
      'finite_by_step': finite_by_step,
      'zero_by_step': zero_by_step,
      'mode': softjax_mode,
      'col_soft_enable': col_soft_enable,
  }


def _quality_rank(metrics):
  return (int(metrics['finite']), int(metrics['any_nonzero']))


class CollisionEnvTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': (
              f'test_{env_name}_{softjax_mode}_'
              f'{"soft" if col_soft_enable else "hard"}'
          ),
          'env_name': env_name,
          'col_soft_enable': col_soft_enable,
          'softjax_mode': softjax_mode,
      }
      for env_name in ENV_NAMES
      for col_soft_enable, softjax_mode in ROLLOUT_MODES
  )
  def test_collision_rollouts_are_finite(
      self, env_name: str, col_soft_enable: bool, softjax_mode: str
  ) -> None:
    _require_soft_collision_support()
    env = _configured_env(env_name, col_soft_enable, softjax_mode)
    state, rewards = _rollout(env)

    self.assertTrue(np.isfinite(np.asarray(state.data.qpos)).all())
    self.assertTrue(np.isfinite(np.asarray(state.data.qvel)).all())
    self.assertTrue(np.isfinite(rewards).all())

  @parameterized.named_parameters(
      {
          'testcase_name': f'test_{env_name}_{softjax_mode}',
          'env_name': env_name,
          'softjax_mode': softjax_mode,
      }
      for env_name in ENV_NAMES
      for softjax_mode in SOFT_MODES
  )
  def test_soft_collision_gradients_are_finite(
      self, env_name: str, softjax_mode: str
  ) -> None:
    _require_soft_collision_support()
    metrics = _gradient_metrics(env_name, True, softjax_mode)

    self.assertTrue(
        metrics['finite'],
        msg=(
            f'{env_name} soft collision gradient is not finite for '
            f'softjax_mode={softjax_mode}: norms={metrics["norms"]}, '
            f'finite_by_step={metrics["finite_by_step"]}'
        ),
    )
    self.assertTrue(
        metrics['any_nonzero'],
        msg=(
            f'{env_name} soft collision gradient unexpectedly vanished for '
            f'softjax_mode={softjax_mode}: norms={metrics["norms"]}, '
            f'zero_by_step={metrics["zero_by_step"]}'
        ),
    )

  @parameterized.named_parameters(
      {'testcase_name': f'test_{env_name}', 'env_name': env_name}
      for env_name in ENV_NAMES
  )
  def test_soft_collision_gradients_improve_or_match_hard_baseline(
      self, env_name: str
  ) -> None:
    _require_soft_collision_support()
    hard_metrics = _gradient_metrics(env_name, False, 'hard')
    soft_metrics = [
        _gradient_metrics(env_name, True, softjax_mode)
        for softjax_mode in SOFT_MODES
    ]

    for metrics in soft_metrics:
      self.assertTrue(
          metrics['finite'],
          msg=(
              f'{env_name} soft collision gradient regressed to non-finite '
              f'for softjax_mode={metrics["mode"]}; hard baseline='
              f'{hard_metrics["finite"]=}, '
              f'{hard_metrics["zero_by_step"]=}, '
              f'{hard_metrics["norms"]=}'
          ),
      )

    if _quality_rank(hard_metrics) < (1, 1):
      self.assertTrue(
          any(
              _quality_rank(metrics) > _quality_rank(hard_metrics)
              for metrics in soft_metrics
          ),
          msg=(
              f'{env_name} hard baseline did not improve under soft '
              f'collision flags; hard={hard_metrics}, soft={soft_metrics}'
          ),
      )
    else:
      self.assertTrue(
          all(
              _quality_rank(metrics) >= _quality_rank(hard_metrics)
              for metrics in soft_metrics
          ),
          msg=(
              f'{env_name} soft collision flags regressed relative to a '
              f'usable hard baseline; hard={hard_metrics}, '
              f'soft={soft_metrics}'
          ),
      )


if __name__ == '__main__':
  absltest.main()
