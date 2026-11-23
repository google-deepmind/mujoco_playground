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

"""Tests for episodic MCAP logging."""

import base64
import collections
import io
import json
import pathlib
import sys
import tempfile
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
import dm_env
from dm_env import specs
from etils import epath
from gdm_robotics.interfaces import environment as gdmr_env
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
from gdm_robotics.runtime import runloop as runloop_lib
from mujoco_playground._src.real.common import log_reader
from mujoco_playground._src.real.common import logger
from mujoco_playground._src.real.common import view_log
from mujoco_playground.robots.so101.real import constant as so101
from mujoco_playground.tasks.sysid.so101 import data_loader
import numpy as np
from PIL import Image
import torch

try:
  from mcap import writer as _  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top,unused-import

  _HAS_MCAP = True
except ImportError:
  _HAS_MCAP = False

_TS = "timestamp"


def _make_fake_mcap() -> dict[str, Any]:
  """Minimal file-backed mcap stub used when the mcap wheel is absent."""

  class _Writer:

    def __init__(self, fh):
      self._fh, self._topics, self._recs = fh, {}, []

    def start(self):
      pass

    def register_schema(self, **_kw):
      return 1

    def register_channel(self, *, topic, **_kw):
      cid = len(self._topics) + 1
      self._topics[cid] = topic
      return cid

    def add_message(self, *, channel_id, log_time, publish_time, data):
      del publish_time
      self._recs.append((
          self._topics[channel_id],
          log_time,
          base64.b64encode(data).decode("ascii"),
      ))

    def finish(self):
      self._fh.write(json.dumps(self._recs).encode("utf-8"))

  class _Reader:

    def __init__(self, fh):
      self._recs = json.loads(fh.read().decode("utf-8"))

    def iter_messages(self, topics=None):
      allowed = set(topics) if topics is not None else None
      for topic, lt, b64 in self._recs:
        if allowed is None or topic in allowed:
          yield (
              None,
              types.SimpleNamespace(topic=topic),
              types.SimpleNamespace(log_time=lt, data=base64.b64decode(b64)),
          )

  w = types.SimpleNamespace(Writer=_Writer)
  r = types.SimpleNamespace(make_reader=_Reader)
  return {
      "mcap": types.SimpleNamespace(writer=w, reader=r),
      "mcap.writer": w,
      "mcap.reader": r,
  }


def _ts_spec(**obs) -> gdmr_types.TimeStepSpec:
  return gdmr_types.TimeStepSpec(
      step_type=specs.BoundedArray((), np.int32, 0, 2),
      reward=specs.Array((), np.float32),
      discount=specs.BoundedArray((), np.float32, 0.0, 1.0),
      observation=obs,
  )


def _mcaps(d: str) -> list[pathlib.Path]:
  return sorted(pathlib.Path(d).glob("**/*.mcap"))


class _McapTestCase(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if not _HAS_MCAP:
      p = mock.patch.dict(sys.modules, _make_fake_mcap())
      p.start()
      self.addCleanup(p.stop)

  def _make_logger(self, out_dir: str, spec, **kw):
    return logger.create_logger(
        timestep_spec=spec,
        action_spec=kw.pop("action_spec", {"act": specs.Array((), np.float32)}),
        policy_extra_spec=kw.pop("policy_extra_spec", {}),
        image_observation_keys=kw.pop("image_keys", []),
        timestamp_key=kw.pop("timestamp_key", _TS),
        task_id=kw.pop("task_id", "test_task"),
        agent_id="test_agent",
        output_directory=out_dir,
        control_timestep_seconds=0.02,
        outcome=kw.pop("outcome", None),
    )


class LoggedSpecTest(absltest.TestCase):

  def test_logged_spec(self):
    self.assertEqual(logger._logged_spec(specs.Array((1, 6), np.float32)).shape, (6,))  # pylint: disable=protected-access
    self.assertEqual(logger._logged_spec(specs.Array((16, 16, 3), np.uint8)).shape, (16, 16, 3))  # pylint: disable=protected-access
    self.assertEqual(logger._logged_spec(specs.Array((2,), bool)).dtype, np.uint8)  # pylint: disable=protected-access


class LoggerTest(_McapTestCase):

  def test_parity_round_trip(self):
    tmp = self.enterContext(tempfile.TemporaryDirectory())
    jk = tuple(so101.JOINT_POSITION_KEYS.values())
    gk = tuple(so101.GOAL_POSITION_KEYS.values())
    rgb = torch.full((1, 16, 12, 3), 100, dtype=torch.uint8)
    gray = np.full((16, 12, 1), 128, dtype=np.uint8)
    q0_buf, a0_buf = np.array(0.1, np.float32), np.array(0.2, np.float32)

    def obs(ts, cmd, q):
      return {
          so101.TIMESTAMP: np.int64(ts),
          so101.FOLLOWER_COMMAND_TIMESTAMP: np.int64(cmd),
          "rgb": rgb,
          "gray": gray,
          **{k: q for k in jk},
      }

    def act(c):
      return {k: c for k in gk}

    ep = self._make_logger(
        tmp,
        _ts_spec(**{
            so101.TIMESTAMP: specs.Array((), np.int64),
            so101.FOLLOWER_COMMAND_TIMESTAMP: specs.Array((), np.int64),
            "rgb": specs.Array((1, 16, 12, 3), np.uint8),
            "gray": specs.Array((16, 12, 1), np.uint8),
            **{k: specs.Array((), np.float32) for k in jk},
        }),
        action_spec={k: specs.Array((), np.float32) for k in gk},
        policy_extra_spec={so101.LEADER_TIMESTAMP: specs.Array((), np.int64)},
        image_keys=["rgb", "gray"],
        timestamp_key=so101.TIMESTAMP,
        task_id="so101_parity",
        outcome=lambda: "success",
    )
    with mock.patch.object(logger, "_encode", wraps=logger._encode) as spy:  # pylint: disable=protected-access
      ep.reset(dm_env.restart(obs(1_000_000_000, 1_000_000_000, q0_buf)))
      ep.record_action_and_next_timestep(
          act(a0_buf),
          dm_env.transition(0.0, obs(1_020_000_000, 1_024_000_000, 0.2)),
          {so101.LEADER_TIMESTAMP: np.int64(1_001_000_000)},
      )
      ep.record_action_and_next_timestep(
          act(0.3),
          dm_env.termination(
              torch.tensor([1.0]), obs(1_040_000_000, 1_044_000_000, 0.3)
          ),
          {so101.LEADER_TIMESTAMP: np.int64(1_021_000_000)},
      )
      self.assertEqual(spy.call_count, 0)  # Deferred encoding.
      q0_buf[...], a0_buf[...] = 9.0, 8.0  # Mutate caller buffers.
      ep.write()
      ep.close()
      self.assertGreater(spy.call_count, 0)

    path = epath.Path(_mcaps(tmp)[0])
    from mcap import reader as mcap_reader  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

    with open(path, "rb") as fh:
      msgs = [
          (ch.topic, m.log_time, log_reader.decode_payload(m.data))
          for _, ch, m in mcap_reader.make_reader(fh).iter_messages()
      ]
    self.assertEqual([t for _, t, _ in msgs], sorted(t for _, t, _ in msgs))
    by_t = collections.defaultdict(list)
    for topic, ts, payload in msgs:
      by_t[topic].append((ts, payload))

    expected_ts = [1_000_000_000, 1_020_000_000, 1_040_000_000]
    for topic in log_reader.STEP_TOPICS:
      self.assertEqual([ts for ts, _ in by_t[topic]], expected_ts)
    self.assertEqual(by_t[log_reader.SESSION_TOPIC][0][0], expected_ts[-1])
    self.assertEqual(by_t[log_reader.SESSION_TOPIC][0][1]["outcome"], "success")
    np.testing.assert_allclose(
        [p[f"action/{gk[0]}"] for _, p in by_t[log_reader.ACTION_TOPIC]],
        [0.2, 0.3, 0.3],
        atol=1e-5,
    )
    self.assertEqual(
        [
            p[f"extra/policy_extra/{so101.LEADER_TIMESTAMP}"]
            for _, p in by_t[log_reader.POLICY_EXTRA_TOPIC]
        ],
        [1_001_000_000, 1_021_000_000, 1_021_000_000],
    )

    app_id, shapes = view_log.session_info(path)
    self.assertEqual(
        (app_id, shapes["observation/rgb"], shapes["observation/gray"]),
        ("so101_parity", (16, 12, 3), (16, 12, 1)),
    )
    ts0 = by_t[log_reader.TIMESTEP_TOPIC][0][1]
    for key, mode in (("observation/rgb", "RGB"), ("observation/gray", "L")):
      raw = ts0[key]
      self.assertTrue(
          isinstance(raw, bytes) and raw.startswith(b"\xff\xd8\xff")
      )
      self.assertEqual(view_log.view_type(raw, ()), "image")
      img = Image.open(io.BytesIO(raw))
      self.assertEqual((img.size, img.mode), ((12, 16), mode))

    traj = data_loader.load_trajectory(path)
    self.assertEqual(traj.measured_qpos.shape, (2, 6))
    np.testing.assert_allclose(traj.timestamps, [0.0, 0.02], atol=1e-6)
    np.testing.assert_allclose(traj.ctrl_timestamps, [0.024, 0.044], atol=1e-6)
    np.testing.assert_allclose(traj.measured_qpos[:, 0], [0.1, 0.2], atol=1e-5)
    np.testing.assert_allclose(traj.commanded_ctrl[:, 0], [0.2, 0.3], atol=1e-5)

  def test_lifecycle(self):
    tmp = self.enterContext(tempfile.TemporaryDirectory())
    spec = _ts_spec(qpos=specs.Array((), np.float32))
    ep = self._make_logger(tmp, spec)
    with self.assertRaisesRegex(ValueError, "qpos"):
      ep.reset(dm_env.restart({}))

    # 1-step skip, reset() flush, and stop() discard + guard.
    ep.reset(dm_env.restart({_TS: 10, "qpos": 0.0}))
    ep.write()
    for t0, a in ((20, 1.0), (40, 2.0)):
      ep.reset(dm_env.restart({_TS: t0, "qpos": 0.0}))
      ep.record_action_and_next_timestep(
          {"act": a}, dm_env.termination(0.0, {_TS: t0 + 10, "qpos": a})
      )
    ep.write()
    ep.reset(dm_env.restart({_TS: 60, "qpos": 0.0}))
    ep.record_action_and_next_timestep(
        {"act": 3.0}, dm_env.transition(0.0, {_TS: 70, "qpos": 3.0})
    )
    ep.stop()
    ep.record_action_and_next_timestep(
        {"act": 4.0}, dm_env.termination(0.0, {_TS: 80, "qpos": 4.0})
    )
    ep.write()
    self.assertLen(_mcaps(tmp), 2)

    # Background write error surfaces on close().
    ep_err = self._make_logger(tmp, spec)
    with mock.patch.object(ep_err, "_flush", side_effect=OSError("disk full")):
      ep_err.reset(dm_env.restart({_TS: 1, "qpos": 0.0}))
      ep_err.record_action_and_next_timestep(
          {"act": 0.0}, dm_env.termination(0.0, {_TS: 2, "qpos": 0.0})
      )
      ep_err.write()
      with self.assertRaisesRegex(OSError, "disk full"):
        ep_err.close()


class RunloopTest(_McapTestCase):

  class _Env(gdmr_env.Environment):

    def __init__(self):
      self._steps = 0

    def default_reset_options(self):
      return None

    def reset_with_options(self, *, options=None):
      del options
      self._steps = 0
      return dm_env.restart(
          {"cam": torch.full((1, 16, 16, 3), 50, dtype=torch.uint8)}
      )

    def step(self, action):
      del action
      self._steps += 1
      fn = dm_env.termination if self._steps >= 3 else dm_env.transition
      return fn(
          torch.tensor(float(self._steps >= 3)),
          {"cam": torch.full((1, 16, 16, 3), 50, dtype=torch.uint8)},
      )

    def timestep_spec(self):
      return _ts_spec(cam=specs.Array((1, 16, 16, 3), np.uint8))

    def action_spec(self):
      return {"act": specs.BoundedArray((1, 6), np.float32, -1.0, 1.0)}

  class _Policy(gdmr_policy.Policy):

    def initial_state(self):
      return ()

    def step_spec(self, timestep_spec):
      del timestep_spec
      return (
          {"act": specs.BoundedArray((1, 6), np.float32, -1.0, 1.0)},
          {},
      ), ()

    def step(self, timestep, prev_state=()):
      del timestep, prev_state
      return ({"act": torch.zeros((1, 6))}, {}), ()

  def test_records_an_episode_end_to_end(self):
    env, tmp = self._Env(), self.enterContext(tempfile.TemporaryDirectory())
    ep = self._make_logger(
        tmp,
        env.timestep_spec(),
        action_spec=env.action_spec(),
        image_keys=["cam"],
    )
    runloop_lib.Runloop(
        environment=env, policy=self._Policy(), loggers=[ep]
    ).run(num_episodes=1)
    ep.close()
    self.assertLen(_mcaps(tmp), 1)


if __name__ == "__main__":
  absltest.main()
