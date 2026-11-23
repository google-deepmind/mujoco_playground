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

"""Episodic MCAP logger (4 JSON topics: step topics aligned + `/session`)."""

import base64
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent import futures
import datetime
import io
import json
import time
from typing import Any
import uuid

import dm_env
from dm_env import specs
from etils import epath
from gdm_robotics.interfaces import types as gdmr_types
from mujoco_playground._src.real.common import log_reader
from mujoco_playground._src.real.common.instruction_feature import INSTRUCTION_KEY
import numpy as np

# JPEG compression quality (0-100) for camera frames.
_JPEG_QUALITY = 80


def _logged_spec(spec: Any) -> specs.Array:
  shape = spec.shape[1:] if spec.shape and spec.shape[0] == 1 else spec.shape
  if isinstance(spec, specs.StringArray):
    return specs.StringArray(shape=shape, name=spec.name)
  dtype = np.uint8 if spec.dtype in (bool, np.bool_) else spec.dtype
  return specs.Array(shape=shape, dtype=dtype, name=spec.name)


def _conform(value: Any, spec: Any) -> Any:
  if value is None:
    return None
  if isinstance(spec, specs.StringArray):
    return str(value)
  if isinstance(value, Mapping):
    return {k: _conform(v, spec[k]) for k, v in value.items()}
  if hasattr(value, "detach") and hasattr(value, "cpu"):
    value = value.detach().cpu().numpy()
  arr = np.asarray(value)
  if arr.shape == (1,) + spec.shape:
    arr = arr[0]
  return arr.astype(spec.dtype, order="C", copy=True)


def _encode_jpeg(image: np.ndarray) -> bytes:
  from PIL import Image  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  buf = io.BytesIO()
  arr = image[:, :, 0] if image.shape[2] == 1 else image
  Image.fromarray(arr).save(buf, format="JPEG", quality=_JPEG_QUALITY)
  return buf.getvalue()


def _encode(val: Any, *, is_image: bool = False) -> Any:
  if isinstance(val, str):
    return val
  arr = np.asarray(val)
  if (
      is_image
      and arr.ndim == 3
      and arr.dtype == np.uint8
      and arr.shape[2] in (1, 3)
  ):
    jpeg = _encode_jpeg(arr)
    return {log_reader.BYTES_TAG: base64.b64encode(jpeg).decode("ascii")}
  return arr.item() if arr.ndim == 0 else arr.tolist()


class _EpisodicLogger:
  """Buffered episodic MCAP logger with non-blocking background writes."""

  def __init__(
      self,
      *,
      timestep_spec: gdmr_types.TimeStepSpec,
      action_spec: Mapping[str, specs.Array],
      policy_extra_spec: Mapping[str, specs.Array],
      image_observation_keys: Sequence[str],
      timestamp_key: str,
      task_id: str,
      agent_id: str,
      output_directory: epath.PathLike,
      control_timestep_seconds: float,
      outcome: Callable[[], str | None] | None = None,
  ):
    try:
      from mcap import writer as _  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top,unused-import
    except ImportError as exc:
      raise ImportError(
          "mcap is required for MCAP logging. Run: uv sync --extra real"
      ) from exc

    self._out_dir = epath.Path(output_directory)
    self._out_dir.mkdir(parents=True, exist_ok=True)
    assert isinstance(timestep_spec.observation, Mapping)
    self._obs_spec = {
        k: _logged_spec(s) for k, s in timestep_spec.observation.items()
    }
    self._obs_spec.setdefault(INSTRUCTION_KEY, specs.StringArray(shape=()))
    self._obs_spec.setdefault(timestamp_key, specs.Array((), np.int64))
    self._reward_spec = _logged_spec(timestep_spec.reward)
    self._discount_spec = _logged_spec(timestep_spec.discount)
    self._act_spec = {k: _logged_spec(s) for k, s in action_spec.items()}
    self._extra_spec = {
        k: _logged_spec(s) for k, s in policy_extra_spec.items()
    }
    self._img_keys = frozenset(image_observation_keys)
    self._ts_key = timestamp_key
    self._task_id = task_id
    self._agent_id = agent_id
    self._dt = control_timestep_seconds
    self._outcome = outcome or (lambda: None)
    self._steps: list[tuple[int, dm_env.TimeStep]] = []
    self._actions: list[dict[str, Any]] = []
    self._extras: list[dict[str, Any]] = []
    self._stopped = False
    self._pool = futures.ThreadPoolExecutor(max_workers=1)
    self._pending: list[futures.Future[None]] = []

  def _clear(self) -> None:
    self._steps, self._actions, self._extras = [], [], []

  def _conform_step(self, ts: dm_env.TimeStep) -> tuple[int, dm_env.TimeStep]:
    obs = dict(ts.observation)
    obs.setdefault(INSTRUCTION_KEY, self._task_id)
    obs.setdefault(self._ts_key, time.time_ns())
    if missing := self._obs_spec.keys() - obs.keys():
      raise ValueError(
          f"Observation missing {sorted(missing)}; got {sorted(obs)}."
      )
    conformed = {k: _conform(obs[k], s) for k, s in self._obs_spec.items()}
    ts_ns = int(conformed[self._ts_key])
    conformed[self._ts_key] = ts_ns
    return ts_ns, ts._replace(
        reward=_conform(ts.reward, self._reward_spec),
        discount=_conform(ts.discount, self._discount_spec),
        observation=conformed,
    )

  def reset(self, timestep: dm_env.TimeStep) -> None:
    if self._stopped:
      return
    if self._steps:  # Flush an unwritten prior episode first.
      self.write()
    self._steps.append(self._conform_step(timestep))

  def record_action_and_next_timestep(
      self,
      action: Any,
      next_timestep: dm_env.TimeStep,
      policy_extra: Any = None,
  ) -> None:
    if self._stopped or not self._steps:
      return
    act = _conform(action, self._act_spec)
    extra = _conform(policy_extra or {}, self._extra_spec)
    step = self._conform_step(next_timestep)
    self._actions.append(act)
    self._extras.append(extra)
    self._steps.append(step)

  def write(self) -> None:
    if self._stopped or len(self._steps) <= 1:  # Skip <=1-step episodes.
      self._clear()
      return
    steps, actions, extras = self._steps, self._actions, self._extras
    self._clear()
    outcome = self._outcome() or "unlabeled"
    self._pending.append(
        self._pool.submit(self._flush, steps, actions, extras, outcome)
    )

  def _flush(self, steps, actions, extras, outcome: str) -> None:
    from mcap import writer as mcap_writer  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

    now = datetime.datetime.now(datetime.timezone.utc)
    day_dir = self._out_dir / now.strftime("%Y/%m/%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    out_path = day_dir / f"episode_{uuid.uuid4().hex}.mcap"

    def meta(g: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
      return {
          k: {"shape": list(s.shape), "dtype": str(s.dtype)}
          for k, s in g.items()
      }

    session = {
        "task_id": self._task_id,
        "agent_id": self._agent_id,
        "control_timestep_seconds": self._dt,
        "outcome": outcome,
        "feature_specs": {
            "observation": meta(self._obs_spec),
            "action": meta(self._act_spec),
            "policy_extra": meta(self._extra_spec),
        },
    }
    # Pad final action/extra so all three step topics have N aligned messages.
    actions, extras = actions + [actions[-1]], extras + [extras[-1]]

    with open(out_path, "wb") as fh:
      w = mcap_writer.Writer(fh)
      w.start()
      sid = w.register_schema(name="json", encoding="jsonschema", data=b"{}")
      ch = {
          t: w.register_channel(topic=t, message_encoding="json", schema_id=sid)
          for t in (log_reader.SESSION_TOPIC, *log_reader.STEP_TOPICS)
      }

      def emit(topic: str, ts_ns: int, payload: dict[str, Any]) -> None:
        w.add_message(
            channel_id=ch[topic],
            log_time=ts_ns,
            publish_time=ts_ns,
            data=json.dumps(payload).encode("utf-8"),
        )

      for (ts_ns, step), act, extra in zip(steps, actions, extras, strict=True):
        obs_msg: dict[str, Any] = {
            "step_type": int(step.step_type),
            **{
                f"observation/{k}": _encode(v, is_image=k in self._img_keys)
                for k, v in step.observation.items()
            },
        }
        if step.reward is not None:
          obs_msg["reward"] = _encode(step.reward)
        if step.discount is not None:
          obs_msg["discount"] = _encode(step.discount)
        emit(log_reader.TIMESTEP_TOPIC, ts_ns, obs_msg)
        emit(
            log_reader.ACTION_TOPIC,
            ts_ns,
            {f"action/{k}": _encode(v) for k, v in act.items()},
        )
        emit(
            log_reader.POLICY_EXTRA_TOPIC,
            ts_ns,
            {f"extra/policy_extra/{k}": _encode(v) for k, v in extra.items()},
        )
      emit(log_reader.SESSION_TOPIC, steps[-1][0], session)
      w.finish()

  def stop(self) -> None:
    # Discard unwritten buffer (callers invoke write()) and drain pending I/O.
    self._stopped = True
    self._clear()
    self._pool.shutdown(wait=True)
    pending, self._pending = self._pending, []
    for fut in pending:
      fut.result()

  def close(self) -> None:
    self.stop()


create_logger = _EpisodicLogger
