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

"""Views a logged mcap episode (from logger.py) in rerun.

uv run --extra real python -m mujoco_playground.real.view_log
uv run --extra real python -m mujoco_playground.real.view_log --file=<episode.mcap>
"""

import collections
from collections.abc import Mapping
from collections.abc import Sequence
import glob
from typing import Any

from absl import app
from absl import flags
from etils import epath
from mujoco_playground import paths
from mujoco_playground._src.real.common import log_reader
import numpy as np

_FILE = flags.DEFINE_string(
    "file",
    None,
    "Episode MCAP to view (default: the most recent logged episode).",
)
_SAVE = flags.DEFINE_string(
    "save",
    None,
    "Write a .rrd recording to this path instead of opening the viewer.",
)

_TIMELINE = "wall_clock"
_STEP_TIMELINE = "step"

_IMAGE_NDIM = 3
# Compressed frames arrive as opaque bytes and announce themselves this way.
_IMAGE_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",
}


def latest_episode() -> epath.Path:
  """Returns the most recently written episode, from any session.

  By modification time, not by name: episode files are named after a random
  uuid, so sorting them alphabetically picks an arbitrary one.
  """
  directory = paths.local_dir()
  matches = glob.glob(str(directory / "**/*.mcap"), recursive=True)
  episodes = sorted(
      [epath.Path(p) for p in matches], key=lambda p: p.stat().mtime
  )
  if not episodes:
    raise FileNotFoundError(f"No logged episodes under {directory}.")
  return episodes[-1]


def session_info(path: epath.Path) -> tuple[str, Mapping[str, Sequence[int]]]:
  """Returns the task/agent name and the shape of every feature in the episode."""
  # Viewer-only dependencies.
  try:
    from safari_sdk.logging.python import constants  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top
    from safari_sdk.logging.python import mcap_parser_utils  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

    session = mcap_parser_utils.read_session_proto_data(
        str(path), constants.SESSION_TOPIC_NAME
    )[0]
    specs = session.policy_environment_metadata.feature_specs
    shapes = {}
    for prefix, group in (
        ("observation", specs.observation),
        ("action", specs.action),
        ("extra/policy_extra", specs.policy_extra_output),
    ):
      for key, spec in group.items():
        shapes[key] = tuple(spec.shape)
        shapes[f"{prefix}/{key}"] = tuple(spec.shape)
        if key.startswith(f"{prefix}/"):
          shapes[key.removeprefix(f"{prefix}/")] = tuple(spec.shape)
    app_id = getattr(session, "task_id", None) or "episode_log"
    return app_id, shapes
  except Exception:  # pylint: disable=broad-except
    return "episode_log", {}


def view_type(feature: Any, shape: Sequence[int]) -> str:
  """How a feature asks to be shown, from its shape and its contents."""
  if len(shape) == _IMAGE_NDIM:
    return "image"
  if feature.WhichOneof("kind") == "bytes_list":
    if feature.bytes_list.value and any(
        feature.bytes_list.value[0].startswith(m) for m in _IMAGE_MAGIC
    ):
      return "image"
    return "text"
  return "plot"


def newest_frame(values: Sequence[float], shape: Sequence[int]) -> np.ndarray:
  """The most recent RGB frame of an image or stacked image observation."""
  if len(shape) != _IMAGE_NDIM:
    return np.asarray(values)

  # (C, H, W) where leading channels are the newest RGB frame.
  if (
      shape[0] == 3
      or (shape[0] % 3 == 0 and shape[0] > 3)
      or (shape[0] in (1, 4) and shape[2] not in (1, 3, 4))
  ):
    channels, height, width = shape
    frame = np.asarray(values, np.float32).reshape(channels, height, width)[:3]
    if frame.max() <= 1.0 and frame.min() >= 0.0 and frame.dtype != np.uint8:
      frame = frame * 255.0
    return (frame.transpose(1, 2, 0)).clip(0.0, 255.0).astype(np.uint8)

  # Otherwise assume (H, W, C).
  h, w, c = shape
  frame = np.asarray(values, np.float32).reshape(h, w, c)
  if frame.max() <= 1.0 and frame.min() >= 0.0 and frame.dtype != np.uint8:
    frame = frame * 255.0
  return frame.clip(0.0, 255.0).astype(np.uint8)


def log_feature(
    entity: str, feature: Any, shape: Sequence[int], view_type_str: str
) -> None:
  """Shows one logged feature under `entity`."""
  import rerun as rr  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  if view_type_str == "image":
    if feature.WhichOneof("kind") == "bytes_list":
      (raw,) = feature.bytes_list.value
      media_type = next(
          (t for m, t in _IMAGE_MAGIC.items() if raw.startswith(m)),
          "image/jpeg",
      )
      rr.log(entity, rr.EncodedImage(contents=raw, media_type=media_type))
    else:
      kind = feature.WhichOneof("kind")
      values = getattr(feature, kind).value
      rr.log(entity, rr.Image(newest_frame(values, shape)))
  elif view_type_str == "text":
    (raw,) = feature.bytes_list.value
    try:
      rr.log(entity, rr.TextDocument(raw.decode("utf-8")))
    except UnicodeDecodeError:
      rr.log(entity, rr.TextDocument(str(raw)))
  else:
    kind = feature.WhichOneof("kind")
    values = getattr(feature, kind).value
    if len(values) == 1:
      rr.log(entity, rr.Scalars(float(values[0])))
    else:
      for i, value in enumerate(values):
        rr.log(f"{entity}/{i}", rr.Scalars(float(value)))


def blueprint(views: Mapping[str, str]):
  """Plots down the left, images and text down the right."""
  import rerun.blueprint as rrb  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  panels = {
      "plot": rrb.TimeSeriesView,
      "image": rrb.Spatial2DView,
      "text": rrb.TextDocumentView,
  }
  by_type = collections.defaultdict(list)
  for entity, view_type_str in sorted(views.items()):
    if view_type_str in panels:
      by_type[view_type_str].append(
          panels[view_type_str](origin=entity, name=entity)
      )

  columns = []
  if by_type["plot"]:
    columns.append(rrb.Grid(contents=by_type["plot"]))
  right_panels = by_type["image"] + by_type["text"]
  if right_panels:
    columns.append(rrb.Vertical(contents=right_panels))

  if len(columns) == 2:
    return rrb.Blueprint(rrb.Horizontal(*columns, column_shares=[3, 2]))
  elif len(columns) == 1:
    return rrb.Blueprint(columns[0])
  return rrb.Blueprint()


def view(path: epath.Path) -> Mapping[str, str]:
  """Replays the episode into rerun, and reports the views it calls for."""
  import rerun as rr  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  print(f"Loading {path}")
  _, shapes = session_info(path)

  # Find timestamp key if present.
  timestamp_key = None
  for topic, features in log_reader.iter_features(path):
    if topic == log_reader.TIMESTEP_TOPIC:
      for key in features:
        if "timestamp" in key.lower():
          timestamp_key = key
          break
      break

  stamps = []
  if timestamp_key:
    for topic, features in log_reader.iter_features(path):
      if topic == log_reader.TIMESTEP_TOPIC and timestamp_key in features:
        feat = features[timestamp_key]
        if feat.int64_list.value:
          stamps.append(feat.int64_list.value[0])
        elif feat.float_list.value:
          stamps.append(int(feat.float_list.value[0]))

  views: dict[str, str] = {}
  steps = collections.Counter()
  for topic, features in log_reader.iter_features(path):
    step = steps[topic]
    steps[topic] += 1
    rr.set_time(_STEP_TIMELINE, sequence=step)
    if step < len(stamps):
      rr.set_time(_TIMELINE, timestamp=stamps[step] * 1e-9)

    for key, feature in features.items():
      if key == timestamp_key:
        continue  # Already the timeline; a plot of it says nothing.
      shape = shapes.get(key, ())
      v_type = view_type(feature, shape)
      entity = key.lstrip("/")
      log_feature(entity, feature, shape, v_type)
      views[entity] = v_type

  return views


def main(argv: Sequence[str] | None = None) -> None:
  del argv  # Unused.
  import rerun as rr  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  path = epath.Path(_FILE.value) if _FILE.value else latest_episode()
  app_id, _ = session_info(path)
  rr.init(app_id)
  if _SAVE.value:
    rr.save(_SAVE.value)
  else:
    rr.spawn()
  # Sent rather than passed as the sink's `default_blueprint`, which the viewer
  # ignores when it already has a layout stored for this application id.
  rr.send_blueprint(blueprint(view(path)), make_active=True, make_default=True)
  if _SAVE.value:
    print(f"Wrote {_SAVE.value}")


if __name__ == "__main__":
  app.run(main)
