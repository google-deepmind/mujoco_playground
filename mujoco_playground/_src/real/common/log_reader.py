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

"""Reading back the MCAP episodes written by `logger.py`.

Each episode file contains four JSON-encoded topics: `/timestep`, `/action`,
`/policy_extra` (N aligned messages each, with the final action/policy_extra
duplicated), and `/session` (episode metadata and feature specs). Binary/JPEG
values are stored as `{"__bytes__": "<base64>"}` and decoded back to `bytes`.
"""

import base64
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import json
from typing import Any

from etils import epath

# Topics an episode logger writes to.
SESSION_TOPIC = "/session"
TIMESTEP_TOPIC = "/timestep"
ACTION_TOPIC = "/action"
POLICY_EXTRA_TOPIC = "/policy_extra"
STEP_TOPICS = (TIMESTEP_TOPIC, ACTION_TOPIC, POLICY_EXTRA_TOPIC)

BYTES_TAG = "__bytes__"


def _decode_value(value: Any) -> Any:
  if isinstance(value, dict) and list(value.keys()) == [BYTES_TAG]:
    return base64.b64decode(value[BYTES_TAG])
  return value


def decode_payload(data: bytes) -> dict[str, Any]:
  """Decodes a JSON-encoded MCAP message payload into a feature dict."""
  raw = json.loads(data.decode("utf-8"))
  return {k: _decode_value(v) for k, v in raw.items()}


def read_session(path: epath.Path) -> Mapping[str, Any]:
  """Returns the episode's `/session` metadata dict, or `{}` if absent."""
  from mcap import reader as mcap_reader  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  with open(path, "rb") as fh:
    for _, _, message in mcap_reader.make_reader(fh).iter_messages(
        topics=[SESSION_TOPIC]
    ):
      return json.loads(message.data.decode("utf-8"))
  return {}


def iter_features(
    path: epath.Path, topics: Sequence[str] = STEP_TOPICS
) -> Iterator[tuple[str, Mapping[str, Any]]]:
  """Yields `(topic, features)` for the episode's logged messages."""
  from mcap import reader as mcap_reader  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  with open(path, "rb") as fh:
    for _, channel, message in mcap_reader.make_reader(fh).iter_messages(
        topics=topics
    ):
      yield channel.topic, decode_payload(message.data)
