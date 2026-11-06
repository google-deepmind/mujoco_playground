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

"""Reading back the episodes written by `logger.py`."""

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from etils import epath

# The topics an episode logger writes one message per control step to.
TIMESTEP_TOPIC = "/timestep"
ACTION_TOPIC = "/action"
POLICY_EXTRA_TOPIC = "/policy_extra"
STEP_TOPICS = (TIMESTEP_TOPIC, ACTION_TOPIC, POLICY_EXTRA_TOPIC)


def iter_features(
    path: epath.Path, topics: Sequence[str] = STEP_TOPICS
) -> Iterator[tuple[str, Mapping[str, Any]]]:
  """Yields `(topic, features)` for the episode's logged messages."""
  # Reader-only dependencies; see the README.
  from mcap import reader as mcap_reader  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top
  from tensorflow.core.example import example_pb2  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  with open(path, "rb") as fh:
    for _, channel, message in mcap_reader.make_reader(fh).iter_messages():
      if channel.topic in topics:
        example = example_pb2.Example.FromString(message.data)
        yield channel.topic, example.features.feature
