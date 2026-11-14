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

from collections.abc import Callable
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


# `iter_features` yields maps from decoded messages. With protobuf's upb backend,
# such a map keeps neither its message nor the descriptor pool that the decoder
# built, and reading it after both are freed crashes the process. So decoders,
# and with them their pools, are never evicted. They are keyed by the schema
# itself, because schema ids are only unique within a file.
_DECODERS: dict[tuple[str, str, str, bytes], Callable[[bytes], Any]] = {}


def _build_decoder(
    message_encoding: str,
    schema_name: str,
    schema_encoding: str,
    schema_data: bytes,
) -> Callable[[bytes], Any]:
  """Builds a decoder for messages with this schema."""
  from mcap import records as mcap_records  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top
  from mcap_protobuf import decoder as mcap_decoder  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  schema = mcap_records.Schema(
      id=0, name=schema_name, encoding=schema_encoding, data=schema_data
  )
  decoder = mcap_decoder.DecoderFactory().decoder_for(message_encoding, schema)
  if decoder is None:
    raise ValueError(
        f"Cannot decode {schema_name} messages encoded as {message_encoding!r}"
        f" with a {schema_encoding!r} schema."
    )
  return decoder


def _decoder(
    message_encoding: str,
    schema_name: str,
    schema_encoding: str,
    schema_data: bytes,
) -> Callable[[bytes], Any]:
  """The decoder for this schema, the same one in every thread."""
  key = (message_encoding, schema_name, schema_encoding, schema_data)
  decoder = _DECODERS.get(key)
  if decoder is None:
    # Two threads may both build one; every reader uses the one stored first.
    decoder = _DECODERS.setdefault(key, _build_decoder(*key))
  return decoder


def iter_messages(
    path: epath.Path, topics: Sequence[str]
) -> Iterator[tuple[str, Any]]:
  """Yields `(topic, message)` for the episode's messages on `topics`."""
  # The messages are decoded with the schema stored in the file, so TensorFlow
  # never loads (it clashes with Triton).
  from mcap import reader as mcap_reader  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  with open(path, "rb") as fh:
    reader = mcap_reader.make_reader(fh)
    for schema, channel, message in reader.iter_messages(topics=topics):
      if schema is None:
        raise ValueError(f"{path} has no schema for {channel.topic}.")
      decoder = _decoder(
          channel.message_encoding,
          schema.name,
          schema.encoding,
          bytes(schema.data),
      )
      yield channel.topic, decoder(message.data)


def iter_features(
    path: epath.Path, topics: Sequence[str] = STEP_TOPICS
) -> Iterator[tuple[str, Mapping[str, Any]]]:
  """Yields `(topic, features)` for the episode's logged messages."""
  for topic, example in iter_messages(path, topics):
    yield topic, example.features.feature
