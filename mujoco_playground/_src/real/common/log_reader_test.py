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

"""Tests for reading back MCAP episodes."""

import pathlib
import subprocess
import sys
import tempfile

from absl.testing import absltest
from etils import epath
from google.protobuf import any_pb2
from google.protobuf import api_pb2
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import duration_pb2
from google.protobuf import json_format
from google.protobuf import message_factory
from google.protobuf import source_context_pb2
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
from google.protobuf import type_pb2
from mujoco_playground._src.real.common import log_reader

try:
  from mcap import writer as mcap_writer  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  _HAS_MCAP = True
except ImportError:
  _HAS_MCAP = False


def _files(*modules) -> descriptor_pb2.FileDescriptorSet:
  files = descriptor_pb2.FileDescriptorSet()
  for module in modules:
    module.DESCRIPTOR.CopyToProto(files.file.add())
  return files


def _write_one_message(path, message, *modules, files=None, data=None):
  """Writes `message` on /clock, with a schema made of the modules' files."""
  files = files or _files(*modules)
  with open(path, "wb") as fh:
    writer = mcap_writer.Writer(fh)
    writer.start()
    schema_id = writer.register_schema(
        name=message.DESCRIPTOR.full_name,
        encoding="protobuf",
        data=files.SerializeToString(),
    )
    channel_id = writer.register_channel(
        topic="/clock", message_encoding="protobuf", schema_id=schema_id
    )
    writer.add_message(
        channel_id,
        log_time=0,
        data=data or message.SerializeToString(),
        publish_time=0,
    )
    writer.finish()


def _example_files() -> descriptor_pb2.FileDescriptorSet:
  """A schema shaped like TensorFlow's Example, with float features only."""
  field = descriptor_pb2.FieldDescriptorProto
  file = descriptor_pb2.FileDescriptorProto(
      name="example.proto", package="tensorflow", syntax="proto3"
  )
  file.message_type.add(name="FloatList").field.add(
      name="value", number=1, type=field.TYPE_FLOAT, label=field.LABEL_REPEATED
  )
  file.message_type.add(name="Feature").field.add(
      name="float_list",
      number=2,
      type=field.TYPE_MESSAGE,
      type_name=".tensorflow.FloatList",
      label=field.LABEL_OPTIONAL,
  )
  features = file.message_type.add(name="Features")
  entry = features.nested_type.add(name="FeatureEntry")
  entry.options.map_entry = True
  entry.field.add(
      name="key", number=1, type=field.TYPE_STRING, label=field.LABEL_OPTIONAL
  )
  entry.field.add(
      name="value",
      number=2,
      type=field.TYPE_MESSAGE,
      type_name=".tensorflow.Feature",
      label=field.LABEL_OPTIONAL,
  )
  features.field.add(
      name="feature",
      number=1,
      type=field.TYPE_MESSAGE,
      type_name=".tensorflow.Features.FeatureEntry",
      label=field.LABEL_REPEATED,
  )
  file.message_type.add(name="Example").field.add(
      name="features",
      number=1,
      type=field.TYPE_MESSAGE,
      type_name=".tensorflow.Features",
      label=field.LABEL_OPTIONAL,
  )
  return descriptor_pb2.FileDescriptorSet(file=[file])


def _read(path):
  return [m for _, m in log_reader.iter_messages(epath.Path(path), ["/clock"])]


@absltest.skipUnless(_HAS_MCAP, "mcap is required to read episodes.")
class LogReaderTest(absltest.TestCase):

  def test_reads_files_that_number_their_schemas_differently(self):
    # Both files give their only schema id 1, for different message types.
    with tempfile.TemporaryDirectory() as tmp_dir:
      clock = pathlib.Path(tmp_dir) / "timestamp.mcap"
      timer = pathlib.Path(tmp_dir) / "duration.mcap"
      _write_one_message(
          clock, timestamp_pb2.Timestamp(seconds=5), timestamp_pb2
      )
      _write_one_message(timer, duration_pb2.Duration(seconds=7), duration_pb2)
      decoded = _read(clock) + _read(timer)
    # The decoded classes are built from the file, so compare names and values.
    self.assertEqual(
        [(m.DESCRIPTOR.full_name, m.seconds) for m in decoded],
        [("google.protobuf.Timestamp", 5), ("google.protobuf.Duration", 7)],
    )

  def test_reads_a_schema_made_of_several_files(self):
    # Like a TensorFlow Example, an Api uses messages from other proto files.
    message = api_pb2.Api(
        name="arm",
        options=[type_pb2.Option(name="speed")],
        source_context=source_context_pb2.SourceContext(file_name="arm.proto"),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
      path = pathlib.Path(tmp_dir) / "api.mcap"
      _write_one_message(
          path, message, any_pb2, source_context_pb2, type_pb2, api_pb2
      )
      (decoded,) = _read(path)
    self.assertEqual(decoded.options[0].name, "speed")
    self.assertEqual(decoded.source_context.file_name, "arm.proto")

  def test_reads_example_features_of_the_given_topics(self):
    files = _example_files()
    pool = descriptor_pool.DescriptorPool()
    pool.Add(files.file[0])
    example = json_format.ParseDict(
        {
            "features": {
                "feature": {
                    "observation/qpos": {"floatList": {"value": [1.0, 2.0]}}
                }
            }
        },
        message_factory.GetMessageClass(
            pool.FindMessageTypeByName("tensorflow.Example")
        )(),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
      path = pathlib.Path(tmp_dir) / "episode.mcap"
      with open(path, "wb") as fh:
        writer = mcap_writer.Writer(fh)
        writer.start()
        schema_id = writer.register_schema(
            name="tensorflow.Example",
            encoding="protobuf",
            data=files.SerializeToString(),
        )
        for topic in (log_reader.TIMESTEP_TOPIC, "/other"):
          channel_id = writer.register_channel(
              topic=topic, message_encoding="protobuf", schema_id=schema_id
          )
          writer.add_message(
              channel_id,
              log_time=0,
              data=example.SerializeToString(),
              publish_time=0,
          )
        writer.finish()
      ((topic, features),) = log_reader.iter_features(
          epath.Path(path), [log_reader.TIMESTEP_TOPIC]
      )
      # In a fresh process, because other tests may have loaded TensorFlow.
      result = subprocess.run(
          [
              sys.executable,
              "-c",
              (
                  "import sys\n"
                  "from etils import epath\n"
                  "from mujoco_playground._src.real.common import log_reader\n"
                  f"list(log_reader.iter_features(epath.Path({str(path)!r})))\n"
                  "sys.exit('tensorflow' in sys.modules)\n"
              ),
          ],
          check=False,
      )
    self.assertEqual(topic, log_reader.TIMESTEP_TOPIC)
    self.assertEqual(
        list(features["observation/qpos"].float_list.value), [1.0, 2.0]
    )
    self.assertEqual(result.returncode, 0)

  def test_keeps_maps_readable_after_their_messages_are_gone(self):
    # A fresh process, because without the decoder cache this crashes it. The
    # second schema read in between would make a cache that evicts drop the
    # first decoder.
    message = struct_pb2.Struct()
    message.update({"x": [1.0, 2.0]})
    with tempfile.TemporaryDirectory() as tmp_dir:
      path = pathlib.Path(tmp_dir) / "struct.mcap"
      other = pathlib.Path(tmp_dir) / "timestamp.mcap"
      _write_one_message(path, message, struct_pb2)
      _write_one_message(
          other, timestamp_pb2.Timestamp(seconds=5), timestamp_pb2
      )
      result = subprocess.run(
          [
              sys.executable,
              "-c",
              (
                  "import gc, sys\n"
                  "from etils import epath\n"
                  "from mujoco_playground._src.real.common import log_reader\n"
                  f"path = epath.Path({str(path)!r})\n"
                  f"other = epath.Path({str(other)!r})\n"
                  "maps = []\n"
                  "for _ in range(500):\n"
                  "  for _, m in log_reader.iter_messages(path, ['/clock']):\n"
                  "    maps.append(m.fields)\n"
                  "  list(log_reader.iter_messages(other, ['/clock']))\n"
                  "gc.collect()\n"
                  "values = sum(len(m['x'].list_value.values) for m in maps)\n"
                  "sys.exit(values != 1000)\n"
              ),
          ],
          check=False,
      )
    self.assertEqual(result.returncode, 0)

  def test_reuses_the_decoder_of_a_schema(self):
    # A new decoder would build a new class; see _decoder.
    with tempfile.TemporaryDirectory() as tmp_dir:
      path = pathlib.Path(tmp_dir) / "clock.mcap"
      _write_one_message(
          path, timestamp_pb2.Timestamp(seconds=5), timestamp_pb2
      )
      (first,), (second,) = _read(path), _read(path)
    self.assertIs(type(first), type(second))

  def test_tells_apart_schemas_with_the_same_name(self):
    extended = _files(timestamp_pb2)
    extended.file[0].message_type[0].field.add(
        name="extra",
        number=3,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
    )
    message = timestamp_pb2.Timestamp(seconds=5)
    # Field 3 set to 9, which only the extended schema knows.
    data = message.SerializeToString() + bytes([3 << 3, 9])
    with tempfile.TemporaryDirectory() as tmp_dir:
      plain = pathlib.Path(tmp_dir) / "plain.mcap"
      wider = pathlib.Path(tmp_dir) / "wider.mcap"
      _write_one_message(plain, message, timestamp_pb2, data=data)
      _write_one_message(wider, message, files=extended, data=data)
      (first,), (second,) = _read(plain), _read(wider)
    self.assertNotIn("extra", type(first).DESCRIPTOR.fields_by_name)
    self.assertEqual(second.extra, 9)


if __name__ == "__main__":
  absltest.main()
