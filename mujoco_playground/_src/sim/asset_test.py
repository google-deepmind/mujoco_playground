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

"""Tests for asset.py and asset_loaders.py."""

import os
import tempfile

from absl.testing import absltest
import mujoco
from mujoco_playground._src import paths
from mujoco_playground._src.sim import asset

_MINIMAL_XML = """\
<mujoco>
  <worldbody>
    <body>
      <joint type="free"/>
      <geom type="sphere" size=".05"/>
    </body>
  </worldbody>
</mujoco>
"""


class AssetRegistryTest(absltest.TestCase):
  """Tests for the asset registry."""

  def test_sources_includes_builtins(self):
    registered = asset.sources()
    self.assertIn('local', registered)
    self.assertIn('mjz', registered)
    self.assertIn('runfile', registered)
    self.assertIn('github', registered)

  def test_register_duplicate_raises(self):
    with self.assertRaises(ValueError):
      asset.register_source('local', asset.LocalAsset)

  def test_get_no_source_raises(self):
    with self.assertRaises(ValueError):
      asset.get('/tmp/foo.xml')


class ParseSourceTest(absltest.TestCase):
  """Tests for URI source parsing."""

  def test_local(self):
    source, rest = asset._parse_source('local:/tmp/foo.xml')
    self.assertEqual(source, 'local')
    self.assertEqual(rest, '/tmp/foo.xml')

  def test_db_with_params(self):
    source, query = asset._parse_source('db:my_uid?origin=ORIGIN_MESHY')
    self.assertEqual(source, 'db')
    self.assertEqual(query, 'my_uid?origin=ORIGIN_MESHY')

  def test_github(self):
    source, query = asset._parse_source(
        'github:google-deepmind/mujoco_menagerie/aloha/scene.xml'
    )
    self.assertEqual(source, 'github')
    self.assertEqual(query, 'google-deepmind/mujoco_menagerie/aloha/scene.xml')

  def test_no_source_raises(self):
    with self.assertRaises(ValueError):
      asset._parse_source('/tmp/foo.xml')


class LocalAssetTest(absltest.TestCase):
  """Tests for loading assets from local file paths."""

  def setUp(self):
    super().setUp()
    tmpdir = tempfile.mkdtemp()
    self._xml_path = os.path.join(tmpdir, 'sphere.xml')
    with open(self._xml_path, 'w') as f:
      f.write(_MINIMAL_XML)

  def test_spec_returns_mjspec(self):
    a = asset.LocalAsset(f'local:{self._xml_path}')
    self.assertIsInstance(a.spec(), mujoco.MjSpec)

  def test_spec_compiles(self):
    a = asset.LocalAsset(f'local:{self._xml_path}')
    model = a.spec().compile()
    self.assertEqual(model.ngeom, 1)

  def test_get_local(self):
    a = asset.get(f'local:{self._xml_path}')
    self.assertIsInstance(a.spec(), mujoco.MjSpec)

  def test_identifier(self):
    a = asset.LocalAsset(f'local:{self._xml_path}')
    self.assertEqual(a.identifier, f'local:{self._xml_path}')


class RunfileAssetTest(absltest.TestCase):
  """Tests for loading assets from local testdata paths."""

  _CUBE = f'{paths.PROJECT_ROOT}/testdata/cube.xml'

  def test_spec_compiles(self):
    a = asset.RunfileAsset(f'runfile:{self._CUBE}')
    spec = a.spec()
    self.assertIsInstance(spec, mujoco.MjSpec)
    model = spec.compile()
    self.assertEqual(model.ngeom, 1)

  def test_get_runfile(self):
    a = asset.get(f'runfile:{self._CUBE}')
    self.assertIsInstance(a.spec(), mujoco.MjSpec)

  def test_identifier(self):
    a = asset.RunfileAsset(f'runfile:{self._CUBE}')
    self.assertEqual(a.identifier, f'runfile:{self._CUBE}')


class GithubLoaderTest(absltest.TestCase):
  """Tests for the github: source (menagerie / aloha_sim)."""

  def test_get_menagerie(self):
    a = asset.get('github:google-deepmind/mujoco_menagerie/aloha/scene.xml')
    self.assertIsInstance(a.spec(), mujoco.MjSpec)

  def test_get_aloha_sim(self):
    a = asset.get(
        'github:google-deepmind/aloha_sim/aloha_sim/assets/edr/pen/model.xml'
    )
    self.assertIsInstance(a.spec(), mujoco.MjSpec)

  def test_parse_github_uri(self):
    repo, path, params = asset._parse_github_uri(
        'google-deepmind/mujoco_menagerie/aloha/scene.xml?commit=abc'
    )
    self.assertEqual(repo, 'google-deepmind/mujoco_menagerie')
    self.assertEqual(path, 'aloha/scene.xml')
    self.assertEqual(params, {'commit': 'abc'})

  def test_parse_github_uri_no_path_raises(self):
    with self.assertRaises(ValueError):
      asset._parse_github_uri('google-deepmind/mujoco_menagerie')


class QueryTest(absltest.TestCase):
  """Tests for the query method."""

  def test_query_unsupported_source_raises(self):
    with self.assertRaises(NotImplementedError):
      list(asset.query('local:foo'))

  def test_query_unknown_source_raises(self):
    with self.assertRaises(KeyError):
      list(asset.query('nosource:foo'))


if __name__ == '__main__':
  absltest.main()
