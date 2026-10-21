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
"""Configuration file for the Sphinx documentation builder."""

import inspect
import os
import sys
import types
import typing

import sphinx.ext.autodoc.mock as autodoc_mock

_DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_DOCS_DIR)
if os.path.isdir(os.path.join(_ROOT_DIR, "reaf")):
  sys.path.insert(0, _ROOT_DIR)
else:
  _reaf_pkg = types.ModuleType("reaf")
  _reaf_pkg.__path__ = [os.path.dirname(_ROOT_DIR)]
  sys.modules.setdefault("reaf", _reaf_pkg)

autodoc_mock._MockObject.__or__ = lambda self, other: self  # pylint: disable=protected-access
autodoc_mock._MockObject.__ror__ = lambda self, other: self  # pylint: disable=protected-access
autodoc_mock._MockObject.__int__ = lambda self: 0  # pylint: disable=protected-access
_orig_mock_getattr = autodoc_mock._MockObject.__getattr__  # pylint: disable=protected-access
_orig_mock_call = autodoc_mock._MockObject.__call__  # pylint: disable=protected-access


def _mock_getattr(self, key: str):
  if key in ("ResetOptions", "Action", "Observation", "PolicyDataType"):
    return typing.TypeVar(key)
  return _orig_mock_getattr(self, key)


def _mock_call(self, *args, **kwargs):
  if len(args) == 1 and not kwargs and inspect.isfunction(args[0]):
    return args[0]
  return _orig_mock_call(self, *args, **kwargs)


autodoc_mock._MockObject.__getattr__ = _mock_getattr  # pylint: disable=protected-access
autodoc_mock._MockObject.__call__ = _mock_call  # pylint: disable=protected-access

autodoc_mock_imports = [
    "absl",
    "dm_env",
    "gdm_robotics",
    "immutabledict",
    "mujoco",
    "mujoco_warp",
    "numpy",
    "reaf.core",
    "reaf.simulation.viewer",
    "torch",
    "tree",
    "warp",
]

project = "reaf-sim"
copyright = "2026, Google LLC"  # pylint: disable=redefined-builtin
author = "Google DeepMind"

master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

napoleon_google_docstring = True
autodoc_class_signature = "separated"
add_module_names = False
toc_object_entries_show_parents = "hide"
default_role = "literal"

rst_prolog = """
.. role:: green
.. role:: red
.. role:: badge-green
.. role:: badge-red
"""

autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "special-members": "__call__",
    "exclude-members": "__init__, __new__",
}

_documented_attrs: set[tuple[str, str]] = set()


def _process_docstring(app, what, name, obj, options, lines):
  """Cleans module/dataclass docstrings and tracks documented attributes."""
  del app, obj, options
  if what == "module":
    lines.clear()
  elif what == "class":
    cls_name = name.rpartition(".")[2]
    if lines and lines[0].startswith(f"{cls_name}("):
      lines.clear()
    for line in lines:
      if line.startswith(".. attribute:: "):
        _documented_attrs.add(
            (name, line.removeprefix(".. attribute:: ").strip())
        )


def _skip_member(app, what, name, obj, skip, options):
  """Skips duplicate class attributes and out-of-section module members."""
  del options
  cur_mod = app.env.current_document.autodoc_module
  cur_cls = app.env.current_document.autodoc_class
  if what == "class" and (f"{cur_mod}.{cur_cls}", name) in _documented_attrs:
    return True
  if what == "module":
    mod = getattr(obj, "__module__", "") or ""
    if not callable(obj) or not mod.startswith("reaf.simulation.torch"):
      return True
    if cur_mod == "reaf.simulation.torch":
      if mod.startswith("reaf.simulation.torch._src.builders"):
        return True
    elif cur_mod == "reaf.simulation.torch.flat_scene":
      if not mod.startswith("reaf.simulation.torch._src.builders"):
        return True
    elif cur_mod == "reaf.simulation.torch.viewer":
      if not mod.startswith("reaf.simulation.torch.viewer._src"):
        return True
  return skip


# pydata-sphinx-theme hides the sidebar TOC on pages without a top-level
# toctree ancestor (index, search, genindex). Our sidebar renders the full
# toctree from the root (see _templates/sidebar-nav-bs.html), so it is never
# empty.
def _keep_sidebar_toctree(app, pagename, templatename, context, doctree):
  """Keeps the left sidebar TOC on every page and sets the section brand."""
  del app, templatename, doctree
  context["suppress_sidebar_toctree"] = lambda *args, **kwargs: False
  context["docstitle"] = "reaf-sim"


def setup(app):
  """Registers Sphinx autodoc and theme hooks."""
  app.connect("autodoc-process-docstring", _process_docstring)
  app.connect("autodoc-skip-member", _skip_member)
  # Runs after pydata-sphinx-theme has populated the page context.
  app.connect("html-page-context", _keep_sidebar_toctree, priority=1000)


exclude_patterns = ["_build"]
templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"
html_title = "reaf-sim"
html_context = {
    "github_user": "google-deepmind",
    "github_repo": "reaf-sim",
    "github_version": "main",
    "doc_path": "docs",
}

# Layout: the full table of contents lives in the left sidebar (see
# _templates/sidebar-nav-bs.html), the header carries no page links, and the
# right sidebar only holds the outline of the current page. Colors and fonts
# are set in css/theme_overrides.css.
html_sidebars = {"**": ["sidebar-nav-bs.html"]}
html_theme_options = {
    "navbar_center": [],
    "primary_sidebar_end": [],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright", "last-updated"],
    "footer_end": [],
    "navigation_depth": 2,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "github_url": "https://github.com/google-deepmind/reaf-sim",
}

html_favicon = "_static/favicons/favicon.svg"

html_last_updated_fmt = "%b %d, %Y"
html_show_sphinx = False

html_static_path = [
    "_static",
    "css",
]
html_css_files = ["theme_overrides.css"]
html_js_files = ["onthispage.js"]
html_permalinks_icon = "#"
