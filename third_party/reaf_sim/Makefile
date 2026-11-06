.PHONY: install
install:
	uv sync --group dev --upgrade-package mujoco --upgrade-package mujoco-warp

.PHONY: install-cu12
install-cu12:
	rm -rf $${UV_PROJECT_ENVIRONMENT:-.venv}/lib/python*/site-packages/nvidia*
	uv sync --group cu12 --group dev --upgrade-package mujoco --upgrade-package mujoco-warp
	uv pip install --find-links https://github.com/NVIDIA/warp/releases/expanded_assets/v1.18.0 "warp-lang==1.18.0+cu12"

.PHONY: install-cpu
install-cpu:
	uv sync --group cpu --group dev --upgrade-package mujoco --upgrade-package mujoco-warp

.PHONY: format
format:
	uv run --group dev pyink .
	uv run --group dev isort .

.venv/typings/mujoco:
	uv run --group dev pybind11-stubgen mujoco -o .venv/typings --ignore-all-errors
	uv run --group dev python -c 'import pathlib, re; mj = pathlib.Path(".venv/typings/mujoco"); (mj / "_specs.pyi").write_text("from typing import Any\ndef __getattr__(name: str) -> Any: ...\n"); [p.write_text("from typing import Any\n" + re.sub(r"@staticmethod\s+def bind\((?:data: MjData|model: MjModel),", "def bind(self,", re.sub(r"numpy\.(?:typing\.NDArray|ndarray)\[[^\]]+\]", "Any", p.read_text()))) for p in mj.glob("*.pyi") if p.name != "_specs.pyi"]'

.PHONY: typecheck
typecheck: .venv/typings/mujoco
	uv run --group dev pyrefly check

# Headless rendering needs osmesa, which exists only on Linux; on macOS the
# default backend works and the variable would be rejected.
ifeq ($(shell uname -s),Linux)
  MUJOCO_GL_ENV := MUJOCO_GL=osmesa
endif

N_WORKERS ?= auto

.PHONY: test
test:
	$(MUJOCO_GL_ENV) uv run --group dev pytest reaf/simulation/ docs/ -n $(N_WORKERS) --dist worksteal -v

DOCS_PORT ?= 8888

.PHONY: docs
docs:
	uv run --no-project --with-requirements docs/requirements.txt sphinx-build -M html docs docs/_build

.PHONY: serve-docs
serve-docs: docs
	uv run --no-project python -m http.server $(DOCS_PORT) --directory docs/_build/html
