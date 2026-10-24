# reaf builds from git until its next PyPI release; its CMake does not pin a
# C++ standard and abseil requires >= 17.
export CMAKE_ARGS := -DCMAKE_CXX_STANDARD=17

.PHONY: install
install:
	uv sync --group dev --upgrade-package mujoco --upgrade-package mujoco-warp --default-index https://pypi.org/simple

.PHONY: install-cu124
install-cu124:
	uv sync --extra cu124 --group dev --upgrade-package mujoco --upgrade-package mujoco-warp --default-index https://pypi.org/simple

.PHONY: install-cpu
install-cpu:
	uv sync --extra cpu --group dev --upgrade-package mujoco --upgrade-package mujoco-warp --default-index https://pypi.org/simple

.PHONY: format
format:
	uv run --group dev pyink .
	uv run --group dev isort .

.venv/typings/mujoco:
	uv run --no-sync --group dev pybind11-stubgen mujoco -o .venv/typings --ignore-all-errors
	uv run --no-sync --group dev python -c 'import pathlib, re; mj = pathlib.Path(".venv/typings/mujoco"); (mj / "_specs.pyi").write_text("from typing import Any\ndef __getattr__(name: str) -> Any: ...\n"); [p.write_text("from typing import Any\n" + re.sub(r"@staticmethod\s+def bind\((?:data: MjData|model: MjModel),", "def bind(self,", re.sub(r"numpy\.(?:typing\.NDArray|ndarray)\[[^\]]+\]", "Any", p.read_text()))) for p in mj.glob("*.pyi") if p.name != "_specs.pyi"]'

.PHONY: typecheck
typecheck: .venv/typings/mujoco
	uv run --no-sync --group dev pyrefly check

# Headless rendering needs osmesa, which exists only on Linux; on macOS the
# default backend works and the variable would be rejected.
ifeq ($(shell uname -s),Linux)
  MUJOCO_GL_ENV := MUJOCO_GL=osmesa
endif

TEST_EXTRAS ?= --extra cpu
N_WORKERS ?= auto

.PHONY: sync-test
sync-test:
	uv sync $(TEST_EXTRAS) --group dev --upgrade-package mujoco --upgrade-package mujoco-warp --default-index https://pypi.org/simple

.PHONY: test
test: sync-test
	$(MUJOCO_GL_ENV) uv run --no-sync --group dev pytest reaf/simulation/ docs/ -n $(N_WORKERS) --dist worksteal -v
