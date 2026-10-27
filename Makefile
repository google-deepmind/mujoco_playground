# reaf builds from git until its next PyPI release; its CMake does not pin a
# C++ standard and abseil requires >= 17.
export CMAKE_ARGS := -DCMAKE_CXX_STANDARD=17
# Workaround until reaf updates its abseil-cpp pin upstream: under GCC 15 with
# -march=x86-64-v3, bmi2intrin.h requires immintrin.h to be included first.
ifeq ($(shell uname -m),x86_64)
  export CMAKE_ARGS += -DCMAKE_CXX_FLAGS="-include immintrin.h"
endif

.PHONY: sync
sync:
	uv sync --extra cu124 --extra rl --group dev --upgrade-package mujoco --upgrade-package mujoco-warp --default-index https://pypi.org/simple

.PHONY: sync-cpu
sync-cpu:
	uv sync --extra cpu --extra rl --group dev --upgrade-package mujoco --upgrade-package mujoco-warp --default-index https://pypi.org/simple

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

_RL_TESTS := mujoco_playground/_src/rl/rsl_rl mujoco_playground/rl
_IGNORE_RL := --ignore=mujoco_playground/_src/rl/rsl_rl --ignore=mujoco_playground/rl
TEST_EXTRAS ?= --extra cpu --extra rl
N_WORKERS ?= auto

.PHONY: sync-test
sync-test:
	uv sync $(TEST_EXTRAS) --group dev --upgrade-package mujoco --upgrade-package mujoco-warp --default-index https://pypi.org/simple

.PHONY: test
test: sync-test
	$(MUJOCO_GL_ENV) uv run --no-sync --group dev pytest mujoco_playground/ --ignore=mujoco_playground/rss2025/ $(_IGNORE_RL) -n $(N_WORKERS) --dist worksteal -v
	$(MUJOCO_GL_ENV) uv run --no-sync --group dev pytest $(_RL_TESTS) -n $(N_WORKERS) --dist worksteal -v

# Optional dependencies must not be imported at module scope.
NO_EXTRAS_ENV := .venv-no-extras
.PHONY: check-imports
check-imports:
	UV_PROJECT_ENVIRONMENT=$(NO_EXTRAS_ENV) uv sync --no-default-groups --upgrade-package mujoco --upgrade-package mujoco-warp --default-index https://pypi.org/simple
	UV_PROJECT_ENVIRONMENT=$(NO_EXTRAS_ENV) uv run --no-sync python -m mujoco_playground.import_check
