.PHONY: sync
sync:
	uv sync --extra rl --group dev --upgrade-package mujoco --upgrade-package mujoco-warp

.PHONY: sync-cu12
sync-cu12:
	rm -rf $${UV_PROJECT_ENVIRONMENT:-.venv}/lib/python*/site-packages/nvidia*
	uv sync --group cu12 --extra rl --group dev --upgrade-package mujoco --upgrade-package mujoco-warp
	uv pip install --find-links https://github.com/NVIDIA/warp/releases/expanded_assets/v1.18.0 "warp-lang==1.18.0+cu12"

.PHONY: sync-cpu
sync-cpu:
	uv sync --group cpu --extra rl --group dev --upgrade-package mujoco --upgrade-package mujoco-warp

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

_RL_TESTS := mujoco_playground/_src/rl/rsl_rl mujoco_playground/rl
_IGNORE_RL := --ignore=mujoco_playground/_src/rl/rsl_rl --ignore=mujoco_playground/rl
N_WORKERS ?= auto

.PHONY: test
test:
	$(MUJOCO_GL_ENV) uv run --extra rl --group dev pytest mujoco_playground/ $(_IGNORE_RL) -n $(N_WORKERS) --dist worksteal -v
	$(MUJOCO_GL_ENV) uv run --extra rl --group dev pytest $(_RL_TESTS) -n $(N_WORKERS) --dist worksteal -v

# executorch pins one exact torch version, so it is added for its tests only.
.PHONY: test-executorch
test-executorch:
	$(MUJOCO_GL_ENV) REQUIRE_EXECUTORCH=1 uv run --extra rl --group dev --with executorch==1.6.0 python -m pytest mujoco_playground/tasks/manipulation/pick_cube/so101/deploy/run_test.py -v

# Optional dependencies must not be imported at module scope.
NO_EXTRAS_ENV := .venv-no-extras
.PHONY: check-imports
check-imports:
	UV_PROJECT_ENVIRONMENT=$(NO_EXTRAS_ENV) uv sync --group cpu --no-default-groups --upgrade-package mujoco --upgrade-package mujoco-warp
	UV_PROJECT_ENVIRONMENT=$(NO_EXTRAS_ENV) uv run --no-sync python -m mujoco_playground.import_check

DOCS_PORT ?= 8888

.PHONY: docs
docs:
	uv run --no-project --with-requirements third_party/reaf_sim/docs/requirements.txt sphinx-build -M html third_party/reaf_sim/docs docs/_build

.PHONY: serve-docs
serve-docs: docs
	uv run --no-project python -m http.server $(DOCS_PORT) --directory docs/_build/html
