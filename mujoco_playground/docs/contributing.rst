How to Contribute
=================

We'd love to accept your patches and contributions to this project. There are just a few small guidelines you need to
follow.

Contributor License Agreement
-----------------------------

Contributions to this project must be accompanied by a Contributor License Agreement. You (or your employer) retain the
copyright to your contribution, this simply gives us permission to use and redistribute your contributions as part of
the project. Head over to https://cla.developers.google.com/ to see your current agreements on file or to sign a new
one.

You generally only need to submit a CLA once, so if you've already submitted one (even if it was for a different
project), you probably don't need to do it again.

Development Setup
-----------------

Clone the repository, sync development dependencies with `uv <https://docs.astral.sh/uv/>`_, and install the
``pre-commit`` hooks:

.. code-block:: bash

    git clone https://github.com/google-deepmind/mujoco_playground.git && cd mujoco_playground
    make sync  # GPU (CUDA 12.4) + RSL-RL + dev (or `make sync-cpu` for CPU-only)
    uv run pre-commit install

To install optional real-robot hardware drivers alongside your environment, pass ``--extra so101`` or ``--extra yam`` to
``uv sync`` (e.g., ``uv sync --extra cpu --extra so101 --group dev``).

Before opening a pull request, run ``pre-commit`` (which formats with ``pyink`` and ``isort``, checks copyright headers
and optional imports via ``make check-imports``, and type-checks with ``pyrefly``) and run the test suite:

.. code-block:: bash

    uv run pre-commit run --all-files  # Format, check imports, and type-check (or `make format` / `make typecheck`)
    make test                          # Run pytest in parallel

To iterate quickly on a single test file on Linux:

.. code-block:: bash

    MUJOCO_GL=osmesa uv run --no-sync --group dev pytest path/to/test.py -v

Conventions
-----------

- **CLI / Entrypoints**: Use ``absl`` (``from absl import app, flags, logging``) for scripts and entrypoints; do not use
  ``argparse``.
- **Paths & Assets**: Use ``etils.epath`` for filesystem paths and ``mujoco_playground.asset`` to resolve model URIs.
- **Optional Hardware SDKs**: Hardware drivers are optional extras and must be imported lazily inside the functions that
  use them, never at module scope (verified by ``make check-imports``).

Code Reviews
------------

All submissions, including submissions by project members, require review via `GitHub pull requests
<https://help.github.com/articles/about-pull-requests/>`_.

Updating the Changelog
----------------------

Every pull request that introduces new features, fixes bugs, or modifies the API must include an entry in
:doc:`changelog` under the ``unreleased`` section. MuJoCo Playground adheres to `MuJoCo's semantic versioning scheme
<https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md>`_.

Community Guidelines
--------------------

This project follows `Google's Open Source Community Guidelines <https://opensource.google/conduct/>`_.
