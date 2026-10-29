# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Development Setup

Clone the repository, sync development dependencies with [`uv`](https://docs.astral.sh/uv/), and install the `pre-commit` hooks:

```bash
git clone https://github.com/google-deepmind/reaf-sim.git && cd reaf-sim
make install  # or `make install-cpu` for CPU-only
uv run --no-sync pre-commit install
```

Before opening a pull request, run `pre-commit` (which formats with `pyink` and
`isort`, checks copyright headers, and type-checks with `pyrefly`) and run the
test suite:

```bash
uv run --no-sync pre-commit run --all-files  # Format and type-check (or `make format` / `make typecheck`)
make test                                    # Run pytest in parallel
make docs                                    # Build HTML docs (or `make serve-docs` to preview locally)
```

## Code Reviews

All submissions, including submissions by project members, require review via
[GitHub pull requests](https://help.github.com/articles/about-pull-requests/).

## Updating the Changelog

Every pull request that introduces new features, bug fixes, or API changes
must include an entry in `docs/changelog.rst` under the `unreleased` section.
`reaf-sim` adheres to [MuJoCo's semantic versioning scheme](https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md);
see `docs/versioning.rst` for release semantics and categorization.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).
