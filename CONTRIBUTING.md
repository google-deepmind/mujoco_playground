# How to Contribute

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Adding New Tasks

We welcome contributions of new tasks, particularly those that demonstrate:

- Impressive capabilities in simulation
- Tasks that have been successfully transferred to real robots

When submitting a new task, please ensure:

1. The task is well-documented with clear objectives and reward structure
2. Add the relevant RL hyperparameters to the config file so that it can be
   independently reproduced
3. Ensure that it works across at least 3 seeds
4. Show a video of the behavior
5. Make sure your new task passes all the tests

For an example of a well-structured task contribution, see @Andrew-Luo1's
excellent [ALOHA Handover Task
PR](https://github.com/google-deepmind/mujoco_playground/pull/29).

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Linting and Code Health

Before submitting a PR, please run:

```shell
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```

or you can run manually

```shell
pyink .
isort .
pylint . --rcfile=pylintrc
pytype .
```

and resolve any issues that pop up.

## Testing

To run the tests, use the following command:

```shell
pytest
```

