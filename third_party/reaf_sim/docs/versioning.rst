Versioning
==========

`reaf-sim` adheres to `MuJoCo's semantic versioning scheme
<https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md>`_ (``SUPERMAJOR.MAJOR.MINOR_OR_PATCH``):

.. list-table::
    :header-rows: 1
    :widths: 25 75

    - - Level
      - Semantics
    - - ``SUPERMAJOR``
      - Breaking API changes and/or fundamentally significant redesigns.
    - - ``MAJOR``
      - Breaking changes and possibly new features (breaking changes are batched into major releases).
    - - ``MINOR_OR_PATCH``
      - Backwards-compatible new features, performance improvements, documentation, or bug fixes.

``MINOR_OR_PATCH`` guarantees API backwards compatibility. Consult the :doc:`Changelog <changelog>` for historical
release notes and upcoming changes staged under unreleased.
