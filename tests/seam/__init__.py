"""SEAM test suite.

Present so that mypy names these modules ``tests.*``, which is what the ``[[tool.mypy.overrides]]``
section in ``pyproject.toml`` targets. Without it mypy names them ``conftest`` / ``test_*``, the
override silently matches nothing, and the test files get checked under the strict settings intended
for ``seam/`` only.
"""
