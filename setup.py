"""Setuptools shim for editable/package installs.

The project metadata lives in ``pyproject.toml``. This file exists only because
some packaging tools still expect a ``setup.py`` entry point.
"""

from setuptools import setup
setup()  # Everything in pyproject.toml
