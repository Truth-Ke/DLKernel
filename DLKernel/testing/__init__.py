# Copyright (c) 2026, Tri Dao.
"""Reusable pytest plugin for DLKernel test workflows.

This subpackage's only contents is the reusable pytest plugin in
:mod:`DLKernel.testing.pytest_plugin`, which wires the ``--async-compile`` pool
(defer-and-retry kernel compilation) into a pytest run::

    # In a downstream project's conftest.py:
    pytest_plugins = ["DLKernel.testing.pytest_plugin"]
"""
