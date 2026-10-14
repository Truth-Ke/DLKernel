# Copyright (c) 2025-2026, Tri Dao.
"""Persistent kernel-cache utilities for DLKernel.

Public API
----------

Persistent ``.o`` cache:
* :func:`jit_cache` — decorator that wraps a kernel-compile function with
  in-memory + persistent ``.o`` caching (see :mod:`DLKernel.cache.jit`).
* :data:`CACHE_ENABLED`, :data:`CACHE_DIR`, :data:`EXTRA_SOURCE_DIRS` —
  static-config flags.
* :class:`FileLock`, :func:`get_cache_path`, :class:`CacheInfo` —
  supporting types.

Async compilation (see :mod:`DLKernel.cache.async_compile`):
* :class:`CompilePending` — raised by ``jit_cache`` on a cold miss while a
  compile pool is active; the caller defers and retries once the ``.o``
  lands.
* :func:`pool_scope` — activate a compile pool for a scoped block (used by
  the autotuner's bench loop).

Discovery + prewarming (see :mod:`DLKernel.cache.capture` /
:mod:`DLKernel.cache.precompile`):
* :func:`precompile` — discover a list of calls' specializations and fill the
  artifact cache in the async pool, ahead of the real workload; returns a
  :class:`PrecompileReport` with counts and elapsed time.
  ``DLKernel.cache.precompile.materialize`` accepts process-local
  :class:`DLKernel.cache.capture.CompileSpec` requests directly.

CRITICAL ORDERING: the static-config flags below MUST be defined before the
``from DLKernel.cache.jit import ...`` block. ``DLKernel/cache/jit.py`` does
``import DLKernel.cache as _state`` at its module top; Python returns the
partially-initialized package object, and lookups inside ``jit_cache``'s
wrapper rely on these names already existing at that checkpoint. Reordering
the imports here, even via an auto-formatter, will break the first kernel
compile with ``AttributeError``. (``capture`` above is the one deliberate
exception: it is dependency-free and reads no package state.) The defensive
unit tests in ``tests/test_cache.py`` exercise this path end-to-end.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# Keep discovery names available for internal imports, but outside __all__;
# precompile() is the ordinary public entry point.
from DLKernel.cache.capture import CompileCaptured, capture_compiles  # noqa: E402,F401

CACHE_ENABLED: bool = os.getenv("DLKERNEL_CACHE_ENABLED", "1") == "1"
# Two independent cache knobs, one per lifecycle (FlashInfer-style: every
# cache concern owns its own variable and default; no cross-fallback):
#
# * ``DLKERNEL_ARTIFACT_CACHE_DIR`` (this flag) — compiled ``.o`` artifacts.
#   Deterministic and safe to share across processes; unset defaults to the
#   per-user temp dir below.
# * ``DLKERNEL_CACHE_DIR`` — the autotuner's winner-JSON store
#   (:class:`DLKernel.autotuner.FileCacheManager`); empirical measurements a
#   caller may want hermetic (per-job benchmark dirs). Never steers
#   artifacts.
#
# Artifacts are reusable within the encoded compatibility domain: Python,
# CuTe-DSL, TVM-FFI, CUDA/PyTorch ABI, target + dispatch arch, cache format,
# DLKernel sources, the defining module's source, and explicit compile args.
# Call sites must pass every codegen input to @jit_cache; downstream helper
# sources must be registered in EXTRA_SOURCE_DIRS. Untracked external inputs
# cannot be inferred from a cache hit alone.
CACHE_DIR: Optional[str] = os.getenv("DLKERNEL_ARTIFACT_CACHE_DIR", None)

#: Downstream projects can append directories here to include their sources
#: in the cache fingerprint. Must be set before the first jit_cache call.
EXTRA_SOURCE_DIRS: List[Path] = []


# ---------------------------------------------------------------------------
# Public API surface. Imported AFTER the flags are defined.
# ---------------------------------------------------------------------------

from DLKernel.cache.jit import (  # noqa: E402
    EXPORT_FUNC_NAME,
    LOCK_TIMEOUT,
    CacheInfo,
    FileLock,
    get_cache_path,
    jit_cache,
)
from DLKernel.cache.async_compile import (  # noqa: E402
    CompilePending,
    pool_scope,
)
from DLKernel.cache.precompile import PrecompileReport, precompile  # noqa: E402,F401

__all__ = [
    # Persistent .o cache.
    "jit_cache",
    "CacheInfo",
    "EXPORT_FUNC_NAME",
    "LOCK_TIMEOUT",
    "FileLock",
    "get_cache_path",
    # Async compilation.
    "CompilePending",
    "pool_scope",
    # Object-cache prewarming (single public entry; discovery internals in
    # DLKernel.cache.capture).
    "precompile",
    "PrecompileReport",
]
