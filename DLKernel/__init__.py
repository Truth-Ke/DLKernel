__version__ = "0.6.5"

import os

import DLKernel.dsl as _dlkernel_dsl  # noqa: F401

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from DLKernel.dsl import cute_dsl_ptxas as _cute_dsl_ptxas

    # Patch before importing any modules that instantiate CuTeDSL. The patch
    # forces PTX dumping so the CUDA library loader can replace CUTLASS DSL's
    # embedded ptxas-library cubin with one assembled by system ptxas.
    _cute_dsl_ptxas.patch()

# Pythonic CuTe tensor indexing (`:` / `...` sugar) is installed as a side effect
# of importing `DLKernel.dsl`, which imports `DLKernel.dsl.cute_tensor_indexing` and
# monkey-patches CuTe's tensor classes process-wide.
from DLKernel.rmsnorm import rmsnorm  # noqa: E402
from DLKernel.softmax import softmax  # noqa: E402
from DLKernel.cross_entropy import cross_entropy  # noqa: E402
from DLKernel.rounding import RoundingMode  # noqa: E402


__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
    "RoundingMode",
]
