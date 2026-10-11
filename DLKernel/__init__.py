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
from DLKernel.cross_entropy import (
    cross_entropy,
    cross_entropy_bwd,
    cross_entropy_fwd,
)
from DLKernel.fused_grouped_gemm import (
    moe_down_proj_dact,
    moe_up_proj_act,
)
from DLKernel.gemm_base import gemm_with_activation, gemm_with_activation_grad
from DLKernel.gemm_interface import gemm, gemm_add
from DLKernel.gemm_symmetric import gemm_symmetric
from DLKernel.linear import linear_act_func, linear_func, linear_gated_func
from DLKernel.linear_cross_entropy import (
    LinearCrossEntropy,
    chunked_linear_cross_entropy,
)
from DLKernel.mlp import MLP
from DLKernel.rmsnorm import RMSNorm, rmsnorm
from DLKernel.rounding import RoundingMode
from DLKernel.softmax import softmax
from DLKernel.transform import hadamard_transform

__all__ = [
    "MLP",
    "LinearCrossEntropy",
    "RMSNorm",
    "RoundingMode",
    "chunked_linear_cross_entropy",
    "cross_entropy",
    "cross_entropy_bwd",
    "cross_entropy_fwd",
    "gemm",
    "gemm_add",
    "gemm_symmetric",
    "gemm_with_activation",
    "gemm_with_activation_grad",
    "hadamard_transform",
    "linear_act_func",
    "linear_func",
    "linear_gated_func",
    "moe_down_proj_dact",
    "moe_up_proj_act",
    "rmsnorm",
    "softmax",
]
