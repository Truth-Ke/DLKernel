"""Compatibility tombstone for the removed DLKernel trace API."""

raise ImportError(
    "DLKernel.trace has been removed. DLKernel now uses NVIDIA IKET directly; import "
    "`cutlass.cute.experimental.iket` and run workloads under "
    "`python -m iket.cli.main ... profile -- ...`. See "
    "`examples/example_iket_trace.py` for a minimal marker workload and "
    "`examples/example_gemm_trace.py` for a real GEMM trace."
)
