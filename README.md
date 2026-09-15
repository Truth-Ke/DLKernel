# DLKernel

High-performance CUDA kernels written in
[CuTe-DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL).

## Installation

The distribution is not published to PyPI yet, so install it from a checkout:

``` bash
git clone <your-fork-url> && cd DLKernel
pip install -e '.'

# With the optional extras
pip install -e '.[cu13]' --extra-index-url https://download.pytorch.org/whl/cu132
pip install -e '.[heuristics]'
```

## Requirements

- H100, B200/B300, 
- CUDA toolkit 12.8+
- Python 3.12

## Kernels

- RMSNorm forward + backward
- Softmax forward + backward
- Cross entropy forward + backward
- Layernorm forward + backward
- Hopper GEMM + epilogue
- Blackwell GEMM + epilogue
- Blackwell GeForce GEMM + epilogue

## Usage

```python
from DLKernel import rmsnorm, softmax, cross_entropy
```

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
