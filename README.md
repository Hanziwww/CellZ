# CELLZ


[![Python 3.10–3.13](https://img.shields.io/badge/Python-3.10%E2%80%933.13-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-language-orange?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![maturin](https://img.shields.io/badge/maturin-build-blue?style=flat-square)](https://github.com/PyO3/maturin)
[![Wheel](https://img.shields.io/pypi/wheel/cellz.svg?style=flat-square)](https://pypi.org/project/cellz/)
[![Implementation](https://img.shields.io/pypi/implementation/cellz.svg?style=flat-square)](https://pypi.org/project/cellz/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square)](https://opensource.org/licenses/BSD-3-Clause)

High-performance compressor for single-cell sparse matrices

## Features

- **Adaptive hybrid value encoding**: optimized for zero-inflated distributions of single-cell UMI count matrices
- **Dynamic threshold selection**: parameters are chosen by exact cost-function optimization
- **Optional block compression**: xz compression on blocks to further reduce size

## Installation

### Build from source

Requires a Rust toolchain and Python 3.10+:

```bash
# Install maturin
pip install maturin

# Install in development mode
maturin develop --release

# Or build a wheel
maturin build --release
pip install target/wheels/cellz-*.whl
```

## Usage

### Python API

```python
import cellz
import scipy.sparse as sp

# Compress a CSR matrix
csr = sp.random(10000, 2000, density=0.1, format='csr')
compressed_bytes = cellz.compress_csr_to_cellz(csr)

# Decompress
decompressed_csr = cellz.decompress_cellz_to_csr(compressed_bytes)

# Compress a .h5ad file
cellz.compress_h5ad_to_cellz("data.h5ad", "data.cellz")

# Decompress a .cellz file
cellz.decompress_cellz_to_h5ad("data.cellz", "data_restored.h5ad")
```

### Advanced options

```python
# Enable block-level xz compression
compressed = cellz.compress_csr_to_cellz(
    csr,
    target_block_nnz=131072,      # target nonzeros per block
    block_xz=True,                 # enable xz compression
    block_xz_level=9,              # compression level
    block_xz_min_bytes=4096,       # minimum byte threshold
    block_xz_min_ratio=0.99        # minimum compression ratio threshold
)
```

## Algorithm

CELLZ uses a multi-stage coding strategy:

1. **Matrix reordering**: sort rows/cols by nonzero counts (descending) to improve locality
2. **Column blocking**: partition the matrix into blocks with a fixed number of nonzeros
3. **Position coding**: use Golomb–Rice coding for row-index gaps within a column block
4. **Value coding**: adaptive hybrid encoding
   - value = 1: 1 bit
   - small values (2 to threshold−1): 2 bits + fixed/truncated binary
   - large values (≥ threshold): 2 bits + Golomb–Rice
5. **Optional block compression**: xz/lzma on the bitstream

All parameters (threshold, k, bit-widths) are selected by exact cost-function optimization.

## License

BSD-3-Clause license
