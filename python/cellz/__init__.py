"""
CELLZ - High-performance compressor for single-cell sparse matrices

Rust-accelerated implementation of adaptive hybrid encoding for scRNA-seq data.
"""

from .api import (
    compress_h5ad_to_cellz,
    decompress_cellz_to_h5ad,
    compress_csr_to_cellz,
    decompress_cellz_to_csr,
)

__version__ = "0.1.0"

# Convenience aliases
compress = compress_csr_to_cellz
decompress = decompress_cellz_to_csr

__all__ = [
    "compress_h5ad_to_cellz",
    "decompress_cellz_to_h5ad",
    "compress_csr_to_cellz",
    "decompress_cellz_to_csr",
    "compress",
    "decompress",
]

