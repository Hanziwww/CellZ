"""
Python API for CELLZ compression/decompression.

This module provides high-level functions for compressing and decompressing
sparse matrices using the CELLZ format, optimized for single-cell genomics data.
"""

import numpy as np
from scipy import sparse as sp
from ._cellz import compress_csc_to_cellz as _compress_csc_to_cellz
from ._cellz import decompress_cellz_to_csr as _decompress_cellz_to_csr


def compress_csr_to_cellz(
    csr_matrix,
    target_block_nnz=131072,
    block_xz=False,
    block_xz_level=9,
    block_xz_min_bytes=4096,
    block_xz_min_ratio=0.99,
):
    """
    Compress a CSR sparse matrix to CELLZ format.

    This function converts a CSR (Compressed Sparse Row) matrix to the CELLZ format,
    which is optimized for single-cell genomics data. The matrix is internally
    converted to CSC format for compression.

    Args:
        csr_matrix: A scipy.sparse.csr_matrix instance or any matrix that can be
            converted to CSR format. If not already CSR, it will be converted automatically.
        target_block_nnz (int, optional): Target number of non-zero elements per block.
            Controls the granularity of compression blocks. Default is 131072 (128K).
        block_xz (bool, optional): Whether to enable block-level XZ compression.
            Can significantly reduce file size at the cost of compression speed.
            Default is False.
        block_xz_level (int, optional): XZ compression level (0-9), where 0 is fastest
            and 9 is best compression. Only used if block_xz is True. Default is 9.
        block_xz_min_bytes (int, optional): Minimum block size in bytes to apply XZ
            compression. Blocks smaller than this threshold will not be XZ-compressed.
            Default is 4096 (4KB).
        block_xz_min_ratio (float, optional): Minimum compression ratio threshold for
            applying XZ compression. If XZ doesn't achieve this ratio, the uncompressed
            block is used instead. Default is 0.99.

    Returns:
        bytes: The compressed data in CELLZ format, ready to be written to a file.

    Examples:
        >>> import scipy.sparse as sp
        >>> import numpy as np
        >>> # Create a sparse matrix
        >>> data = np.random.randint(0, 100, 1000)
        >>> row = np.random.randint(0, 100, 1000)
        >>> col = np.random.randint(0, 50, 1000)
        >>> matrix = sp.csr_matrix((data, (row, col)), shape=(100, 50))
        >>> # Compress to CELLZ
        >>> compressed = compress_csr_to_cellz(matrix)
        >>> # With XZ compression enabled
        >>> compressed = compress_csr_to_cellz(matrix, block_xz=True, block_xz_level=6)
    """
    if not sp.isspmatrix_csr(csr_matrix):
        csr_matrix = sp.csr_matrix(csr_matrix)
    
    # Convert to CSC format for compression
    csc = csr_matrix.tocsc()
    
    nrows, ncols = csc.shape
    indptr = np.asarray(csc.indptr, dtype=np.int64)
    indices = np.asarray(csc.indices, dtype=np.int32)
    data = np.asarray(csc.data, dtype=np.float64)
    
    # Determine data type
    dtype_str = "float64"
    if np.issubdtype(csc.data.dtype, np.integer):
        if csc.data.dtype.itemsize <= 4:
            dtype_str = "int32"
        else:
            dtype_str = "int64"
    elif np.issubdtype(csc.data.dtype, np.floating):
        if csc.data.dtype.itemsize <= 4:
            dtype_str = "float32"
        else:
            dtype_str = "float64"
    
    return _compress_csc_to_cellz(
        nrows,
        ncols,
        indptr,
        indices,
        data,
        dtype_str,
        target_block_nnz=target_block_nnz,
        block_xz=block_xz,
        block_xz_level=block_xz_level,
        block_xz_min_bytes=block_xz_min_bytes,
        block_xz_min_ratio=block_xz_min_ratio,
    )


def decompress_cellz_to_csr(cellz_bytes):
    """
    Decompress CELLZ format data to a CSR sparse matrix.

    This function takes compressed CELLZ data and reconstructs the original sparse
    matrix in CSR (Compressed Sparse Row) format. The data type of the matrix is
    preserved from the compression metadata.

    Args:
        cellz_bytes (bytes): The compressed CELLZ data, typically read from a .cellz file.

    Returns:
        scipy.sparse.csr_matrix: The decompressed sparse matrix in CSR format with
            the original data type (int32, int64, float32, or float64).

    Raises:
        RuntimeError: If the CELLZ data is corrupted or invalid.

    Examples:
        >>> # Decompress from bytes
        >>> with open('data.cellz', 'rb') as f:
        ...     cellz_data = f.read()
        >>> matrix = decompress_cellz_to_csr(cellz_data)
        >>> print(matrix.shape, matrix.dtype)
    """
    result = _decompress_cellz_to_csr(cellz_bytes)
    
    shape = result["shape"]
    indptr = result["indptr"]
    indices = result["indices"]
    data = result["data"]
    dtype_str = result["dtype"]
    
    # Convert to the appropriate dtype
    if dtype_str == "int32":
        data = data.astype(np.int32)
    elif dtype_str == "int64":
        data = data.astype(np.int64)
    elif dtype_str == "float32":
        data = data.astype(np.float32)
    else:
        data = data.astype(np.float64)
    
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def compress_h5ad_to_cellz(
    h5ad_path,
    output_path=None,
    target_block_nnz=131072,
    block_xz=False,
    block_xz_level=9,
    block_xz_min_bytes=4096,
    block_xz_min_ratio=0.99,
):
    """
    Compress an AnnData .h5ad file to CELLZ format.

    This function reads an AnnData .h5ad file, extracts the expression matrix (X),
    and compresses it to the CELLZ format. This is particularly useful for reducing
    storage requirements of large single-cell genomics datasets.

    Args:
        h5ad_path (str): Path to the input .h5ad file.
        output_path (str, optional): Path for the output .cellz file. If None,
            automatically generates a path by replacing the .h5ad extension with .cellz.
            Default is None.
        target_block_nnz (int, optional): Target number of non-zero elements per block.
            Controls the granularity of compression blocks. Default is 131072 (128K).
        block_xz (bool, optional): Whether to enable block-level XZ compression.
            Can significantly reduce file size at the cost of compression speed.
            Default is False.
        block_xz_level (int, optional): XZ compression level (0-9), where 0 is fastest
            and 9 is best compression. Only used if block_xz is True. Default is 9.
        block_xz_min_bytes (int, optional): Minimum block size in bytes to apply XZ
            compression. Blocks smaller than this threshold will not be XZ-compressed.
            Default is 4096 (4KB).
        block_xz_min_ratio (float, optional): Minimum compression ratio threshold for
            applying XZ compression. If XZ doesn't achieve this ratio, the uncompressed
            block is used instead. Default is 0.99.

    Returns:
        str: Path to the output .cellz file.

    Note:
        This function only compresses the expression matrix (X). Other AnnData
        components (obs, var, obsm, varm, uns, etc.) are not preserved in the
        CELLZ format.

    Examples:
        >>> # Basic compression
        >>> output = compress_h5ad_to_cellz('data.h5ad')
        >>> print(f"Compressed to: {output}")
        >>> 
        >>> # With custom output path and XZ compression
        >>> output = compress_h5ad_to_cellz(
        ...     'data.h5ad',
        ...     output_path='compressed_data.cellz',
        ...     block_xz=True,
        ...     block_xz_level=6
        ... )
    """
    import anndata as ad
    import os
    
    if output_path is None:
        output_path = os.path.splitext(h5ad_path)[0] + ".cellz"
    
    # Read the h5ad file
    adata = ad.read_h5ad(h5ad_path)
    X = adata.X
    
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    elif not sp.isspmatrix_csr(X):
        X = X.tocsr()
    
    # Compress the matrix
    cellz_bytes = compress_csr_to_cellz(
        X,
        target_block_nnz=target_block_nnz,
        block_xz=block_xz,
        block_xz_level=block_xz_level,
        block_xz_min_bytes=block_xz_min_bytes,
        block_xz_min_ratio=block_xz_min_ratio,
    )
    
    # Write to file
    with open(output_path, "wb") as f:
        f.write(cellz_bytes)
    
    return output_path


def decompress_cellz_to_h5ad(cellz_path, output_path=None):
    """
    Decompress a CELLZ file to AnnData .h5ad format.

    This function reads a .cellz compressed file, decompresses the expression matrix,
    and creates an AnnData object containing only the expression matrix (X). The
    resulting .h5ad file can be used with standard single-cell analysis tools.

    Args:
        cellz_path (str): Path to the input .cellz file.
        output_path (str, optional): Path for the output .h5ad file. If None,
            automatically generates a path by replacing the .cellz extension with
            '_decompressed.h5ad'. Default is None.

    Returns:
        str: Path to the output .h5ad file.

    Note:
        The resulting AnnData object will only contain the expression matrix (X).
        Observation and variable annotations, embeddings, and other metadata are
        not stored in the CELLZ format and will not be present in the output.

    Examples:
        >>> # Basic decompression
        >>> output = decompress_cellz_to_h5ad('data.cellz')
        >>> print(f"Decompressed to: {output}")
        >>> 
        >>> # With custom output path
        >>> output = decompress_cellz_to_h5ad(
        ...     'compressed_data.cellz',
        ...     output_path='restored_data.h5ad'
        ... )
        >>> 
        >>> # Read the decompressed data
        >>> import anndata as ad
        >>> adata = ad.read_h5ad(output)
        >>> print(adata.shape)
    """
    import anndata as ad
    import os
    
    if output_path is None:
        output_path = os.path.splitext(cellz_path)[0] + "_decompressed.h5ad"
    
    # Read the cellz file
    with open(cellz_path, "rb") as f:
        cellz_bytes = f.read()
    
    # Decompress the data
    csr = decompress_cellz_to_csr(cellz_bytes)
    
    # Create AnnData object
    adata = ad.AnnData(X=csr)
    
    # Write to file
    adata.write_h5ad(output_path)
    
    return output_path

