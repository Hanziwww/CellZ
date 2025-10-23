"""
测试 CELLZ 压缩/解压缩的往返一致性
"""

import numpy as np
import scipy.sparse as sp
import cellz


def test_csr_roundtrip():
    """测试 CSR 矩阵的压缩和解压缩"""
    # 创建测试矩阵
    np.random.seed(42)
    nrows, ncols = 1000, 500
    density = 0.05
    
    # 创建稀疏矩阵（整数值，模拟 UMI counts）
    data = np.random.poisson(5, int(nrows * ncols * density))
    data = data[data > 0]  # 移除零值
    nnz = len(data)
    
    row_indices = np.random.randint(0, nrows, nnz)
    col_indices = np.random.randint(0, ncols, nnz)
    
    csr_original = sp.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(nrows, ncols)
    ).tocsr()
    
    # 压缩
    compressed = cellz.compress_csr_to_cellz(csr_original)
    print(f"原始 nnz: {csr_original.nnz}")
    print(f"压缩大小: {len(compressed)} bytes")
    print(f"平均每非零元素: {len(compressed) * 8 / csr_original.nnz:.2f} bits")
    
    # 解压缩
    csr_restored = cellz.decompress_cellz_to_csr(compressed)
    
    # 验证
    csr_original.sort_indices()
    csr_restored.sort_indices()
    
    assert csr_original.shape == csr_restored.shape
    assert csr_original.nnz == csr_restored.nnz
    assert np.array_equal(csr_original.indptr, csr_restored.indptr)
    assert np.array_equal(csr_original.indices, csr_restored.indices)
    assert np.allclose(csr_original.data, csr_restored.data)
    
    print("[OK] CSR roundtrip test passed")


def test_value_distribution():
    """测试不同值分布的编码"""
    np.random.seed(123)
    
    # 测试 1: 大量值为 1
    data1 = np.ones(1000, dtype=np.float64)
    indices1 = np.arange(1000)
    indptr1 = np.array([0, 1000])
    csr1 = sp.csr_matrix((data1, indices1, indptr1), shape=(1, 1000))
    
    compressed1 = cellz.compress_csr_to_cellz(csr1)
    restored1 = cellz.decompress_cellz_to_csr(compressed1)
    
    assert np.allclose(csr1.data, restored1.data)
    print(f"[OK] Value=1 distribution: {len(compressed1) * 8 / 1000:.2f} bits/nnz")
    
    # 测试 2: 小值分布
    data2 = np.random.randint(1, 10, 1000).astype(np.float64)
    indices2 = np.arange(1000)
    indptr2 = np.array([0, 1000])
    csr2 = sp.csr_matrix((data2, indices2, indptr2), shape=(1, 1000))
    
    compressed2 = cellz.compress_csr_to_cellz(csr2)
    restored2 = cellz.decompress_cellz_to_csr(compressed2)
    
    assert np.allclose(csr2.data, restored2.data)
    print(f"[OK] Small value distribution [1,10): {len(compressed2) * 8 / 1000:.2f} bits/nnz")
    
    # 测试 3: 混合分布
    data3 = np.concatenate([
        np.ones(500),
        np.random.randint(2, 15, 300),
        np.random.randint(15, 100, 200)
    ]).astype(np.float64)
    np.random.shuffle(data3)
    indices3 = np.arange(1000)
    indptr3 = np.array([0, 1000])
    csr3 = sp.csr_matrix((data3, indices3, indptr3), shape=(1, 1000))
    
    compressed3 = cellz.compress_csr_to_cellz(csr3)
    restored3 = cellz.decompress_cellz_to_csr(compressed3)
    
    assert np.allclose(csr3.data, restored3.data)
    print(f"[OK] Mixed distribution: {len(compressed3) * 8 / 1000:.2f} bits/nnz")


def test_edge_cases():
    """测试边界情况"""
    # 空矩阵
    csr_empty = sp.csr_matrix((10, 10))
    compressed_empty = cellz.compress_csr_to_cellz(csr_empty)
    restored_empty = cellz.decompress_cellz_to_csr(compressed_empty)
    assert restored_empty.nnz == 0
    print("[OK] Empty matrix test passed")
    
    # 单个非零元素
    csr_single = sp.csr_matrix(([42.0], ([5], [7])), shape=(10, 10))
    compressed_single = cellz.compress_csr_to_cellz(csr_single)
    restored_single = cellz.decompress_cellz_to_csr(compressed_single)
    assert restored_single.nnz == 1
    assert restored_single[5, 7] == 42.0
    print("[OK] Single element test passed")
    
    # 稠密列 (100 rows × 1 column, all non-zero)
    dense_col = np.random.randint(1, 20, 100).astype(np.float64)
    indices_dense = np.zeros(100, dtype=np.int32)  # All in column 0
    indptr_dense = np.arange(101, dtype=np.int64)  # Each row has 1 element
    csr_dense = sp.csr_matrix((dense_col, indices_dense, indptr_dense), shape=(100, 1))
    compressed_dense = cellz.compress_csr_to_cellz(csr_dense)
    restored_dense = cellz.decompress_cellz_to_csr(compressed_dense)
    assert np.allclose(csr_dense.data, restored_dense.data)
    print("[OK] Dense column test passed")


if __name__ == "__main__":
    test_csr_roundtrip()
    test_value_distribution()
    test_edge_cases()
    print("\n[OK] All tests passed!")

