# CELLZ - 高性能单细胞稀疏矩阵压缩器

Rust 实现的 CELLZ 压缩器，通过 PyO3 提供 Python 接口。

## 特性

- **自适应混合值编码**：针对单细胞 UMI counts 矩阵的零膨胀分布优化
- **动态阈值选择**：基于成本函数自动选择最优编码参数
- **Golomb-Rice 编码**：对位置间隙和大值进行高效编码
- **矩阵重排序**：按非零元素数量重排行列，提高压缩率
- **可选块级压缩**：支持 xz 压缩进一步减小文件大小
- **高性能**：Rust 实现，比纯 Python 快数倍

## 安装

### 从源码构建

需要 Rust 工具链和 Python 3.8+：

```bash
# 安装 maturin
pip install maturin

# 开发模式安装
maturin develop --release

# 或构建 wheel 包
maturin build --release
pip install target/wheels/cellz-*.whl
```

## 使用

### Python API

```python
import cellz
import scipy.sparse as sp

# 压缩 CSR 矩阵
csr = sp.random(10000, 2000, density=0.1, format='csr')
compressed_bytes = cellz.compress_csr_to_cellz(csr)

# 解压缩
decompressed_csr = cellz.decompress_cellz_to_csr(compressed_bytes)

# 压缩 .h5ad 文件
cellz.compress_h5ad_to_cellz("data.h5ad", "data.cellz")

# 解压缩 .cellz 文件
cellz.decompress_cellz_to_h5ad("data.cellz", "data_restored.h5ad")
```

### 高级选项

```python
# 启用块级 xz 压缩
compressed = cellz.compress_csr_to_cellz(
    csr,
    target_block_nnz=131072,      # 每块目标非零元素数
    block_xz=True,                 # 启用 xz 压缩
    block_xz_level=9,              # 压缩级别
    block_xz_min_bytes=4096,       # 最小字节数阈值
    block_xz_min_ratio=0.99        # 最小压缩比阈值
)
```

## 算法说明

CELLZ 使用多层编码策略：

1. **矩阵重排序**：按行列非零元素数降序排列，提高局部性
2. **列分块**：将矩阵分割为固定非零元素数的块
3. **位置编码**：使用 Golomb-Rice 编码压缩行索引间隙
4. **值编码**：自适应混合编码
   - 值为 1：1 bit
   - 小值（2 到阈值-1）：2 bit + 定长/截断二进制
   - 大值（≥阈值）：2 bit + Golomb-Rice
5. **可选块压缩**：xz/lzma 压缩位流

所有参数（阈值、k 参数、位数）均通过精确成本函数自动优化。

## 性能

相比纯 Python 实现：
- 压缩速度：约 3-5 倍提升
- 解压速度：约 4-6 倍提升
- 压缩率：相同（算法完全一致）

## 开发

运行测试：

```bash
# Rust 单元测试
cargo test

# Python 集成测试
maturin develop --release
pytest tests/
```

## 许可证

MIT License

