use crate::bitstream::BitReader;
use crate::hybrid_value::AdaptiveHybridValueDecoder;
use crate::varint::read_varint;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::io::{Cursor, Read};

/// dtype 解码
fn numpy_dtype_from_code(code: u8) -> &'static str {
    match code {
        1 => "int32",
        2 => "int64",
        3 => "float32",
        4 => "float64",
        _ => "float64",
    }
}

/// 解压缩 CELLZ 字节流到 CSR 矩阵
#[pyfunction]
pub fn decompress_cellz_to_csr(py: Python, blob: &[u8]) -> PyResult<PyObject> {
    let mut bio = Cursor::new(blob);

    // 读取 magic
    let mut magic = [0u8; 5];
    bio.read_exact(&mut magic)?;
    if &magic != b"CELLZ" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Invalid CELLZ file: magic mismatch",
        ));
    }

    // 读取 version
    let mut version_bytes = [0u8; 4];
    bio.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported version: {} (only v4 supported)",
            version
        )));
    }

    // 读取 dims
    let mut dims_bytes = [0u8; 16];
    bio.read_exact(&mut dims_bytes)?;
    let nrows = u64::from_le_bytes(dims_bytes[0..8].try_into().unwrap()) as usize;
    let ncols = u64::from_le_bytes(dims_bytes[8..16].try_into().unwrap()) as usize;

    // 读取 dtype
    let mut dt_code_bytes = [0u8; 1];
    bio.read_exact(&mut dt_code_bytes)?;
    let dt_code = dt_code_bytes[0];
    let data_dtype = numpy_dtype_from_code(dt_code);

    // 读取重排序表
    let n_cell_order = read_varint(&mut bio)? as usize;
    let mut cell_order = Vec::with_capacity(n_cell_order);
    for _ in 0..n_cell_order {
        cell_order.push(read_varint(&mut bio)? as i64);
    }

    let n_gene_order = read_varint(&mut bio)? as usize;
    let mut gene_order = Vec::with_capacity(n_gene_order);
    for _ in 0..n_gene_order {
        gene_order.push(read_varint(&mut bio)? as i64);
    }

    // 读取块数量
    let n_plans = read_varint(&mut bio)? as usize;

    // 收集每列 nnz 和块元数据
    let mut col_nnz = vec![0i64; ncols];
    let mut blocks_meta = Vec::new();

    for _ in 0..n_plans {
        let start = read_varint(&mut bio)? as usize;
        let block_ncols = read_varint(&mut bio)? as usize;
        let _block_nnz = read_varint(&mut bio)?;

        let mut cols_meta = Vec::new();
        for c in 0..block_ncols {
            let nnz_col = read_varint(&mut bio)? as usize;
            if nnz_col == 0 {
                cols_meta.push((0, 0u32));
                col_nnz[start + c] = 0;
                continue;
            }
            let k_pos = read_varint(&mut bio)? as u32;
            cols_meta.push((nnz_col, k_pos));
            col_nnz[start + c] = nnz_col as i64;
        }

        // 读取流元数据
        let pos_enc = read_varint(&mut bio)?;
        let pos_len = read_varint(&mut bio)? as usize;
        let val_enc = read_varint(&mut bio)?;
        let val_len = read_varint(&mut bio)? as usize;

        let mut pos_bytes = vec![0u8; pos_len];
        bio.read_exact(&mut pos_bytes)?;
        let mut val_bytes = vec![0u8; val_len];
        bio.read_exact(&mut val_bytes)?;

        // 解压缩（如果需要）
        if pos_enc == 1 {
            #[cfg(feature = "xz")]
            {
                use xz2::read::XzDecoder;
                let mut decoder = XzDecoder::new(&pos_bytes[..]);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                pos_bytes = decompressed;
            }
            #[cfg(not(feature = "xz"))]
            {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "xz decompression required but not available",
                ));
            }
        }

        if val_enc == 1 {
            #[cfg(feature = "xz")]
            {
                use xz2::read::XzDecoder;
                let mut decoder = XzDecoder::new(&val_bytes[..]);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                val_bytes = decompressed;
            }
            #[cfg(not(feature = "xz"))]
            {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "xz decompression required but not available",
                ));
            }
        }

        blocks_meta.push((start, cols_meta, pos_bytes, val_bytes));
    }

    // 构建 CSC 数组
    let mut indptr = vec![0i64; ncols + 1];
    for j in 0..ncols {
        indptr[j + 1] = indptr[j] + col_nnz[j];
    }
    let total_nnz = indptr[ncols] as usize;

    let mut indices = vec![0i32; total_nnz];
    let mut data = vec![0.0f64; total_nnz];
    let mut write_pos = indptr[..ncols].to_vec();

    // 解码流并填充数组
    for (start, cols_meta, pos_bytes, val_bytes) in blocks_meta {
        let mut pos_br = BitReader::new(pos_bytes);
        let mut val_br = BitReader::new(val_bytes);
        let mut hybrid_decoder = AdaptiveHybridValueDecoder::new();

        for (c_offset, (nnz_col, k_pos)) in cols_meta.iter().enumerate() {
            let col_idx = start + c_offset;
            if *nnz_col == 0 {
                continue;
            }

            // 解码位置
            let mut prev = -1i64;
            for _ in 0..*nnz_col {
                let q = pos_br.read_unary_ones_then_zero_count();
                let r = if *k_pos > 0 {
                    pos_br.read_bits(*k_pos)
                } else {
                    0
                };
                let gap = ((q << k_pos) | r) as i64;
                prev = prev + gap + 1;
                let pos = write_pos[col_idx] as usize;
                indices[pos] = prev as i32;
                write_pos[col_idx] += 1;
            }

            // 解码值
            let start_pos = indptr[col_idx] as usize;
            hybrid_decoder.read_header(&mut val_br);
            for i in 0..*nnz_col {
                let vprime = hybrid_decoder.decode_value(&mut val_br);
                let val = vprime + 1;
                data[start_pos + i] = val as f64;
            }
        }
    }

    // 构建 CSC
    let csc_nrows = nrows;
    let csc_ncols = ncols;
    let csc_indptr = indptr;
    let csc_indices = indices;
    let csc_data = data;

    // 恢复原始顺序
    // cell_order[new_i] = old_i (正向排列)
    // permute_csc 会在内部计算逆排列
    let csc_restored = permute_csc(
        csc_nrows,
        csc_ncols,
        &csc_indptr,
        &csc_indices,
        &csc_data,
        &cell_order,
        &gene_order,
    );

    // 转换为 CSR
    let (csr_indptr, csr_indices, csr_data) = csc_to_csr_rust(
        csc_restored.0,
        csc_restored.1,
        &csc_restored.2,
        &csc_restored.3,
        &csc_restored.4,
    );

    // 返回 Python 字典
    let result = PyDict::new(py);
    result.set_item("shape", (nrows, ncols))?;
    result.set_item("indptr", PyArray1::from_vec(py, csr_indptr))?;
    result.set_item("indices", PyArray1::from_vec(py, csr_indices))?;
    result.set_item("data", PyArray1::from_vec(py, csr_data))?;
    result.set_item("dtype", data_dtype)?;

    Ok(result.into())
}

fn permute_csc(
    nrows: usize,
    ncols: usize,
    indptr: &[i64],
    indices: &[i32],
    data: &[f64],
    row_order: &[i64],  // row_order[new_i] = old_i (正向排列)
    col_order: &[i64],  // col_order[new_j] = old_j (正向排列)
) -> (usize, usize, Vec<i64>, Vec<i32>, Vec<f64>) {
    // Python实现: csc[inv_cell, :][:, inv_gene]
    // 其中 inv_cell[old_i] = new_i, inv_gene[old_j] = new_j
    // 
    // 这等价于：
    // 1. 按照原始列顺序遍历 (old_j = 0..ncols)
    // 2. 对于每个old_j，找到对应的重排序后的列 new_j (通过inv_gene[old_j])
    // 3. 从重排序矩阵的列new_j中取数据
    // 4. 对于列中的每个行索引new_i，映射回old_i (通过cell_order[new_i])
    
    // 计算逆排列: inv_col[old_j] = new_j
    let mut inv_col = vec![0usize; ncols];
    for new_j in 0..ncols {
        let old_j = col_order[new_j] as usize;
        inv_col[old_j] = new_j;
    }
    
    let total_nnz = data.len();
    let mut new_indptr = vec![0i64; ncols + 1];
    let mut new_indices = vec![0i32; total_nnz];
    let mut new_data = vec![0.0f64; total_nnz];

    let mut pos = 0usize;
    for old_j in 0..ncols {
        // 找到对应的重排序列
        let new_j = inv_col[old_j];
        let start = indptr[new_j] as usize;
        let end = indptr[new_j + 1] as usize;
        let count = end - start;
        new_indptr[old_j + 1] = new_indptr[old_j] + count as i64;
        
        // 复制数据并重映射行索引
        for i in 0..count {
            let new_row_idx = indices[start + i] as usize;
            // 使用正向排列映射: cell_order[new_i] = old_i
            let old_row_idx = row_order[new_row_idx];
            new_indices[pos + i] = old_row_idx as i32;
            new_data[pos + i] = data[start + i];
        }
        pos += count;
    }

    (nrows, ncols, new_indptr, new_indices, new_data)
}

fn csc_to_csr_rust(
    nrows: usize,
    ncols: usize,
    indptr: &[i64],
    indices: &[i32],
    data: &[f64],
) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let nnz = data.len();

    // 计算每行的 nnz
    let mut row_counts = vec![0usize; nrows];
    for &row in indices {
        row_counts[row as usize] += 1;
    }

    // 构建 row_indptr
    let mut row_indptr = vec![0i64; nrows + 1];
    for i in 0..nrows {
        row_indptr[i + 1] = row_indptr[i] + row_counts[i] as i64;
    }

    // 填充数据
    let mut row_indices = vec![0i32; nnz];
    let mut row_data = vec![0.0f64; nnz];
    let mut row_write_pos = row_indptr[..nrows].to_vec();

    for j in 0..ncols {
        let start = indptr[j] as usize;
        let end = indptr[j + 1] as usize;
        for k in start..end {
            let i = indices[k] as usize;
            let pos = row_write_pos[i] as usize;
            row_indices[pos] = j as i32;
            row_data[pos] = data[k];
            row_write_pos[i] += 1;
        }
    }

    (row_indptr, row_indices, row_data)
}

