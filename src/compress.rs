use crate::bitstream::BitWriter;
use crate::golomb_rice::{encode_golomb_rice_numbers, select_optimal_gr_k};
use crate::hybrid_value::AdaptiveHybridValueEncoder;
use crate::matrix::{plan_blocks_by_target_nnz, reorder_matrix_by_nnz, CscMatrix};
use crate::varint::write_varint;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::{Cursor, Write};
use rayon::prelude::*;

/// dtype 编码
fn dtype_code_from_str(dtype: &str) -> u8 {
    match dtype {
        "int32" => 1,
        "int64" => 2,
        "float32" => 3,
        "float64" => 4,
        _ => 0,
    }
}

/// round-half-to-even (banker's rounding), to match numpy.rint
fn round_ties_to_even(v: f64) -> i64 {
    if !v.is_finite() {
        return v.round() as i64;
    }
    let floor = v.floor();
    let frac = v - floor;
    let rounded = if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1.0
    } else {
        // exactly halfway; choose the even integer
        let floor_i = floor as i64;
        if floor_i % 2 == 0 { floor } else { floor + 1.0 }
    };
    rounded as i64
}

/// 压缩 CSC 矩阵到 CELLZ 字节流
#[pyfunction]
#[pyo3(signature = (
    nrows,
    ncols,
    indptr,
    indices,
    data,
    dtype_str,
    target_block_nnz=131072,
    block_xz=false,
    block_xz_level=9,
    block_xz_min_bytes=4096,
    block_xz_min_ratio=0.99
))]
pub fn compress_csc_to_cellz(
    py: Python,
    nrows: usize,
    ncols: usize,
    indptr: PyReadonlyArrayDyn<i64>,
    indices: PyReadonlyArrayDyn<i32>,
    data: PyReadonlyArrayDyn<f64>,
    dtype_str: &str,
    target_block_nnz: usize,
    block_xz: bool,
    block_xz_level: i32,
    block_xz_min_bytes: usize,
    block_xz_min_ratio: f64,
) -> PyResult<Py<PyBytes>> {
    let indptr = indptr.as_slice()?;
    let indices = indices.as_slice()?;
    let data = data.as_slice()?;

    // 重计算部分释放 GIL
    let bytes = py.allow_threads(|| {
        // 转换为 CSR 格式以进行重排序
        let csc = CscMatrix {
            nrows,
            ncols,
            indptr: indptr.to_vec(),
            indices: indices.to_vec(),
            data: data.to_vec(),
        };

        // 从 CSC 转为 CSR
        let (row_indptr, row_indices, row_data) = crate::matrix::csc_to_csr(&csc);

        // 重排序
        let (csc_reordered, cell_order, gene_order) = reorder_matrix_by_nnz(
            nrows,
            ncols,
            &row_indptr,
            &row_indices,
            &row_data,
        );

        // 压缩
        compress_csc_internal(
            &csc_reordered,
            &cell_order,
            &gene_order,
            dtype_str,
            target_block_nnz,
            block_xz,
            block_xz_level,
            block_xz_min_bytes,
            block_xz_min_ratio,
        )
    })?;

    Ok(PyBytes::new(py, &bytes).into())
}

fn compress_csc_internal(
    csc: &CscMatrix,
    cell_order: &[i64],
    gene_order: &[i64],
    dtype_str: &str,
    target_block_nnz: usize,
    block_xz: bool,
    block_xz_level: i32,
    block_xz_min_bytes: usize,
    block_xz_min_ratio: f64,
) -> PyResult<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());

    // Magic + version
    out.write_all(b"CELLZ")?;
    out.write_all(&4u32.to_le_bytes())?; // version 4

    // dims
    out.write_all(&(csc.nrows as u64).to_le_bytes())?;
    out.write_all(&(csc.ncols as u64).to_le_bytes())?;

    // dtype code
    let dt_code = dtype_code_from_str(dtype_str);
    out.write_all(&[dt_code])?;

    // 写入重排序表
    write_varint(&mut out, cell_order.len() as u64)?;
    for &v in cell_order {
        write_varint(&mut out, v as u64)?;
    }
    write_varint(&mut out, gene_order.len() as u64)?;
    for &v in gene_order {
        write_varint(&mut out, v as u64)?;
    }

    // 规划块
    let plans = plan_blocks_by_target_nnz(&csc.indptr, target_block_nnz);
    write_varint(&mut out, plans.len() as u64)?;

    struct EncodedBlock { order: usize, bytes: Vec<u8> }

    let encoded_blocks: Vec<EncodedBlock> = plans
        .par_iter()
        .enumerate()
        .map(|(bi, plan)| -> PyResult<EncodedBlock> {
            let start = plan.start_col;
            let end = plan.end_col;
            let block_ncols = end - start;

            let mut blk = Cursor::new(Vec::new());
            write_varint(&mut blk, start as u64)?;
            write_varint(&mut blk, block_ncols as u64)?;
            write_varint(&mut blk, plan.nnz as u64)?;

            let mut pos_bw = BitWriter::new();
            let mut val_bw = BitWriter::new();
            let mut hybrid_encoder = AdaptiveHybridValueEncoder::new();

            for j in start..end {
                let s = csc.indptr[j] as usize;
                let e = csc.indptr[j + 1] as usize;
                let col_indices = &csc.indices[s..e];
                let col_values = &csc.data[s..e];
                let nnz_col = e - s;

                if nnz_col == 0 {
                    write_varint(&mut blk, 0)?;
                    continue;
                }

                write_varint(&mut blk, nnz_col as u64)?;

                // 计算 gaps
                let mut gaps = Vec::with_capacity(nnz_col);
                let mut prev = -1i64;
                for &r in col_indices {
                    gaps.push(r as i64 - prev - 1);
                    prev = r as i64;
                }

                // 位置编码
                let k_pos = select_optimal_gr_k(&gaps, Some(30));
                write_varint(&mut blk, k_pos as u64)?;
                encode_golomb_rice_numbers(&gaps, k_pos, &mut pos_bw);

                // 值编码：v' = v - 1
                let vprime: Vec<i64> = col_values
                    .iter()
                    .map(|&v| round_ties_to_even(v) - 1)
                    .collect();
                hybrid_encoder.encode_values(&vprime, &mut val_bw);
            }

            // 完成流
            let mut pos_bytes = pos_bw.finish();
            let mut val_bytes = val_bw.finish();

            // 可选的块级 xz 压缩
            let mut enc_pos_id = 0u64;
            let mut enc_val_id = 0u64;

            if block_xz {
                #[cfg(feature = "xz")]
                {
                    use xz2::write::XzEncoder;
                    // Clamp level to [0,9]
                    let lvl: u32 = if block_xz_level < 0 {
                        0
                    } else if block_xz_level > 9 {
                        9
                    } else {
                        block_xz_level as u32
                    };

                    if pos_bytes.len() >= block_xz_min_bytes {
                        let mut encoder = XzEncoder::new(Vec::new(), lvl);
                        encoder.write_all(&pos_bytes)?;
                        let compressed = encoder.finish()?;
                        if (compressed.len() as f64) < (pos_bytes.len() as f64) * block_xz_min_ratio {
                            enc_pos_id = 1;
                            pos_bytes = compressed;
                        }
                    }

                    if val_bytes.len() >= block_xz_min_bytes {
                        let mut encoder = XzEncoder::new(Vec::new(), lvl);
                        encoder.write_all(&val_bytes)?;
                        let compressed = encoder.finish()?;
                        if (compressed.len() as f64) < (val_bytes.len() as f64) * block_xz_min_ratio {
                            enc_val_id = 1;
                            val_bytes = compressed;
                        }
                    }
                }
            }

            // 写入流头和负载到块缓冲
            write_varint(&mut blk, enc_pos_id)?;
            write_varint(&mut blk, pos_bytes.len() as u64)?;
            write_varint(&mut blk, enc_val_id)?;
            write_varint(&mut blk, val_bytes.len() as u64)?;
            blk.write_all(&pos_bytes)?;
            blk.write_all(&val_bytes)?;

            Ok(EncodedBlock { order: bi, bytes: blk.into_inner() })
        })
        .collect::<PyResult<Vec<_>>>()?;

    // 按原顺序写出块
    let mut blocks_sorted = encoded_blocks;
    blocks_sorted.sort_by_key(|b| b.order);
    for b in blocks_sorted {
        out.write_all(&b.bytes)?;
    }

    Ok(out.into_inner())
}
