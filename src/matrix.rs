/// CSC 稀疏矩阵表示
#[derive(Clone)]
pub struct CscMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<f64>,
}

/// 块规划
#[derive(Debug, Clone)]
pub struct BlockPlan {
    pub start_col: usize,
    pub end_col: usize,
    pub nnz: usize,
}

/// 根据目标非零元素数量规划块
pub fn plan_blocks_by_target_nnz(indptr: &[i64], target_block_nnz: usize) -> Vec<BlockPlan> {
    let mut plans = Vec::new();
    let ncols = indptr.len() - 1;
    let mut acc_nnz = 0usize;
    let mut block_start = 0usize;

    for j in 0..ncols {
        let col_nnz = (indptr[j + 1] - indptr[j]) as usize;
        if acc_nnz == 0 {
            block_start = j;
        }
        acc_nnz += col_nnz;
        if acc_nnz >= target_block_nnz && j >= block_start {
            plans.push(BlockPlan {
                start_col: block_start,
                end_col: j + 1,
                nnz: acc_nnz,
            });
            acc_nnz = 0;
        }
    }
    if acc_nnz > 0 {
        plans.push(BlockPlan {
            start_col: block_start,
            end_col: ncols,
            nnz: acc_nnz,
        });
    }
    plans
}

/// 按 nnz 重新排序矩阵（从 CSR 转为重排序的 CSC）
pub fn reorder_matrix_by_nnz(
    nrows: usize,
    ncols: usize,
    row_indptr: &[i64],
    row_indices: &[i32],
    row_data: &[f64],
) -> (CscMatrix, Vec<i64>, Vec<i64>) {
    // 计算每行的 nnz
    let row_nnz: Vec<usize> = (0..nrows)
        .map(|i| (row_indptr[i + 1] - row_indptr[i]) as usize)
        .collect();

    // 按 nnz 降序排序行
    let mut cell_order: Vec<usize> = (0..nrows).collect();
    cell_order.sort_by(|&a, &b| row_nnz[b].cmp(&row_nnz[a]));

    // 构建重排序后的 CSR
    let total_nnz: usize = row_nnz.iter().sum();
    let mut new_row_indptr = vec![0i64; nrows + 1];
    let mut new_row_indices = vec![0i32; total_nnz];
    let mut new_row_data = vec![0.0f64; total_nnz];

    let mut pos = 0usize;
    for (new_i, &old_i) in cell_order.iter().enumerate() {
        let start = row_indptr[old_i] as usize;
        let end = row_indptr[old_i + 1] as usize;
        let count = end - start;
        new_row_indptr[new_i + 1] = new_row_indptr[new_i] + count as i64;
        new_row_indices[pos..pos + count].copy_from_slice(&row_indices[start..end]);
        new_row_data[pos..pos + count].copy_from_slice(&row_data[start..end]);
        pos += count;
    }

    // 转换为 CSC
    let csc = csr_to_csc(
        nrows,
        ncols,
        &new_row_indptr,
        &new_row_indices,
        &new_row_data,
    );

    // 计算每列的 nnz 并排序列
    let col_nnz: Vec<usize> = (0..ncols)
        .map(|j| (csc.indptr[j + 1] - csc.indptr[j]) as usize)
        .collect();

    let mut gene_order: Vec<usize> = (0..ncols).collect();
    gene_order.sort_by(|&a, &b| col_nnz[b].cmp(&col_nnz[a]));

    // 重排序列
    let mut final_indptr = vec![0i64; ncols + 1];
    let mut final_indices = vec![0i32; total_nnz];
    let mut final_data = vec![0.0f64; total_nnz];

    let mut pos = 0usize;
    for (new_j, &old_j) in gene_order.iter().enumerate() {
        let start = csc.indptr[old_j] as usize;
        let end = csc.indptr[old_j + 1] as usize;
        let count = end - start;
        final_indptr[new_j + 1] = final_indptr[new_j] + count as i64;
        final_indices[pos..pos + count].copy_from_slice(&csc.indices[start..end]);
        final_data[pos..pos + count].copy_from_slice(&csc.data[start..end]);
        pos += count;
    }

    let final_csc = CscMatrix {
        nrows,
        ncols,
        indptr: final_indptr,
        indices: final_indices,
        data: final_data,
    };

    let cell_order_i64: Vec<i64> = cell_order.iter().map(|&x| x as i64).collect();
    let gene_order_i64: Vec<i64> = gene_order.iter().map(|&x| x as i64).collect();

    (final_csc, cell_order_i64, gene_order_i64)
}

/// CSR 转 CSC
fn csr_to_csc(
    nrows: usize,
    ncols: usize,
    row_indptr: &[i64],
    row_indices: &[i32],
    row_data: &[f64],
) -> CscMatrix {
    let nnz = row_data.len();
    
    // 计算每列的 nnz
    let mut col_counts = vec![0usize; ncols];
    for &col in row_indices {
        col_counts[col as usize] += 1;
    }

    // 构建 col_indptr
    let mut col_indptr = vec![0i64; ncols + 1];
    for j in 0..ncols {
        col_indptr[j + 1] = col_indptr[j] + col_counts[j] as i64;
    }

    // 填充数据
    let mut col_indices = vec![0i32; nnz];
    let mut col_data = vec![0.0f64; nnz];
    let mut col_write_pos = col_indptr[..ncols].to_vec();

    for i in 0..nrows {
        let start = row_indptr[i] as usize;
        let end = row_indptr[i + 1] as usize;
        for k in start..end {
            let j = row_indices[k] as usize;
            let pos = col_write_pos[j] as usize;
            col_indices[pos] = i as i32;
            col_data[pos] = row_data[k];
            col_write_pos[j] += 1;
        }
    }

    CscMatrix {
        nrows,
        ncols,
        indptr: col_indptr,
        indices: col_indices,
        data: col_data,
    }
}

/// CSC 转 CSR
pub fn csc_to_csr(csc: &CscMatrix) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let nrows = csc.nrows;
    let ncols = csc.ncols;
    let nnz = csc.data.len();

    // 计算每行的 nnz
    let mut row_counts = vec![0usize; nrows];
    for &row in &csc.indices {
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
        let start = csc.indptr[j] as usize;
        let end = csc.indptr[j + 1] as usize;
        for k in start..end {
            let i = csc.indices[k] as usize;
            let pos = row_write_pos[i] as usize;
            row_indices[pos] = j as i32;
            row_data[pos] = csc.data[k];
            row_write_pos[i] += 1;
        }
    }

    (row_indptr, row_indices, row_data)
}
