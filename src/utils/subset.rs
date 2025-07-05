use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use std::ptr;

pub fn subset_csr_internal<T>(
    row_offsets: Vec<usize>,
    col_indices_orig: Vec<usize>,
    values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    let nrows = row_offsets.len() - 1;
    let ncols = col_indices_orig.iter().max().map(|&x| x + 1).unwrap_or(0);
    if row_indices.is_empty() || col_indices.is_empty() {
        return create_empty_csr_matrix(row_indices.len(), col_indices.len());
    }

    if row_indices.len() == nrows && col_indices.len() == ncols {
        let pattern = unsafe {
            SparsityPattern::from_offset_and_indices_unchecked(
                nrows,
                ncols,
                row_offsets,
                col_indices_orig,
            )
        };
        return CsrMatrix::try_from_pattern_and_values(pattern, values)
            .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e));
    }

    if col_indices.len() == ncols {
        subset_rows_only_in_place(row_offsets, col_indices_orig, values, row_indices, ncols)
    } else if is_contiguous(col_indices) {
        subset_contiguous_columns_in_place(
            row_offsets,
            col_indices_orig,
            values,
            row_indices,
            col_indices,
        )
    } else {
        subset_sparse_columns_in_place(
            row_offsets,
            col_indices_orig,
            values,
            row_indices,
            col_indices,
        )
    }
}

pub fn subset_rows_only_in_place<T>(
    row_offsets: Vec<usize>,
    mut col_indices: Vec<usize>,
    mut values: Vec<T>,
    row_indices: &[usize],
    ncols: usize,
) -> anyhow::Result<CsrMatrix<T>> {
    if row_indices.is_empty() {
        return create_empty_csr_matrix(0, ncols);
    }

    let total_nnz: usize = row_indices
        .iter()
        .map(|&row_idx| row_offsets[row_idx + 1] - row_offsets[row_idx])
        .sum();

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    new_row_offsets.push(0);

    let mut write_pos = 0;

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];
        let row_nnz = end - start;

        if write_pos != start {
            unsafe {
                // Move column indices
                ptr::copy(
                    col_indices.as_ptr().add(start),
                    col_indices.as_mut_ptr().add(write_pos),
                    row_nnz,
                );

                // Move values
                ptr::copy(
                    values.as_ptr().add(start),
                    values.as_mut_ptr().add(write_pos),
                    row_nnz,
                );
            }
        }

        write_pos += row_nnz;
        new_row_offsets.push(write_pos);
    }

    col_indices.truncate(total_nnz);
    values.truncate(total_nnz);

    if col_indices.capacity() > total_nnz * 2 {
        col_indices.shrink_to_fit();
        values.shrink_to_fit();
    }

    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            ncols,
            new_row_offsets,
            col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(pattern, values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

pub fn subset_contiguous_columns_in_place<T>(
    row_offsets: Vec<usize>,
    mut col_indices: Vec<usize>,
    mut values: Vec<T>,
    row_indices: &[usize],
    col_range: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    if row_indices.is_empty() || col_range.is_empty() {
        return create_empty_csr_matrix(row_indices.len(), col_range.len());
    }

    let col_start = col_range[0];
    let col_end = col_range[col_range.len() - 1] + 1;

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    new_row_offsets.push(0);

    let mut write_pos = 0;

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        let mut row_write_pos = write_pos;
        for read_pos in start..end {
            let col = col_indices[read_pos];
            if col >= col_start && col < col_end {
                let new_col = col - col_start;

                col_indices[row_write_pos] = new_col;
                if row_write_pos != read_pos {
                    values[row_write_pos] = unsafe { ptr::read(&values[read_pos]) };
                }
                row_write_pos += 1;
            }
        }
        write_pos = row_write_pos;
        new_row_offsets.push(write_pos);
    }

    let final_nnz = write_pos;
    col_indices.truncate(final_nnz);
    values.truncate(final_nnz);

    if col_indices.capacity() > final_nnz * 2 {
        col_indices.shrink_to_fit();
        values.shrink_to_fit();
    }

    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            col_range.len(),
            new_row_offsets,
            col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(pattern, values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

pub fn subset_sparse_columns_in_place<T>(
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
    row_indices: &[usize],
    target_cols: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    if row_indices.is_empty() || target_cols.is_empty() {
        return create_empty_csr_matrix(row_indices.len(), target_cols.len());
    }

    let max_col = target_cols.iter().max().copied().unwrap_or(0);
    let density = target_cols.len() as f64 / (max_col + 1) as f64;

    if target_cols.len() < 512 && max_col < 8192 && density < 0.1 {
        subset_sparse_with_dense_lookup(
            row_offsets,
            col_indices,
            values,
            row_indices,
            target_cols,
            max_col,
        )
    } else if target_cols.len() < 2048 {
        subset_sparse_with_binary_search_optimized(
            row_offsets,
            col_indices,
            values,
            row_indices,
            target_cols,
        )
    } else {
        subset_sparse_with_hash_lookup(row_offsets, col_indices, values, row_indices, target_cols)
    }
}

fn subset_sparse_with_dense_lookup<T>(
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
    row_indices: &[usize],
    target_cols: &[usize],
    max_col: usize,
) -> anyhow::Result<CsrMatrix<T>> {
    const MAX_STACK_SIZE: usize = 1024;

    if max_col < MAX_STACK_SIZE {
        let mut lookup = [None; MAX_STACK_SIZE];
        for (new_idx, &old_idx) in target_cols.iter().enumerate() {
            if old_idx < MAX_STACK_SIZE {
                lookup[old_idx] = Some(new_idx);
            }
        }

        subset_with_stack_lookup(
            row_offsets,
            col_indices,
            values,
            row_indices,
            target_cols,
            &lookup[..=max_col],
        )
    } else {
        let mut lookup = vec![None; max_col + 1];
        for (new_idx, &old_idx) in target_cols.iter().enumerate() {
            lookup[old_idx] = Some(new_idx);
        }

        subset_with_heap_lookup(
            row_offsets,
            col_indices,
            values,
            row_indices,
            target_cols,
            &lookup,
        )
    }
}

fn subset_with_stack_lookup<T>(
    row_offsets: Vec<usize>, 
    mut col_indices: Vec<usize>,
    mut values: Vec<T>,
    row_indices: &[usize],
    target_cols: &[usize],
    lookup: &[Option<usize>],
) -> anyhow::Result<CsrMatrix<T>> {
    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    new_row_offsets.push(0);

    let mut write_pos = 0;

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        let mut row_write_pos = write_pos;
        for read_pos in start..end {
            let old_col = col_indices[read_pos];
            if old_col < lookup.len() {
                if let Some(new_col) = lookup[old_col] {
                    col_indices[row_write_pos] = new_col;
                    if row_write_pos != read_pos {
                        values[row_write_pos] = unsafe { ptr::read(&values[read_pos]) };
                    }
                    row_write_pos += 1;
                }
            }
        }
        write_pos = row_write_pos;
        new_row_offsets.push(write_pos);
    }

    finalize_vectors(
        col_indices,
        values,
        write_pos,
        target_cols.len(),
        new_row_offsets,
    )
}

fn subset_with_heap_lookup<T>(
    row_offsets: Vec<usize>,
    mut col_indices: Vec<usize>,
    mut values: Vec<T>,
    row_indices: &[usize],
    target_cols: &[usize],
    lookup: &[Option<usize>],
) -> anyhow::Result<CsrMatrix<T>> {
    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    new_row_offsets.push(0);

    let mut write_pos = 0;

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        let mut row_write_pos = write_pos;
        for read_pos in start..end {
            let old_col = col_indices[read_pos];
            if let Some(new_col) = lookup.get(old_col).and_then(|&x| x) {
                col_indices[row_write_pos] = new_col;
                if row_write_pos != read_pos {
                    values[row_write_pos] = unsafe { ptr::read(&values[read_pos]) };
                }
                row_write_pos += 1;
            }
        }
        write_pos = row_write_pos;
        new_row_offsets.push(write_pos);
    }

    finalize_vectors(
        col_indices,
        values,
        write_pos,
        target_cols.len(),
        new_row_offsets,
    )
}

fn subset_sparse_with_binary_search_optimized<T>(
    row_offsets: Vec<usize>, 
    mut col_indices: Vec<usize>,
    mut values: Vec<T>,
    row_indices: &[usize],
    target_cols: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    let mut sorted_mapping = Vec::with_capacity(target_cols.len());
    sorted_mapping.extend(
        target_cols
            .iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| (old_idx, new_idx)),
    );
    sorted_mapping.sort_unstable_by_key(|&(old_idx, _)| old_idx);

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    new_row_offsets.push(0);

    let mut write_pos = 0;

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        let mut row_write_pos = write_pos;
        for read_pos in start..end {
            let old_col = col_indices[read_pos];

            if let Ok(found_idx) =
                sorted_mapping.binary_search_by_key(&old_col, |&(old_idx, _)| old_idx)
            {
                let new_col = sorted_mapping[found_idx].1;
                col_indices[row_write_pos] = new_col;
                if row_write_pos != read_pos {
                    values[row_write_pos] = unsafe { ptr::read(&values[read_pos]) };
                }
                row_write_pos += 1;
            }
        }
        write_pos = row_write_pos;
        new_row_offsets.push(write_pos);
    }

    finalize_vectors(
        col_indices,
        values,
        write_pos,
        target_cols.len(),
        new_row_offsets,
    )
}

fn subset_sparse_with_hash_lookup<T>(
    row_offsets: Vec<usize>, 
    mut col_indices: Vec<usize>,
    mut values: Vec<T>,
    row_indices: &[usize],
    target_cols: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    use std::collections::HashMap;

    let col_map: HashMap<usize, usize> = target_cols
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx, new_idx))
        .collect();

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    new_row_offsets.push(0);

    let mut write_pos = 0;

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        let mut row_write_pos = write_pos;
        for read_pos in start..end {
            let old_col = col_indices[read_pos];

            if let Some(&new_col) = col_map.get(&old_col) {
                col_indices[row_write_pos] = new_col;
                if row_write_pos != read_pos {
                    values[row_write_pos] = unsafe { ptr::read(&values[read_pos]) };
                }
                row_write_pos += 1;
            }
        }
        write_pos = row_write_pos;
        new_row_offsets.push(write_pos);
    }

    finalize_vectors(
        col_indices,
        values,
        write_pos,
        target_cols.len(),
        new_row_offsets,
    )
}

fn finalize_vectors<T>(
    mut col_indices: Vec<usize>,
    mut values: Vec<T>,
    final_nnz: usize,
    ncols: usize,
    row_offsets: Vec<usize>,
) -> anyhow::Result<CsrMatrix<T>> {
    col_indices.truncate(final_nnz);
    values.truncate(final_nnz);

    if col_indices.capacity() > final_nnz * 2 {
        col_indices.shrink_to_fit();
        values.shrink_to_fit();
    }

    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_offsets.len() - 1,
            ncols,
            row_offsets,
            col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(pattern, values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

pub fn is_contiguous(indices: &[usize]) -> bool {
    if indices.len() <= 1 {
        return true;
    }
    indices.windows(2).all(|w| w[1] == w[0] + 1)
}

fn create_empty_csr_matrix<T>(nrows: usize, ncols: usize) -> anyhow::Result<CsrMatrix<T>> {
    let row_offsets = vec![0; nrows + 1];
    let col_indices: Vec<usize> = Vec::new();
    let values: Vec<T> = Vec::new();

    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(nrows, ncols, row_offsets, col_indices)
    };

    CsrMatrix::try_from_pattern_and_values(pattern, values)
        .map_err(|e| anyhow::anyhow!("Failed to create empty CSR matrix: {:?}", e))
}
