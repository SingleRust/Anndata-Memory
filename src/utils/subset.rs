use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use std::ptr;


/// Internal optimized CSR matrix subset that minimizes data copying
pub fn subset_csr_internal<T>(
    row_offsets: Vec<usize>,
    col_indices_orig: Vec<usize>, 
    values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    let nrows = row_offsets.len() - 1;
    let ncols = col_indices_orig.iter().max().map(|&x| x + 1).unwrap_or(0);
    
    // Fast path: no subsetting needed
    if row_indices.len() == nrows && col_indices.len() == ncols {
        let pattern = unsafe {
            SparsityPattern::from_offset_and_indices_unchecked(nrows, ncols, row_offsets, col_indices_orig)
        };
        return CsrMatrix::try_from_pattern_and_values(pattern, values)
            .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e));
    }
    
    // Determine subset strategy based on column pattern
    if col_indices.len() == ncols {
        // Only row subsetting - most efficient case
        subset_rows_only(&row_offsets, &col_indices_orig, values, row_indices, ncols)
    } else if is_contiguous(col_indices) {
        // Contiguous column range - second most efficient
        subset_with_contiguous_columns(&row_offsets, &col_indices_orig, values, row_indices, col_indices)
    } else {
        // Sparse column selection - requires more work
        subset_with_sparse_columns(&row_offsets, &col_indices_orig, values, row_indices, col_indices)
    }
}

/// Most efficient case: only subset rows, keep all columns
pub fn subset_rows_only<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    mut values: Vec<T>,
    row_indices: &[usize],
    ncols: usize,
) -> anyhow::Result<CsrMatrix<T>> {
    // Calculate exact output size
    let total_nnz: usize = row_indices.iter()
        .map(|&row_idx| row_offsets[row_idx + 1] - row_offsets[row_idx])
        .sum();
    
    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::with_capacity(total_nnz);
    new_row_offsets.push(0);

    // Strategy: Move values in place to avoid copying
    // We'll reuse the original values vector by swapping elements
    let mut write_pos = 0;
    
    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];
        let row_nnz = end - start;
        
        // Copy column indices (unavoidable, but small)
        new_col_indices.extend_from_slice(&col_indices_orig[start..end]);
        
        // Move values efficiently: swap needed values to the front
        for i in 0..row_nnz {
            if write_pos + i != start + i {
                values.swap(write_pos + i, start + i);
            }
        }
        write_pos += row_nnz;
        new_row_offsets.push(new_col_indices.len());
    }
    
    // Truncate and reuse the original values vector
    values.truncate(total_nnz);
    
    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            ncols,
            new_row_offsets,
            new_col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(pattern, values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

/// Second most efficient: contiguous column range
pub fn subset_with_contiguous_columns<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    mut values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    let col_start = col_indices[0];
    let col_end = col_indices[col_indices.len() - 1] + 1;
    
    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::new();
    let mut write_pos = 0;
    new_row_offsets.push(0);

    // Strategy: reorder values in place using swaps
    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];
        
        let mut row_write_pos = write_pos;
        for idx in start..end {
            let col = col_indices_orig[idx];
            if col >= col_start && col < col_end {
                let new_col = col - col_start;
                new_col_indices.push(new_col);
                
                // Move value to correct position if needed
                if row_write_pos != idx {
                    values.swap(row_write_pos, idx);
                }
                row_write_pos += 1;
            }
        }
        write_pos = row_write_pos;
        new_row_offsets.push(new_col_indices.len());
    }
    
    // Truncate to actual size needed - reuses original vector
    values.truncate(write_pos);
    
    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            col_indices.len(),
            new_row_offsets,
            new_col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(pattern, values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

/// Sparse column selection with memory-efficient approach
pub fn subset_with_sparse_columns<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    // Choose strategy based on column set size
    let max_col = col_indices.iter().max().copied().unwrap_or(0);
    let use_lookup_table = col_indices.len() < 1000 && max_col < 10000;
    
    if use_lookup_table {
        subset_sparse_with_lookup(row_offsets, col_indices_orig, values, row_indices, col_indices, max_col)
    } else {
        subset_sparse_with_binary_search(row_offsets, col_indices_orig, values, row_indices, col_indices)
    }
}

/// Sparse subset using lookup table for small column sets
fn subset_sparse_with_lookup<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    mut values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
    max_col: usize,
) -> anyhow::Result<CsrMatrix<T>> {
    // Create O(1) lookup table
    let mut col_mapping = vec![None; max_col + 1];
    for (new_idx, &old_idx) in col_indices.iter().enumerate() {
        col_mapping[old_idx] = Some(new_idx);
    }

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::new();
    let mut write_pos = 0;
    new_row_offsets.push(0);

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];
        
        let mut row_write_pos = write_pos;
        for idx in start..end {
            let old_col = col_indices_orig[idx];
            if old_col <= max_col {
                if let Some(new_col) = col_mapping[old_col] {
                    new_col_indices.push(new_col);
                    
                    // Swap value to correct position
                    if row_write_pos != idx {
                        values.swap(row_write_pos, idx);
                    }
                    row_write_pos += 1;
                }
            }
        }
        write_pos = row_write_pos;
        new_row_offsets.push(new_col_indices.len());
    }
    
    // Reuse original vector, just truncated
    values.truncate(write_pos);
    
    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            col_indices.len(),
            new_row_offsets,
            new_col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(pattern, values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

/// Sparse subset using binary search for large column sets
fn subset_sparse_with_binary_search<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    mut values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    // Sort columns with their new indices for binary search
    let mut sorted_col_with_new_idx: Vec<(usize, usize)> = col_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx, new_idx))
        .collect();
    sorted_col_with_new_idx.sort_unstable_by_key(|&(old_idx, _)| old_idx);

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::new();
    let mut write_pos = 0;
    new_row_offsets.push(0);

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];
        
        let mut row_write_pos = write_pos;
        for idx in start..end {
            let old_col = col_indices_orig[idx];
            
            if let Ok(found_idx) = sorted_col_with_new_idx.binary_search_by_key(&old_col, |&(old_idx, _)| old_idx) {
                let new_col = sorted_col_with_new_idx[found_idx].1;
                new_col_indices.push(new_col);
                
                // Swap value to correct position
                if row_write_pos != idx {
                    values.swap(row_write_pos, idx);
                }
                row_write_pos += 1;
            }
        }
        write_pos = row_write_pos;
        new_row_offsets.push(new_col_indices.len());
    }
    
    // Reuse original vector, just truncated
    values.truncate(write_pos);
    
    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            col_indices.len(),
            new_row_offsets,
            new_col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(pattern, values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

/// Check if indices form a contiguous range
pub fn is_contiguous(indices: &[usize]) -> bool {
    if indices.len() <= 1 {
        return true;
    }
    indices.windows(2).all(|w| w[1] == w[0] + 1)
}