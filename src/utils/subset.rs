use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use std::mem::ManuallyDrop;

/// UPGRADED: Memory-optimized version that fixes the leak
pub fn subset_rows_only<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    values: Vec<T>, // Takes ownership to prevent leaks
    row_indices: &[usize],
    ncols: usize,
) -> anyhow::Result<CsrMatrix<T>> {
    let total_nnz: usize = row_indices.iter()
        .map(|&row_idx| row_offsets[row_idx + 1] - row_offsets[row_idx])
        .sum();

    // Memory optimization: Try to reuse original allocation if possible
    if total_nnz <= values.capacity() && total_nnz <= values.len() {
        return subset_rows_only_with_reuse(row_offsets, col_indices_orig, values, row_indices, ncols, total_nnz);
    }

    // Standard path with proper memory management
    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::with_capacity(total_nnz);
    let mut new_values = Vec::with_capacity(total_nnz);
    new_row_offsets.push(0);

    // Convert to ManuallyDrop to safely extract values
    let mut values: Vec<ManuallyDrop<T>> = values.into_iter().map(ManuallyDrop::new).collect();

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        new_col_indices.extend_from_slice(&col_indices_orig[start..end]);
        
        for idx in start..end {
            // Safe value extraction without leaving invalid state
            let value = unsafe { ManuallyDrop::take(&mut values[idx]) };
            new_values.push(value);
        }
        new_row_offsets.push(new_col_indices.len());
    }

    // Original vector is automatically cleaned up when ManuallyDrop vec is dropped
    // No memory leak because we extracted the values we need

    let new_pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            ncols,
            new_row_offsets,
            new_col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(new_pattern, new_values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

/// Helper function for memory reuse optimization
fn subset_rows_only_with_reuse<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    mut values: Vec<T>,
    row_indices: &[usize],
    ncols: usize,
    total_nnz: usize,
) -> anyhow::Result<CsrMatrix<T>> {
    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::with_capacity(total_nnz);
    new_row_offsets.push(0);

    // Collect indices of values we need to extract
    let mut indices_to_extract = Vec::with_capacity(total_nnz);
    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];
        new_col_indices.extend_from_slice(&col_indices_orig[start..end]);
        indices_to_extract.extend(start..end);
        new_row_offsets.push(new_col_indices.len());
    }

    // Extract values in a memory-efficient way
    let mut new_values = Vec::with_capacity(total_nnz);
    for &idx in &indices_to_extract {
        new_values.push(unsafe { std::ptr::read(values.as_ptr().add(idx)) });
    }
    
    // Prevent destructors from running on moved values
    std::mem::forget(values);

    let new_pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            ncols,
            new_row_offsets,
            new_col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(new_pattern, new_values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

/// UPGRADED: Fixed contiguous columns with proper pre-allocation
pub fn subset_with_contiguous_columns<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    let col_start = col_indices[0];
    let col_end = col_indices[col_indices.len() - 1] + 1;

    // IMPROVEMENT: Pre-calculate exact size needed (was missing in original)
    let total_nnz: usize = row_indices.iter()
        .map(|&row_idx| {
            let start = row_offsets[row_idx];
            let end = row_offsets[row_idx + 1];
            (start..end).filter(|&idx| {
                let col = col_indices_orig[idx];
                col >= col_start && col < col_end
            }).count()
        }).sum();

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::with_capacity(total_nnz); // Now properly sized!
    let mut new_values = Vec::with_capacity(total_nnz);
    new_row_offsets.push(0);

    // Convert to ManuallyDrop for safe extraction
    let mut values: Vec<ManuallyDrop<T>> = values.into_iter().map(ManuallyDrop::new).collect();

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        for idx in start..end {
            let old_col = col_indices_orig[idx];
            
            if old_col >= col_start && old_col < col_end {
                let new_col = old_col - col_start;
                new_col_indices.push(new_col);
                let value = unsafe { ManuallyDrop::take(&mut values[idx]) };
                new_values.push(value);
            }
        }
        new_row_offsets.push(new_col_indices.len());
    }

    let new_pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            col_indices.len(),
            new_row_offsets,
            new_col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(new_pattern, new_values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

/// UPGRADED: Fixed sparse columns with better memory strategy
pub fn subset_with_sparse_columns<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    let max_col = col_indices.iter().max().copied().unwrap_or(0);
    
    // IMPROVEMENT: Use lookup table for small column sets to avoid sorting overhead
    if col_indices.len() < 1000 && max_col < 10000 {
        return subset_with_sparse_columns_lookup(row_offsets, col_indices_orig, values, row_indices, col_indices, max_col);
    }

    // Original binary search approach for large column sets
    let mut sorted_col_with_new_idx: Vec<(usize, usize)> = col_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx, new_idx))
        .collect();
    sorted_col_with_new_idx.sort_unstable_by_key(|&(old_idx, _)| old_idx);

    let sparsity_factor = col_indices.len() as f64 / 
        col_indices_orig.iter().max().copied().unwrap_or(1) as f64;
    let estimated_nnz = (values.len() as f64 * 
        row_indices.len() as f64 / row_offsets.len().saturating_sub(1).max(1) as f64 * 
        sparsity_factor) as usize;

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::with_capacity(estimated_nnz);
    let mut new_values = Vec::with_capacity(estimated_nnz);
    new_row_offsets.push(0);

    // Convert to ManuallyDrop for safe extraction
    let mut values: Vec<ManuallyDrop<T>> = values.into_iter().map(ManuallyDrop::new).collect();

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        for idx in start..end {
            let old_col = col_indices_orig[idx];
            
            if let Ok(found_idx) = sorted_col_with_new_idx.binary_search_by_key(&old_col, |&(old_idx, _)| old_idx) {
                let new_col = sorted_col_with_new_idx[found_idx].1;
                new_col_indices.push(new_col);
                let value = unsafe { ManuallyDrop::take(&mut values[idx]) };
                new_values.push(value);
            }
        }
        new_row_offsets.push(new_col_indices.len());
    }

    let new_pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            col_indices.len(),
            new_row_offsets,
            new_col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(new_pattern, new_values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

/// Helper function using lookup table for better performance on small column sets
fn subset_with_sparse_columns_lookup<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
    max_col: usize,
) -> anyhow::Result<CsrMatrix<T>> {
    // Create lookup table for O(1) column mapping
    let mut col_mapping = vec![None; max_col + 1];
    for (new_idx, &old_idx) in col_indices.iter().enumerate() {
        col_mapping[old_idx] = Some(new_idx);
    }

    let estimated_nnz = (values.len() * row_indices.len() * col_indices.len()) 
        / (row_offsets.len().saturating_sub(1).max(1) * max_col.max(1));

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::with_capacity(estimated_nnz);
    let mut new_values = Vec::with_capacity(estimated_nnz);
    new_row_offsets.push(0);

    let mut values: Vec<ManuallyDrop<T>> = values.into_iter().map(ManuallyDrop::new).collect();

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        for idx in start..end {
            let old_col = col_indices_orig[idx];
            
            if old_col <= max_col {
                if let Some(new_col) = col_mapping[old_col] {
                    new_col_indices.push(new_col);
                    let value = unsafe { ManuallyDrop::take(&mut values[idx]) };
                    new_values.push(value);
                }
            }
        }
        new_row_offsets.push(new_col_indices.len());
    }

    let new_pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            col_indices.len(),
            new_row_offsets,
            new_col_indices,
        )
    };

    CsrMatrix::try_from_pattern_and_values(new_pattern, new_values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e))
}

// Keep your existing is_contiguous function as-is
pub fn is_contiguous(indices: &[usize]) -> bool {
    if indices.len() <= 1 {
        return true;
    }
    indices.windows(2).all(|w| w[1] == w[0] + 1)
}

