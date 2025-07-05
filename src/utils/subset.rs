use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use std::mem::{replace, ManuallyDrop, MaybeUninit};

pub fn subset_rows_only<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    values: Vec<T>,
    row_indices: &[usize],
    ncols: usize,
) -> anyhow::Result<CsrMatrix<T>> {
    let total_nnz: usize = row_indices.iter()
        .map(|&row_idx| row_offsets[row_idx + 1] - row_offsets[row_idx])
        .sum();

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::with_capacity(total_nnz);
    let mut new_values = Vec::with_capacity(total_nnz);
    new_row_offsets.push(0);

    let mut values: Vec<ManuallyDrop<T>> = values.into_iter().map(ManuallyDrop::new).collect();

    for &row_idx in row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        new_col_indices.extend_from_slice(&col_indices_orig[start..end]);
        
        for idx in start..end {
            let value = unsafe { ManuallyDrop::take(&mut values[idx]) };
            new_values.push(value);
        }
        new_row_offsets.push(new_col_indices.len());
    }

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

pub fn subset_with_contiguous_columns<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
    let col_start = col_indices[0];
    let col_end = col_indices[col_indices.len() - 1] + 1;

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
    let mut new_col_indices = Vec::with_capacity(total_nnz);
    let mut new_values = Vec::with_capacity(total_nnz);
    new_row_offsets.push(0);

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

pub fn subset_with_sparse_columns<T>(
    row_offsets: &[usize],
    col_indices_orig: &[usize],
    values: Vec<T>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> anyhow::Result<CsrMatrix<T>> {
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

pub fn is_contiguous(indices: &[usize]) -> bool {
    if indices.len() <= 1 {
        return true;
    }
    indices.windows(2).all(|w| w[1] == w[0] + 1)
}