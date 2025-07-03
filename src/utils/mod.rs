use std::{collections::HashMap, mem::replace};

use anndata::data::{DynCscMatrix, DynCsrMatrix, SelectInfoElem};
use nalgebra_sparse::{pattern::SparsityPattern, CscMatrix, CsrMatrix};
use ndarray::Slice;

pub(crate) fn select_info_elem_to_indices(
    elem: &SelectInfoElem,
    bound: usize,
) -> anyhow::Result<Vec<usize>> {
    match elem {
        SelectInfoElem::Index(indices) => {
            for &idx in indices {
                if idx >= bound {
                    anyhow::bail!("Index out of bounds: {} >= {}", idx, bound);
                }
            }
            Ok(indices.clone())
        }
        SelectInfoElem::Slice(slice) => {
            let Slice { start, end, step } = *slice;
            let end = end.unwrap_or(bound as isize);

            if start as usize >= bound || end as usize > bound {
                anyhow::bail!(
                    "Slice out of bounds: start={}, end={}, bound={}",
                    start,
                    end,
                    bound
                );
            }

            let indices: Vec<usize> = (start..end)
                .step_by(step as usize)
                .map(|i| i as usize)
                .collect();

            Ok(indices)
        }
    }
}

pub(crate) fn subset_dyn_csc_matrix(
    dyn_csc: DynCscMatrix,
    s: &[&SelectInfoElem],
) -> anyhow::Result<DynCscMatrix> {
    use DynCscMatrix::*;

    macro_rules! subset_csc {
        ($matrix:expr, $variant:ident) => {{
            let result = subset_csc_matrix($matrix, s)?;
            $variant(result)
        }};
    }

    Ok(match dyn_csc {
        I8(m) => subset_csc!(m, I8),
        I16(m) => subset_csc!(m, I16),
        I32(m) => subset_csc!(m, I32),
        I64(m) => subset_csc!(m, I64),
        U8(m) => subset_csc!(m, U8),
        U16(m) => subset_csc!(m, U16),
        U32(m) => subset_csc!(m, U32),
        U64(m) => subset_csc!(m, U64),
        F32(m) => subset_csc!(m, F32),
        F64(m) => subset_csc!(m, F64),
        Bool(m) => subset_csc!(m, Bool),
        String(m) => subset_csc!(m, String),
    })
}

pub(crate) fn subset_dyn_csr_matrix(
    dyn_csr: DynCsrMatrix,
    s: &[&SelectInfoElem],
) -> anyhow::Result<DynCsrMatrix> {
    use DynCsrMatrix::*;

    macro_rules! subset_csr {
        ($matrix:expr, $variant:ident) => {{
            let result = subset_csr_matrix($matrix, s)?;
            $variant(result)
        }};
    }

    Ok(match dyn_csr {
        I8(m) => subset_csr!(m, I8),
        I16(m) => subset_csr!(m, I16),
        I32(m) => subset_csr!(m, I32),
        I64(m) => subset_csr!(m, I64),
        U8(m) => subset_csr!(m, U8),
        U16(m) => subset_csr!(m, U16),
        U32(m) => subset_csr!(m, U32),
        U64(m) => subset_csr!(m, U64),
        F32(m) => subset_csr!(m, F32),
        F64(m) => subset_csr!(m, F64),
        Bool(m) => subset_csr!(m, Bool),
        String(m) => subset_csr!(m, String),
    })
}

fn subset_csr_matrix<T>(
    matrix: CsrMatrix<T>,
    s: &[&SelectInfoElem],
) -> anyhow::Result<CsrMatrix<T>> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();

    let row_indices = select_info_elem_to_indices(s[0], nrows)?;
    let col_indices = if s.len() > 1 {
        select_info_elem_to_indices(s[1], ncols)?
    } else {
        (0..ncols).collect()
    };

    if row_indices.len() == nrows && col_indices.len() == ncols {
        return Ok(matrix);
    }

    let (row_offsets, col_indices_orig, values) = matrix.disassemble();

    if col_indices.len() == ncols {
        let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
        let mut new_col_indices = Vec::new();
        let mut new_values = Vec::new();
        new_row_offsets.push(0);

        let mut values_iter = values.into_iter();
        let mut current_pos = 0;

        for &row_idx in &row_indices {
            let start = row_offsets[row_idx];
            let end = row_offsets[row_idx + 1];

            for _ in current_pos..start {
                values_iter.next();
            }

            new_col_indices.extend_from_slice(&col_indices_orig[start..end]);
            for _ in start..end {
                if let Some(val) = values_iter.next() {
                    new_values.push(val);
                }
            }

            current_pos = end;
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

        return CsrMatrix::try_from_pattern_and_values(new_pattern, new_values)
            .map_err(|e| anyhow::anyhow!("Failed to create CSR matrix: {:?}", e));
    }

    let col_map: HashMap<usize, usize> = col_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx, new_idx))
        .collect();

    let capacity: usize = row_indices
        .iter()
        .flat_map(|&row| {
            let start = row_offsets[row];
            let end = row_offsets[row + 1];
            (start..end).filter(|&idx| col_map.contains_key(&col_indices_orig[idx]))
        })
        .count();

    let mut new_row_offsets = Vec::with_capacity(row_indices.len() + 1);
    let mut new_col_indices = Vec::with_capacity(capacity);
    let mut new_values = Vec::with_capacity(capacity);
    new_row_offsets.push(0);

    let mut values_vec = values;

    for &row_idx in &row_indices {
        let start = row_offsets[row_idx];
        let end = row_offsets[row_idx + 1];

        for idx in start..end {
            let col = col_indices_orig[idx];
            if let Some(&new_col) = col_map.get(&col) {
                new_col_indices.push(new_col);
                new_values.push(replace(&mut values_vec[idx], unsafe { std::mem::zeroed() }));
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

fn subset_csc_matrix<T>(
    matrix: CscMatrix<T>,
    s: &[&SelectInfoElem],
) -> anyhow::Result<CscMatrix<T>> {
    use crate::utils::select_info_elem_to_indices;

    let nrows = matrix.nrows();
    let ncols = matrix.ncols();

    let row_indices = select_info_elem_to_indices(s[0], nrows)?;
    let col_indices = if s.len() > 1 {
        select_info_elem_to_indices(s[1], ncols)?
    } else {
        (0..ncols).collect()
    };

    if row_indices.len() == nrows && col_indices.len() == ncols {
        return Ok(matrix);
    }

    let (col_offsets, row_indices_orig, values) = matrix.disassemble();

    if row_indices.len() == nrows {
        let mut new_col_offsets = Vec::with_capacity(col_indices.len() + 1);
        let mut new_row_indices = Vec::new();
        let mut new_values = Vec::new();
        new_col_offsets.push(0);

        let mut values_iter = values.into_iter();
        let mut current_pos = 0;

        for &col_idx in &col_indices {
            let start = col_offsets[col_idx];
            let end = col_offsets[col_idx + 1];

            for _ in current_pos..start {
                values_iter.next();
            }

            new_row_indices.extend_from_slice(&row_indices_orig[start..end]);
            for _ in start..end {
                if let Some(val) = values_iter.next() {
                    new_values.push(val);
                }
            }

            current_pos = end;
            new_col_offsets.push(new_row_indices.len());
        }

        let new_pattern = unsafe {
            nalgebra_sparse::pattern::SparsityPattern::from_offset_and_indices_unchecked(
                nrows,
                col_indices.len(),
                new_col_offsets,
                new_row_indices,
            )
        };

        return CscMatrix::try_from_pattern_and_values(new_pattern, new_values)
            .map_err(|e| anyhow::anyhow!("Failed to create CSC matrix: {:?}", e));
    }

    let row_map: HashMap<usize, usize> = row_indices
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx, new_idx))
        .collect();

    let capacity: usize = col_indices
        .iter()
        .flat_map(|&col| {
            let start = col_offsets[col];
            let end = col_offsets[col + 1];
            (start..end).filter(|&idx| row_map.contains_key(&row_indices_orig[idx]))
        })
        .count();

    let mut new_col_offsets = Vec::with_capacity(col_indices.len() + 1);
    let mut new_row_indices = Vec::with_capacity(capacity);
    let mut new_values = Vec::with_capacity(capacity);
    new_col_offsets.push(0);

    let mut values_vec = values;

    for &col_idx in &col_indices {
        let start = col_offsets[col_idx];
        let end = col_offsets[col_idx + 1];

        for idx in start..end {
            let row = row_indices_orig[idx];
            if let Some(&new_row) = row_map.get(&row) {
                new_row_indices.push(new_row);
                new_values.push(std::mem::replace(&mut values_vec[idx], unsafe {
                    std::mem::zeroed()
                }));
            }
        }

        new_col_offsets.push(new_row_indices.len());
    }

    let new_pattern = unsafe {
        nalgebra_sparse::pattern::SparsityPattern::from_offset_and_indices_unchecked(
            row_indices.len(),
            col_indices.len(),
            new_col_offsets,
            new_row_indices,
        )
    };

    CscMatrix::try_from_pattern_and_values(new_pattern, new_values)
        .map_err(|e| anyhow::anyhow!("Failed to create CSC matrix: {:?}", e))
}
