use anndata::backend::AttributeOp;
use anndata::data::index::Interval;
use anndata::data::{DataFrameIndex};
use anndata::{
    backend::{DataContainer, DatasetOp, GroupOp, ScalarType},
    data::{DynCscMatrix, DynCsrMatrix, SelectInfoElem},
    ArrayData, Backend,
};
use nalgebra_sparse::{pattern::SparsityPattern, CscMatrix, CsrMatrix};
use ndarray::Slice;
use std::collections::HashMap;

use crate::utils::subset::{subset_csr_internal};
use crate::{LoadingConfig, LoadingStrategy};

mod subset;

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

// ####################################################################################################
//                              Subsetting
// ####################################################################################################


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

    let row_indices = crate::utils::select_info_elem_to_indices(s[0], nrows)?;
    let col_indices = if s.len() > 1 {
        crate::utils::select_info_elem_to_indices(s[1], ncols)?
    } else {
        (0..ncols).collect()
    };

    // Use matrix disassembly to get ownership of the data
    let (row_offsets, col_indices_orig, values) = matrix.disassemble();
    
    // Call the internal optimized subset function
    subset_csr_internal(row_offsets, col_indices_orig, values, &row_indices, &col_indices)
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

// ####################################################################################################
//                              Optimized loader
// ####################################################################################################

pub fn read_array_as_usize_optimized<B: Backend>(
    dataset: &B::Dataset,
) -> anyhow::Result<Vec<usize>> {
    match dataset.dtype()? {
        #[cfg(target_pointer_width = "64")]
        ScalarType::U64 => {
            let arr = dataset.read_array::<u64, ndarray::Ix1>()?;
            let (vec, _) = arr.into_raw_vec_and_offset();
            Ok(unsafe { std::mem::transmute::<Vec<u64>, Vec<usize>>(vec) })
        }

        #[cfg(target_pointer_width = "32")]
        ScalarType::U32 => {
            let arr = dataset.read_array::<u32, ndarray::Ix1>()?;
            let (vec, _) = arr.into_raw_vec_and_offset();
            Ok(unsafe { std::mem::transmute::<Vec<u32>, Vec<usize>>(vec) })
        }

        #[cfg(target_pointer_width = "64")]
        ScalarType::I64 => {
            let arr = dataset.read_array::<i64, ndarray::Ix1>()?;
            let (vec, _) = arr.into_raw_vec_and_offset();

            if vec.iter().all(|&x| x >= 0) {
                Ok(unsafe { std::mem::transmute::<Vec<i64>, Vec<usize>>(vec) })
            } else {
                vec.into_iter()
                    .map(|x| {
                        if x < 0 {
                            anyhow::bail!("Negative value {} cannot be converted to usize", x);
                        }
                        Ok(x as usize)
                    })
                    .collect()
            }
        }

        #[cfg(target_pointer_width = "32")]
        ScalarType::I32 => {
            let arr = dataset.read_array::<i32, ndarray::Ix1>()?;
            let (vec, _) = arr.into_raw_vec_and_offset();

            if vec.iter().all(|&x| x >= 0) {
                Ok(unsafe { std::mem::transmute::<Vec<i32>, Vec<usize>>(vec) })
            } else {
                vec.into_iter()
                    .map(|x| {
                        if x < 0 {
                            anyhow::bail!("Negative value {} cannot be converted to usize", x);
                        }
                        Ok(x as usize)
                    })
                    .collect()
            }
        }

        // For other types, fall back to the safe original implementation
        _ => read_array_as_usize::<B>(dataset),
    }
}

pub fn read_array_as_usize<B: Backend>(dataset: &B::Dataset) -> anyhow::Result<Vec<usize>> {
    match dataset.dtype()? {
        ScalarType::U64 => {
            let arr = dataset.read_array::<u64, ndarray::Ix1>()?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::U32 => {
            let arr = dataset.read_array::<u32, ndarray::Ix1>()?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::U16 => {
            let arr = dataset.read_array::<u16, ndarray::Ix1>()?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::U8 => {
            let arr = dataset.read_array::<u8, ndarray::Ix1>()?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::I64 => {
            let arr = dataset.read_array::<i64, ndarray::Ix1>()?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::I32 => {
            let arr = dataset.read_array::<i32, ndarray::Ix1>()?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::I16 => {
            let arr = dataset.read_array::<i16, ndarray::Ix1>()?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::I8 => {
            let arr = dataset.read_array::<i8, ndarray::Ix1>()?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        dt => anyhow::bail!("Cannot read {:?} as usize array", dt),
    }
}

pub fn read_array_slice_as_usize<B: Backend>(
    dataset: &B::Dataset,
    selection: &[SelectInfoElem],
) -> anyhow::Result<Vec<usize>> {
    match dataset.dtype()? {
        ScalarType::U64 => {
            let arr = dataset.read_array_slice::<u64, _, ndarray::Ix1>(selection)?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::U32 => {
            let arr = dataset.read_array_slice::<u32, _, ndarray::Ix1>(selection)?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::U16 => {
            let arr = dataset.read_array_slice::<u16, _, ndarray::Ix1>(selection)?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::U8 => {
            let arr = dataset.read_array_slice::<u8, _, ndarray::Ix1>(selection)?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::I64 => {
            let arr = dataset.read_array_slice::<i64, _, ndarray::Ix1>(selection)?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::I32 => {
            let arr = dataset.read_array_slice::<i32, _, ndarray::Ix1>(selection)?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::I16 => {
            let arr = dataset.read_array_slice::<i16, _, ndarray::Ix1>(selection)?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        ScalarType::I8 => {
            let arr = dataset.read_array_slice::<i8, _, ndarray::Ix1>(selection)?;
            Ok(arr.into_iter().map(|x| x as usize).collect())
        }
        dt => anyhow::bail!("Cannot read {:?} as usize array", dt),
    }
}

pub fn should_use_chunked_loading<B: Backend>(
    container: &DataContainer<B>,
    config: &LoadingConfig,
) -> anyhow::Result<bool> {
    // Check for explicit user override first
    match config.loading_strategy {
        LoadingStrategy::ForceComplete => return Ok(false), // Force complete loading
        LoadingStrategy::ForceChunked => return Ok(true),   // Force chunked loading
        LoadingStrategy::Auto => {}                         // Continue with automatic decision
    }

    // Only consider chunked loading for CSR matrices
    match container.encoding_type()? {
        anndata::backend::DataType::CsrMatrix(_) => {
            let group = container.as_group()?;
            let shape: Vec<u64> = group.get_attr("shape")?;
            let nrows = shape[0] as usize;
            let nnz = group.open_dataset("data")?.shape()[0];

            // Estimate total memory needed for CSR matrix construction
            let data_type_size = match group.open_dataset("data")?.dtype()? {
                ScalarType::F64 | ScalarType::I64 | ScalarType::U64 => 8,
                ScalarType::F32 | ScalarType::I32 | ScalarType::U32 => 4,
                ScalarType::I16 | ScalarType::U16 => 2,
                ScalarType::I8 | ScalarType::U8 | ScalarType::Bool => 1,
                ScalarType::String => 24, // Rough estimate for String
            };

            let estimated_memory_mb =
                estimate_csr_total_memory_usage(nnz, nrows, data_type_size) / 1_048_576;

            if config.show_progress {
                println!(
                    "  Estimated peak memory usage: {} MB (threshold: {} MB)",
                    estimated_memory_mb, config.memory_threshold_mb
                );
            }

            // Use chunked loading if estimated memory exceeds threshold
            Ok(estimated_memory_mb > config.memory_threshold_mb)
        }
        _ => Ok(false), // Never use chunked loading for non-CSR data
    }
}

fn estimate_csr_total_memory_usage(nnz: usize, nrows: usize, data_type_size: usize) -> usize {
    // During loading, we temporarily need:
    let data_array_size = nnz * data_type_size;
    let indices_array_size = nnz * std::mem::size_of::<usize>();
    let indptr_array_size = (nrows + 1) * std::mem::size_of::<usize>();

    let final_csr_size = data_array_size + indices_array_size + indptr_array_size;

    let peak_usage = (data_array_size + indices_array_size + indptr_array_size) + final_csr_size;

    (peak_usage as f64 * 1.2) as usize
}

pub fn build_csr_matrix<T>(
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<T>,
) -> anyhow::Result<ArrayData>
where
    CsrMatrix<T>: Into<ArrayData>,
{
    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(nrows, ncols, indptr, indices)
    };
    let csr = CsrMatrix::try_from_pattern_and_values(pattern, data)
        .map_err(|e| anyhow::anyhow!("Building the CSR encountered an error, {}", e))?;
    Ok(csr.into())
}

pub fn build_csc_matrix<T>(
    nrows: usize,
    ncols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<T>,
) -> anyhow::Result<ArrayData>
where
    CscMatrix<T>: Into<ArrayData>,
{
    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(nrows, ncols, indptr, indices)
    };
    let csc = CscMatrix::try_from_pattern_and_values(pattern, data)
        .map_err(|e| anyhow::anyhow!("Building the CSC matrix encountered an error: {}", e))?;
    Ok(csc.into())
}

pub fn read_dataframe_index(
    container: &DataContainer<anndata_hdf5::H5>,
) -> anyhow::Result<DataFrameIndex> {
    let index_name: String = container.get_attr("_index")?;
    let dataset = container.as_group()?.open_dataset(&index_name)?;
    match dataset
        .get_attr::<String>("index_type")
        .as_ref()
        .map_or("list", |x| x.as_str())
    {
        "list" => {
            let data = dataset.read_array()?;
            let mut index: DataFrameIndex = data.to_vec().into();
            index.index_name = index_name;
            Ok(index)
        }
        "intervals" => {
            let keys: Vec<String> = dataset.get_attr("names")?;
            let values: Vec<Vec<u64>> = dataset.get_attr("intervals")?;
            Ok(keys
                .into_iter()
                .zip(values.into_iter().map(|row| Interval {
                    start: row[0] as usize,
                    end: row[1] as usize,
                    size: row[2] as usize,
                    step: row[3] as usize,
                }))
                .collect())
        }
        "range" => {
            let start: u64 = dataset.get_attr("start")?;
            let end: u64 = dataset.get_attr("end")?;
            Ok((start as usize..end as usize).into())
        }
        x => anyhow::bail!("Unknown index type: {}", x),
    }
}
