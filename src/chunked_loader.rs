use anndata::{
    backend::{AttributeOp, Backend, DataContainer, DatasetOp, GroupOp, ScalarType},
    data::{ArrayData, SelectInfoElem},
    ArrayElemOp,
};
use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};
use ndarray::Ix1;

use crate::{converter::LoadingConfig, utils::{read_array_as_usize_optimized, read_array_slice_as_usize}};

pub fn load_csr_chunked<B: Backend>(
    container: &DataContainer<B>,
    config: &LoadingConfig,
) -> anyhow::Result<ArrayData> {
    let group = container.as_group()?;
    let shape: Vec<u64> = group.get_attr("shape")?;
    let (nrows, ncols) = (shape[0] as usize, shape[1] as usize);

    let data_ds = group.open_dataset("data")?;
    let indices_ds = group.open_dataset("indices")?;
    let indptr_ds = group.open_dataset("indptr")?;

    let indptr = read_array_as_usize_optimized::<B>(&indptr_ds)?;
    let nnz = data_ds.shape()[0];

    if config.show_progress && nnz > 10_000_000 {
        println!("Loading CSR matrix: {} rows, {} cols, {} non-zeros", nrows, ncols, nnz);
    }

    use ScalarType::*;
    match data_ds.dtype()? {
        F64 => load_csr_typed::<B, f64>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        F32 => load_csr_typed::<B, f32>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        I64 => load_csr_typed::<B, i64>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        I32 => load_csr_typed::<B, i32>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        I16 => load_csr_typed::<B, i16>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        I8 => load_csr_typed::<B, i8>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        U64 => load_csr_typed::<B, u64>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        U32 => load_csr_typed::<B, u32>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        U16 => load_csr_typed::<B, u16>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        U8 => load_csr_typed::<B, u8>(nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config),
        dt => anyhow::bail!("Unsupported data type for CSR matrix: {:?}", dt),
    }
}

fn load_csr_typed<B: Backend, T: anndata::backend::BackendData>(
    nrows: usize,
    ncols: usize,
    nnz: usize,
    indptr: Vec<usize>,
    data_ds: &B::Dataset,
    indices_ds: &B::Dataset,
    config: &LoadingConfig,
) -> anyhow::Result<ArrayData>
where
    ArrayData: From<CsrMatrix<T>>,
{
    let chunk_size = ((config.chunk_size_mb << 20) / (std::mem::size_of::<T>() + 8)).max(1000);

    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);

    let show_progress = config.show_progress && nnz > 10_000_000;
    let progress_interval = if show_progress { nnz / 10 } else { usize::MAX };
    let mut next_progress = progress_interval;
    
    let mut offset = 0;
    while offset < nnz {
        let chunk_end = (offset + chunk_size).min(nnz);
        let range = [SelectInfoElem::from(offset..chunk_end)];
        let data_array = data_ds.read_array_slice::<T, _, Ix1>(&range)?;
        let (data_vec, data_offset) = data_array.into_raw_vec_and_offset();
        if data_offset.is_none() {
            data.extend(data_vec);
        } else {
            data.extend(data_vec);
        }

        let indices_chunk = read_array_slice_as_usize::<B>(indices_ds, &range)?;
        indices.extend(indices_chunk);

        offset = chunk_end;

        if show_progress && offset >= next_progress {
            println!("Loading CSR matrix: {}%", offset * 100 / nnz);
            next_progress += progress_interval;
        }
    }

    if show_progress {
        println!("Constructing CSR matrix structure...");
    }

    let pattern = unsafe { SparsityPattern::from_offset_and_indices_unchecked(nrows, ncols, indptr, indices) };
    CsrMatrix::try_from_pattern_and_values(pattern, data)
        .map(ArrayData::from)
        .map_err(|e| anyhow::anyhow!("Failed to construct CSR matrix: {}", e))
}