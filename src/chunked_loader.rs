use anndata::{
    backend::{AttributeOp, Backend, DataContainer, DatasetOp, GroupOp, ScalarType},
    data::{ArrayData, SelectInfoElem},
    ArrayElemOp,
};
use nalgebra_sparse::{pattern::SparsityPattern, CsrMatrix};







pub fn load_csr_chunked<B: Backend>(
    container: &DataContainer<B>,
    config: &LoadingConfig,
) -> anyhow::Result<ArrayData> {
    let group = container.as_group()?;
    let shape: Vec<u64> = group.get_attr("shape")?;
    let nrows = shape[0] as usize;
    let ncols = shape[1] as usize;

    let data_ds = group.open_dataset("data")?;
    let indices_ds = group.open_dataset("indices")?;
    let indptr_ds = group.open_dataset("indptr")?;

    // Use the helper function to read indptr
    let indptr = read_array_as_usize::<B>(&indptr_ds)?;

    let nnz = data_ds.shape()[0];

    if config.show_progress && nnz > 10_000_000 {
        println!(
            "Loading CSR matrix: {} rows, {} cols, {} non-zeros",
            nrows, ncols, nnz
        );
    }

    match data_ds.dtype()? {
        ScalarType::F64 => load_csr_typed::<B, f64>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
        ScalarType::F32 => load_csr_typed::<B, f32>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
        ScalarType::I64 => load_csr_typed::<B, i64>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
        ScalarType::I32 => load_csr_typed::<B, i32>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
        ScalarType::I16 => load_csr_typed::<B, i16>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
        ScalarType::I8 => load_csr_typed::<B, i8>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
        ScalarType::U64 => load_csr_typed::<B, u64>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
        ScalarType::U32 => load_csr_typed::<B, u32>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
        ScalarType::U16 => load_csr_typed::<B, u16>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
        ScalarType::U8 => load_csr_typed::<B, u8>(
            nrows, ncols, nnz, indptr, &data_ds, &indices_ds, config,
        ),
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
    anndata::ArrayData: std::convert::From<nalgebra_sparse::CsrMatrix<T>>
{
    let chunk_size = (config.chunk_size_mb * 1_048_576) / (std::mem::size_of::<T>() + 8);
    let chunk_size = chunk_size.max(1000);

    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);

    let mut offset = 0;
    let mut last_progress = 0;

    while offset < nnz {
        let chunk_end = (offset + chunk_size).min(nnz);

        let data_array = data_ds.read_array_slice::<T, _, ndarray::Ix1>(&[SelectInfoElem::from(offset..chunk_end)])?;
        let data_chunk: Vec<T> = data_array.into_raw_vec();

 
        match indices_ds.dtype()? {
            ScalarType::U64 => {
                let indices_array = indices_ds.read_array_slice::<u64, _, ndarray::Ix1>(&[SelectInfoElem::from(offset..chunk_end)])?;
                let (indices_u64, _) = indices_array.into_raw_vec_and_offset();
                indices.extend(indices_u64.into_iter().map(|x| x as usize));
            }
            ScalarType::U32 => {
                let indices_array = indices_ds.read_array_slice::<u32, _, ndarray::Ix1>(&[SelectInfoElem::from(offset..chunk_end)])?;
                let (indices_u32, _) = indices_array.into_raw_vec_and_offset();
                indices.extend(indices_u32.into_iter().map(|x| x as usize));
            }
            ScalarType::I64 => {
                let indices_array = indices_ds.read_array_slice::<i64, _, ndarray::Ix1>(&[SelectInfoElem::from(offset..chunk_end)])?;
                let (indices_i64, _) = indices_array.into_raw_vec_and_offset();
                indices.extend(indices_i64.into_iter().map(|x| x as usize));
            }
            ScalarType::I32 => {
                let indices_array = indices_ds.read_array_slice::<i32, _, ndarray::Ix1>(&[SelectInfoElem::from(offset..chunk_end)])?;
                let (indices_i32, _) = indices_array.into_raw_vec_and_offset();
                indices.extend(indices_i32.into_iter().map(|x| x as usize));
            }
            _ => anyhow::bail!("Unsupported index type for CSR matrix"),
        }

        data.extend(data_chunk);

        offset = chunk_end;

        if config.show_progress && nnz > 10_000_000 {
            let progress = (offset as f64 / nnz as f64 * 100.0) as usize;
            if progress >= last_progress + 10 {
                println!("Loading CSR matrix: {}%", progress);
                last_progress = progress;
            }
        }
    }

    if config.show_progress && nnz > 10_000_000 {
        println!("Constructing CSR matrix structure...");
    }

    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(nrows, ncols, indptr, indices)
    };
    let csr = CsrMatrix::try_from_pattern_and_values(pattern, data).map_err(|e| anyhow::anyhow!("There was an error constructing the matrix {}", e))?;

    Ok(ArrayData::from(csr))
}
