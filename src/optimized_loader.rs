pub fn load_csr_optimized<B: Backend>(
    container: &DataContainer<B>,
) -> anyhow::Result<ArrayData> {
    let group = container.as_group()?;
    let shape: Vec<u64> = group.get_attr("shape")?;
    let nrows = shape[0] as usize;
    let ncols = shape[1] as usize;

    let data_ds = group.open_dataset("data")?;
    let indices_ds = group.open_dataset("indices")?;
    let indptr_ds = group.open_dataset("indptr")?;

    // Use your existing function but optimize it to avoid iterator when possible
    let indptr = read_array_as_usize_optimized::<B>(&indptr_ds)?;
    let indices = read_array_as_usize_optimized::<B>(&indices_ds)?;

    // Read data based on type - optimize to avoid copying when possible
    match data_ds.dtype()? {
        ScalarType::F64 => {
            let arr = data_ds.read_array::<f64, Ix1>()?;
            let (data, offset) = arr.into_raw_vec_and_offset();
            if offset.is_none() {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            } else {
                build_csr_matrix(nrows, ncols, indptr, indices, arr.to_vec())
            }
        }
        ScalarType::F32 => {
            let arr = data_ds.read_array::<f32, Ix1>()?;
            let (data, offset) = arr.into_raw_vec_and_offset();
            if offset.is_none() {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            } else {
                build_csr_matrix(nrows, ncols, indptr, indices, arr.to_vec())
            }
        }
        ScalarType::I64 => {
            let arr = data_ds.read_array::<i64, Ix1>()?;
            let (data, offset) = arr.into_raw_vec_and_offset();
            if offset.is_none() {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            } else {
                build_csr_matrix(nrows, ncols, indptr, indices, arr.to_vec())
            }
        }
        ScalarType::I32 => {
            let arr = data_ds.read_array::<i32, Ix1>()?;
            let (data, offset) = arr.into_raw_vec_and_offset();
            if offset.is_none() {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            } else {
                build_csr_matrix(nrows, ncols, indptr, indices, arr.to_vec())
            }
        }
        _ => {
            // Fallback to standard loading for other types
            anndata::data::ArrayData::read(container)
        }
    }
}