use anndata::{backend::{AttributeOp, DataContainer, DatasetOp, GroupOp, ScalarType}, ArrayData, Backend};
use ndarray::Ix1;

use crate::utils::{build_csr_matrix, read_array_as_usize_optimized};

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

    // Load indices arrays with zero-copy optimization
    let indptr = read_array_as_usize_optimized::<B>(&indptr_ds)?;
    let indices = read_array_as_usize_optimized::<B>(&indices_ds)?;

    // Load data array with zero-copy optimization
    match data_ds.dtype()? {
        ScalarType::F64 => {
            let arr = data_ds.read_array::<f64, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            // Zero-copy successful if _offset is None
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::F32 => {
            let arr = data_ds.read_array::<f32, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::I64 => {
            let arr = data_ds.read_array::<i64, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::I32 => {
            let arr = data_ds.read_array::<i32, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::I16 => {
            let arr = data_ds.read_array::<i16, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::I8 => {
            let arr = data_ds.read_array::<i8, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::U64 => {
            let arr = data_ds.read_array::<u64, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::U32 => {
            let arr = data_ds.read_array::<u32, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::U16 => {
            let arr = data_ds.read_array::<u16, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::U8 => {
            let arr = data_ds.read_array::<u8, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::Bool => {
            let arr = data_ds.read_array::<bool, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
        ScalarType::String => {
            let arr = data_ds.read_array::<String, Ix1>()?;
            let (data, _offset) = arr.into_raw_vec_and_offset();
            build_csr_matrix(nrows, ncols, indptr, indices, data)
        }
    }
}

