use anndata::{backend::{AttributeOp, DataContainer, DatasetOp, GroupOp, ScalarType}, ArrayData, Backend, Readable};
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

    let indptr = read_array_as_usize_optimized::<B>(&indptr_ds)?;
    let indices = read_array_as_usize_optimized::<B>(&indices_ds)?;

    match data_ds.dtype()? {
        ScalarType::F64 => {
            let arr = data_ds.read_array::<f64, Ix1>()?;
            let (data, offset) = arr.into_raw_vec_and_offset();
            if offset.is_none() {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            } else {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            }
        }
        ScalarType::F32 => {
            let arr = data_ds.read_array::<f32, Ix1>()?;
            let (data, offset) = arr.into_raw_vec_and_offset();
            if offset.is_none() {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            } else {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            }
        }
        ScalarType::I64 => {
            let arr = data_ds.read_array::<i64, Ix1>()?;
            let (data, offset) = arr.into_raw_vec_and_offset();
            if offset.is_none() {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            } else {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            }
        }
        ScalarType::I32 => {
            let arr = data_ds.read_array::<i32, Ix1>()?;
            let (data, offset) = arr.into_raw_vec_and_offset();
            if offset.is_none() {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            } else {
                build_csr_matrix(nrows, ncols, indptr, indices, data)
            }
        }
        _ => {
            anndata::data::ArrayData::read(container)
        }
    }
}

