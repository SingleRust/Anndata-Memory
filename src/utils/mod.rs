use anndata::backend::AttributeOp;
use anndata::data::index::Interval;
use anndata::data::{DataFrameIndex};
use anndata::{
    backend::{DataContainer, DatasetOp, GroupOp, ScalarType},
    data::{SelectInfoElem},
    ArrayData, Backend,
};
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
            let start = slice.start;
            let end = slice.end.unwrap_or(bound as isize);
            let step = slice.step;

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
