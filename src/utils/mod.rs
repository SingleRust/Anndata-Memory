use anndata::backend::AttributeOp;
use anndata::data::index::Interval;
use anndata::data::{DataFrameIndex};
use anndata::{
    backend::{DataContainer, DatasetOp, GroupOp},
    data::{SelectInfoElem},
};

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
