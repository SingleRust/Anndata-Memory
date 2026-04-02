use anndata::data::{SelectInfoElem};

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
