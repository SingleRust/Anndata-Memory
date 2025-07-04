use std::ops::{Deref, DerefMut};

use anndata::data::DataFrameIndex;
use anndata::{
    AnnData, AnnDataOp, ArrayData, ArrayElemOp, AxisArrays, Backend, ElemCollection,
};
use anndata_hdf5::H5;
use anyhow::Ok;

use crate::{
    ad::helpers::{IMAxisArrays, IMElement},
    IMAnnData, IMArrayElement, IMElementCollection,
};

#[derive(Clone, Debug)]
pub struct LoadingConfig {
    pub use_chunked_loading: bool,
    pub chunk_size_mb: usize,
    pub memory_threshold_mb: usize,
    pub show_progress: bool,
}

impl Default for LoadingConfig {
    fn default() -> Self {
        Self {
            use_chunked_loading: false,
            chunk_size_mb: 100,
            memory_threshold_mb: 1024,
            show_progress: true,
        }
    }
}

pub fn convert_to_in_memory<B: Backend>(anndata: AnnData<B>) -> anyhow::Result<IMAnnData> {
    let obs_df = anndata.read_obs()?;
    let obs_names = anndata.obs_names();
    let var_df = anndata.read_var()?;
    let var_names = anndata.var_names();
    let x = anndata.x().get::<ArrayData>()?.unwrap();
    let imad = IMAnnData::new_extended(
        x,
        obs_names.into_vec(),
        var_names.into_vec(),
        obs_df,
        var_df,
    )?;
    convert_axis_arrays_to_mem(anndata.obsm(), imad.obsm())?;
    convert_axis_arrays_to_mem(anndata.obsp(), imad.obsp())?;
    convert_axis_arrays_to_mem(anndata.varm(), imad.varm())?;
    convert_axis_arrays_to_mem(anndata.varp(), imad.varp())?;
    convert_axis_arrays_to_mem(anndata.layers(), imad.layers())?;
    convert_uns_to_mem(anndata.uns(), imad.uns())?;
    anndata.close()?;
    Ok(imad)
}

fn convert_axis_arrays_to_mem<B: Backend>(
    axis_arr: &AxisArrays<B>,
    reference_element: IMAxisArrays,
) -> anyhow::Result<()> {
    if axis_arr.is_none() {
        return Ok(());
    }
    let x = axis_arr.inner();
    let iax = x.deref();
    let data = iax.deref();
    for (k, v) in data.iter() {
        let arr = v.get::<ArrayData>()?.unwrap();
        let im_arr = IMArrayElement::new(arr);
        reference_element.add_array(k.to_string(), im_arr)?;
    }
    Ok(())
}

fn convert_uns_to_mem<B: Backend>(
    elem_col: &ElemCollection<B>,
    reference_element: IMElementCollection,
) -> anyhow::Result<()> {
    if elem_col.is_none() {
        return Ok(());
    }
    let x = elem_col.inner();
    let iax = x.deref();
    let data = iax.deref();
    for (k, v) in data.iter() {
        let data = v.inner().data();
        let d = IMElement::new(data?);
        reference_element.add_data(k.to_string(), d)?;
    }
    Ok(())
}

fn convert_axis_arrays_to_backed<B: Backend>(
    reference: IMAxisArrays,
    target_axis_arrays: &AxisArrays<B>,
) -> anyhow::Result<()> {
    if reference.is_empty() {
        return Ok(());
    }

    for key in reference.keys() {
        let value = reference.get_array(&key)?;
        let array_data = value.get_data()?;
        let mut guard = target_axis_arrays.lock();
        let data = guard.deref_mut();
        match (data) {
            None => {}
            Some(data) => data.add_data(&key, array_data)?,
        };
    }

    Ok(())
}

fn convert_uns_to_backed<B: Backend>(
    reference: IMElementCollection,
    target_file: &AnnData<B>,
) -> anyhow::Result<()> {
    let uns = target_file.uns();

    let keys = {
        let read_guard = reference.0.read_inner();
        read_guard.keys().cloned().collect::<Vec<String>>()
    };

    for key in keys {
        let val = reference.get_data(&key)?;
        {
            let elem_data = val.get_data()?;

            // Lock the uns structure and update it
            let mut guard = uns.lock();
            let data = guard.deref_mut();
            if let Some(data) = data {
                data.add_data(&key, elem_data)?;
            }
        }
    }
    Ok(())
}

pub fn convert_to_backed<B: Backend>(imad: &IMAnnData, anndata: &AnnData<B>) -> anyhow::Result<()> {
    let x_data = imad.x().get_data()?;
    anndata.set_x(x_data)?;

    anndata.set_obs(imad.obs().get_data())?;
    anndata.set_obs_names(DataFrameIndex::from(imad.obs_names()))?;
    anndata.set_var(imad.var().get_data())?;
    anndata.set_var_names(DataFrameIndex::from(imad.var_names()))?;

    convert_axis_arrays_to_backed(imad.obsm(), anndata.obsm())?;
    convert_axis_arrays_to_backed(imad.obsp(), anndata.obsp())?;
    convert_axis_arrays_to_backed(imad.varm(), anndata.varm())?;
    convert_axis_arrays_to_backed(imad.varp(), anndata.varp())?;

    convert_axis_arrays_to_backed(imad.layers(), anndata.layers())?;

    convert_uns_to_backed(imad.uns(), anndata)?;

    Ok(())
}

pub fn convert_to_new_backed_h5(
    imad: &IMAnnData,
    path: impl AsRef<std::path::Path>,
) -> anyhow::Result<AnnData<H5>> {
    let anndata = AnnData::<H5>::new(path)?;

    convert_to_backed(imad, &anndata)?;

    Ok(anndata)
}
