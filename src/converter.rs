use anndata::{
    AnnData, AnnDataOp, ArrayData, Backend,
    ArrayElemOp, AxisArraysOp, ElemCollectionOp,
};
use anndata_hdf5::H5;
use anndata::data::DynArray;
use ndarray::Array2;
use std::ops::DerefMut;

use crate::{
    ad::helpers::{IMArrayElement, IMElement},
    IMAnnData,
};

/// Converts a backed AnnData object to an in-memory IMAnnData object.
/// This method uses the anndata-rs API to transfer data into memory.
pub fn convert_to_in_memory<B: Backend>(anndata: AnnData<B>) -> anyhow::Result<IMAnnData> {
    convert_to_in_memory_with_options(anndata, false)
}

/// Converts a backed AnnData object to an in-memory IMAnnData object with optional progress reporting.
pub fn convert_to_in_memory_with_options<B: Backend>(
    anndata: AnnData<B>,
    show_progress: bool,
) -> anyhow::Result<IMAnnData> {
    if show_progress {
        println!("Starting AnnData conversion to in-memory...");
    }

    let obs_names = anndata.obs_names().into_vec();
    let var_names = anndata.var_names().into_vec();

    if show_progress { println!("  Loading X matrix..."); }
    // Use get() instead of take() to avoid requiring write intent on backed files
    let x = anndata.x().get::<ArrayData>()?.unwrap_or_else(|| {
        ArrayData::Array(DynArray::from(Array2::<f64>::zeros((0, 0))))
    });

    if show_progress { println!("  Loading observations..."); }
    let obs_df = anndata.read_obs()?;

    if show_progress { println!("  Loading variables..."); }
    let var_df = anndata.read_var()?;

    let imad = IMAnnData::new_extended(
        x,
        obs_names,
        var_names,
        obs_df,
        var_df,
    )?;

    if show_progress { println!("  Loading obsm, obsp, varm, varp..."); }
    for name in anndata.obsm().keys() {
        if let Some(data) = anndata.obsm().get_item::<ArrayData>(&name)? {
            imad.obsm().add_array(name, IMArrayElement::new(data))?;
        }
    }
    for name in anndata.obsp().keys() {
        if let Some(data) = anndata.obsp().get_item::<ArrayData>(&name)? {
            imad.obsp().add_array(name, IMArrayElement::new(data))?;
        }
    }
    for name in anndata.varm().keys() {
        if let Some(data) = anndata.varm().get_item::<ArrayData>(&name)? {
            imad.varm().add_array(name, IMArrayElement::new(data))?;
        }
    }
    for name in anndata.varp().keys() {
        if let Some(data) = anndata.varp().get_item::<ArrayData>(&name)? {
            imad.varp().add_array(name, IMArrayElement::new(data))?;
        }
    }

    if show_progress { println!("  Loading layers..."); }
    for name in anndata.layers().keys() {
        if let Some(data) = anndata.layers().get_item::<ArrayData>(&name)? {
            imad.layers().add_array(name, IMArrayElement::new(data))?;
        }
    }

    if show_progress { println!("  Loading uns..."); }
    for name in anndata.uns().keys() {
        if let Some(data) = anndata.uns().get_item::<anndata::Data>(&name)? {
            imad.uns().add_data(name, IMElement::new(data))?;
        }
    }

    if show_progress {
        println!("Conversion complete.");
    }

    anndata.close()?;
    Ok(imad)
}

pub fn convert_to_backed<B: Backend>(imad: &IMAnnData, anndata: &AnnData<B>) -> anyhow::Result<()> {
    anndata.set_x(imad.x().get_data()?)?;

    anndata.set_obs(imad.obs().get_data())?;
    anndata.set_obs_names(imad.obs_names().into())?;
    anndata.set_var(imad.var().get_data())?;
    anndata.set_var_names(imad.var_names().into())?;

    for key in imad.obsm().keys() {
        let array_data = imad.obsm().get_array(&key)?.get_data()?;
        let mut guard = anndata.obsm().lock();
        if let Some(ref mut data) = guard.deref_mut() {
            data.add_data(&key, array_data)?;
        }
    }
    for key in imad.obsp().keys() {
        let array_data = imad.obsp().get_array(&key)?.get_data()?;
        let mut guard = anndata.obsp().lock();
        if let Some(ref mut data) = guard.deref_mut() {
            data.add_data(&key, array_data)?;
        }
    }
    for key in imad.varm().keys() {
        let array_data = imad.varm().get_array(&key)?.get_data()?;
        let mut guard = anndata.varm().lock();
        if let Some(ref mut data) = guard.deref_mut() {
            data.add_data(&key, array_data)?;
        }
    }
    for key in imad.varp().keys() {
        let array_data = imad.varp().get_array(&key)?.get_data()?;
        let mut guard = anndata.varp().lock();
        if let Some(ref mut data) = guard.deref_mut() {
            data.add_data(&key, array_data)?;
        }
    }
    for key in imad.layers().keys() {
        let array_data = imad.layers().get_array(&key)?.get_data()?;
        let mut guard = anndata.layers().lock();
        if let Some(ref mut data) = guard.deref_mut() {
            data.add_data(&key, array_data)?;
        }
    }

    // Convert uns
    let keys = {
        let uns_slot = imad.uns();
        let read_guard = uns_slot.0.read_inner();
        read_guard.keys().cloned().collect::<Vec<String>>()
    };

    for key in keys {
        let element_data = imad.uns().get_data(&key)?.get_data()?;
        let mut guard = anndata.uns().lock();
        if let Some(ref mut data) = guard.deref_mut() {
            data.add_data(&key, element_data)?;
        }
    }
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
