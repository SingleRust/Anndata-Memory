use std::ops::Deref;

use anndata::{AnnData, AnnDataOp, ArrayData, ArrayElemOp, AxisArrays, Backend, Data, ElemCollection};
use anyhow::Ok;

use crate::{ad::helpers::{Element, IMAxisArrays}, IMAnnData, IMArrayElement, IMElementCollection};

pub fn convert_to_in_memory<B: Backend>(anndata: AnnData<B>) -> anyhow::Result<IMAnnData> {
    let obs_df = anndata.read_obs()?; 
    let obs_names = anndata.obs_names();
    let var_df = anndata.read_var()?;
    let var_names = anndata.var_names();
    let x = anndata.x().get::<ArrayData>()?.unwrap();
    let imad = IMAnnData::new_extended(x, obs_names.into_vec(), var_names.into_vec(), obs_df, var_df)?;
    convert_axis_arrays_to_mem(anndata.obsm(), imad.obsm())?;
    convert_axis_arrays_to_mem(anndata.obsp(), imad.obsp())?;
    convert_axis_arrays_to_mem(anndata.varm(), imad.varm())?;
    convert_axis_arrays_to_mem(anndata.varp(), imad.varp())?;
    convert_uns_to_mem(anndata.uns(), imad.uns())?;
    anndata.close()?;
    Ok(imad)
}

fn convert_axis_arrays_to_mem<B: Backend>(axis_arr: &AxisArrays<B>, reference_element: IMAxisArrays) -> anyhow::Result<()> {
    if axis_arr.is_none() {
        return Ok(());
    }
    let x = axis_arr.inner();
    let iax = x.deref();
    let data = iax.deref();
    for (k,v) in data.iter() {
        let arr = v.get::<ArrayData>()?.unwrap();
        let im_arr = IMArrayElement::new(arr);
        reference_element.add_array(k.to_string(), im_arr)?;
    }
    Ok(())
}

fn convert_uns_to_mem<B: Backend>(elem_col: &ElemCollection<B>, reference_element: IMElementCollection) -> anyhow::Result<()> {
    if elem_col.is_none() {
        return Ok(());
    }
    let x = elem_col.inner();
    let iax = x.deref();
    let data = iax.deref();
    for (k,v) in data.iter() {
        let data = v.inner().data::<Data>();
        let d = Element::new(data?);
        reference_element.add_data(k.to_string(), d)?;
    }
    Ok(())
}
