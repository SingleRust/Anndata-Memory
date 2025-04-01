use anndata::{AnnData, AnnDataOp, ArrayData, AxisArraysOp, Backend};
use anndata_hdf5::H5;
use anndata_memory::*;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use ndarray::Array2;
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::tempdir;

#[test]
fn test_round_trip_conversion() -> anyhow::Result<()> {
    let n_rows = 10;
    let n_cols = 20;
    let density = 0.3;
    let nnz = (n_rows * n_cols) as f64 * density;
    let mut rng = StdRng::seed_from_u64(42);

    let mut coo_matrix = CooMatrix::new(n_rows, n_cols);

    for _ in 0..(nnz as usize) {
        let row = rng.gen_range(0..n_rows);
        let col = rng.gen_range(0..n_cols);
        let value = rng.gen::<f64>();
        coo_matrix.push(row, col, value);
    }

    let csr_matrix: CsrMatrix<f64> = (&coo_matrix).into();

    let im_array = IMArrayElement::new(ArrayData::from(csr_matrix));

    let obs_names: Vec<String> = (0..n_rows).map(|i| format!("cell_{}", i)).collect();

    let cell_types = vec!["T cell", "B cell", "NK cell", "Monocyte", "Dendritic"];

    let obs_cell_types: Vec<&str> = (0..n_rows)
        .map(|i| cell_types[i % cell_types.len()])
        .collect();

    let obs_df = DataFrame::new(vec![Column::from(Series::new(
        PlSmallStr::from("cell_type"),
        obs_cell_types,
    ))])?;
    let obs_element = IMDataFrameElement::new(obs_df, obs_names.clone().into());

    let var_names: Vec<String> = (0..n_cols).map(|i| format!("gene_{}", i)).collect();
    let gene_types = vec!["protein_coding", "lincRNA", "pseudogene", "miRNA"];
    let var_gene_types: Vec<&str> = (0..n_cols)
        .map(|i| gene_types[i % gene_types.len()])
        .collect();

    let var_df = DataFrame::new(vec![Column::from(Series::new(
        PlSmallStr::from("gene_type"),
        var_gene_types,
    ))])?;
    let var_element = IMDataFrameElement::new(var_df, var_names.clone().into());

    let mut im_anndata = IMAnnData::new(im_array, obs_element, var_element)?;

    let mut coo_norm = CooMatrix::new(n_rows, n_cols);

    for _ in 0..(nnz as usize) {
        let row = rng.gen_range(0..n_rows);
        let col = rng.gen_range(0..n_cols);
        let value = rng.gen::<f64>() * 0.1; // Smaller values
        coo_norm.push(row, col, value);
    }

    let csr_norm: CsrMatrix<f64> = CsrMatrix::from(&coo_norm);

    let layer_im_array = IMArrayElement::new(ArrayData::from(csr_norm));
    im_anndata.add_layer("normalized".to_string(), layer_im_array)?;

    let n_components = 2;
    let pca_data: Array2<f64> = Array2::from_shape_fn((n_rows, n_components), |_| rng.gen());
    let pca_array_data = ArrayData::from(pca_data);
    let pca_im_array = IMArrayElement::new(pca_array_data);
    im_anndata
        .obsm()
        .add_array("X_pca".to_string(), pca_im_array)?;

    let temp_dir = tempdir()?;
    let h5_path = temp_dir.path().join("test_anndata.h5ad");
    let h5_anndata = convert_to_new_backed_h5(&im_anndata, &h5_path)?;

    assert_eq!(h5_anndata.n_obs(), n_rows);
    assert_eq!(h5_anndata.n_vars(), n_cols);

    let h5_obs_names = h5_anndata.obs_names().into_vec();
    assert_eq!(h5_obs_names, obs_names);

    let h5_var_names = h5_anndata.var_names().into_vec();
    assert_eq!(h5_var_names, var_names);

    assert!(h5_anndata
        .layers()
        .keys()
        .contains(&"normalized".to_string()));

    assert!(h5_anndata.obsm().keys().contains(&"X_pca".to_string()));

    h5_anndata.close()?;

    // Now test reading back from H5 to in-memory
    let h5_file = H5::open(&h5_path)?;
    let reopened_anndata = AnnData::<H5>::open(h5_file)?;
    let im_anndata2 = convert_to_in_memory(reopened_anndata)?;

    println!("{:?}", im_anndata2
        .layers()
        .keys());

    // Verify round-trip conversion
    assert_eq!(im_anndata2.n_obs(), n_rows);
    assert_eq!(im_anndata2.n_vars(), n_cols);
    assert_eq!(im_anndata2.obs_names(), obs_names);
    assert_eq!(im_anndata2.var_names(), var_names);
    assert!(im_anndata2
        .layers()
        .keys()
        .contains(&"normalized".to_string()));
    assert!(im_anndata2.obsm().keys().contains(&"X_pca".to_string()));

    Ok(())
}
