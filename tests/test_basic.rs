use std::ops::Deref;

use anndata::{
    container::Axis,
    data::{SelectInfoElem},
    ArrayData,
};
use anndata_memory::{IMAnnData, IMArrayElement};
use sprs::{CsMatI, TriMatI};
use ndarray::Array2;
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, SeedableRng};

fn create_test_data() -> (ArrayData, Vec<String>, Vec<String>) {
    let nrows = 3;
    let ncols = 3;

    let mut coo_matrix = TriMatI::<f64, u32>::new((nrows, ncols));

    coo_matrix.add_triplet(0, 0, 1.0);
    coo_matrix.add_triplet(1, 2, 2.0);
    coo_matrix.add_triplet(2, 1, 3.0);
    coo_matrix.add_triplet(2, 2, 4.0);

    let csr_matrix: CsMatI<f64, u32, u64> = coo_matrix.to_csr();

    let obs_names = vec!["obs1".to_string(), "obs2".to_string(), "obs3".to_string()];
    let var_names = vec!["var1".to_string(), "var2".to_string(), "var3".to_string()];
    (csr_matrix.into(), obs_names, var_names)
}

fn create_random_test_data(
    nrows: usize,
    ncols: usize,
    density: f64,
    seed: Option<u64>,
) -> (ArrayData, Vec<String>, Vec<String>) {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let nnz = ((nrows * ncols) as f64 * density) as usize;

    let mut coo_matrix = TriMatI::<f64, u32>::new((nrows, ncols));

    let row_dist = Uniform::from(0..nrows as u32);
    let col_dist = Uniform::from(0..ncols as u32);
    let value_dist = Uniform::from(0.0..10.0);

    let mut filled_positions = std::collections::HashSet::new();

    let mut attempts = 0;
    let max_attempts = nnz * 10;

    while filled_positions.len() < nnz && attempts < max_attempts {
        let row = row_dist.sample(&mut rng);
        let col = col_dist.sample(&mut rng);

        if filled_positions.insert((row, col)) {
            let value = value_dist.sample(&mut rng);
            coo_matrix.add_triplet(row as usize, col as usize, value);
        }

        attempts += 1;
    }

    let csr_matrix: CsMatI<f64, u32, u64> = coo_matrix.to_csr();

    let obs_names: Vec<String> = (0..nrows).map(|i| format!("cell_{}", i)).collect();

    let var_names: Vec<String> = (0..ncols).map(|i| format!("gene_{}", i)).collect();

    (csr_matrix.into(), obs_names, var_names)
}

// convert_matrix_format is disabled during sprs transition
// #[test]
// fn test_convert_matrix_format() { ... }

#[test]
fn test_new_basic() {
    let (matrix, obs_names, var_names) = create_test_data();
    let adata = IMAnnData::new_basic(matrix, obs_names, var_names).unwrap();

    assert_eq!(adata.n_obs(), 3);
    assert_eq!(adata.n_vars(), 3);
}

#[test]
fn test_getters() {
    let (matrix, obs_names, var_names) = create_test_data();
    let adata = IMAnnData::new_basic(matrix, obs_names, var_names).unwrap();

    assert_eq!(adata.n_obs(), 3);
    assert_eq!(adata.n_vars(), 3);

    let x = adata.x();
    let shape = x.get_shape().unwrap();
    assert_eq!(vec![shape[0], shape[1]], vec![3, 3]);

    let obs = adata.obs();
    assert_eq!(obs.get_data().height(), 3);

    let var = adata.var();
    assert_eq!(var.get_data().height(), 3);
}

#[test]
fn test_add_and_get_layer() {
    let (matrix, obs_names, var_names) = create_test_data();
    let mut adata = IMAnnData::new_basic(matrix.clone(), obs_names, var_names).unwrap();

    let layer_name = "test_layer".to_string();
    let layer_data = IMArrayElement::new(matrix);

    adata.add_layer(layer_name.clone(), layer_data).unwrap();

    let retrieved_layer = adata.get_layer(&layer_name).unwrap();
    let shape = retrieved_layer.get_shape().unwrap();
    assert_eq!(vec![shape[0], shape[1]], vec![3, 3]);
}

#[test]
fn test_remove_layer() {
    let (matrix, obs_names, var_names) = create_test_data();
    let mut adata = IMAnnData::new_basic(matrix.clone(), obs_names, var_names).unwrap();

    let layer_name = "test_layer".to_string();
    let layer_data = IMArrayElement::new(matrix);

    adata.add_layer(layer_name.clone(), layer_data).unwrap();
    let removed_layer = adata.remove_layer(&layer_name).unwrap();

    let shape = removed_layer.get_shape().unwrap();
    assert_eq!(vec![shape[0], shape[1]], vec![3, 3]);

    assert!(adata.get_layer(&layer_name).is_err());
}

#[test]
fn test_obsm_varm() {
    let (matrix, obs_names, var_names) = create_test_data();
    let adata = IMAnnData::new_basic(matrix, obs_names, var_names).unwrap();

    let obsm = adata.obsm();
    assert_eq!(obsm.axis(), Axis::Row);
    assert_eq!(obsm.dimensions().0.get(), 3);

    let varm = adata.varm();
    assert_eq!(varm.axis(), Axis::Row);
    assert_eq!(varm.dimensions().0.get(), 3);
}

#[test]
fn test_obsp_varp() {
    let (matrix, obs_names, var_names) = create_test_data();
    let adata = IMAnnData::new_basic(matrix, obs_names, var_names).unwrap();

    let obsp = adata.obsp();
    assert_eq!(obsp.axis(), Axis::Pairwise);
    assert_eq!(obsp.dimensions().0.get(), 3);

    let varp = adata.varp();
    assert_eq!(varp.axis(), Axis::Pairwise);
    assert_eq!(varp.dimensions().0.get(), 3);
}

#[test]
fn test_uns() {
    let (matrix, obs_names, var_names) = create_test_data();
    let adata = IMAnnData::new_basic(matrix, obs_names, var_names).unwrap();

    let uns = adata.uns();
    assert!(uns.get_data("test_key").is_err());
}

#[test]
fn test_subset_inplace_basic() {
    let (matrix, obs_names, var_names) = create_random_test_data(10, 20, 0.3, Some(42));
    let mut adata = IMAnnData::new_basic(matrix, obs_names.clone(), var_names.clone()).unwrap();

    let original_n_obs = adata.n_obs();
    let original_n_vars = adata.n_vars();

    let obs_selection = SelectInfoElem::Index(vec![0, 2, 4, 6, 8]);
    let var_selection = SelectInfoElem::from(0..10);

    adata
        .subset_inplace(&[&obs_selection, &var_selection])
        .unwrap();

    assert_eq!(adata.n_obs(), 5);
    assert_eq!(adata.n_vars(), 10);
    assert_ne!(adata.n_obs(), original_n_obs);
    assert_ne!(adata.n_vars(), original_n_vars);

    let new_obs_names = adata.obs_names();
    assert_eq!(
        new_obs_names,
        vec!["cell_0", "cell_2", "cell_4", "cell_6", "cell_8"]
    );

    let new_var_names = adata.var_names();
    let expected_var_names: Vec<String> = (0..10).map(|i| format!("gene_{}", i)).collect();
    assert_eq!(new_var_names, expected_var_names);
}

#[test]
fn test_subset_inplace_with_layers() {
    let (matrix, obs_names, var_names) = create_random_test_data(15, 25, 0.2, Some(123));
    let mut adata = IMAnnData::new_basic(matrix.clone(), obs_names, var_names).unwrap();

    let (layer_matrix, _, _) = create_random_test_data(15, 25, 0.15, Some(456));
    adata
        .add_layer("normalized".to_string(), IMArrayElement::new(layer_matrix))
        .unwrap();

    let obs_selection = SelectInfoElem::Index(vec![1, 3, 5, 7, 9, 11, 13]);
    let var_selection = SelectInfoElem::from(5..15);

    adata
        .subset_inplace(&[&obs_selection, &var_selection])
        .unwrap();

    assert_eq!(adata.n_obs(), 7);
    assert_eq!(adata.n_vars(), 10);

    let layer = adata.get_layer("normalized").unwrap();
    let layer_shape = layer.get_shape().unwrap();
    assert_eq!(layer_shape.as_ref(), &[7, 10]);
}

#[test]
fn test_subset_inplace_with_obsm_obsp() {
    let (matrix, obs_names, var_names) = create_random_test_data(20, 30, 0.25, Some(789));
    let mut adata = IMAnnData::new_basic(matrix, obs_names, var_names).unwrap();

    let pca_data = Array2::<f64>::from_shape_fn((20, 10), |(i, j)| (i * 10 + j) as f64);
    adata
        .obsm()
        .add_array(
            "X_pca".to_string(),
            IMArrayElement::new(ArrayData::from(pca_data)),
        )
        .unwrap();

    let similarity_data =
        Array2::<f64>::from_shape_fn((20, 20), |(i, j)| if i == j { 1.0 } else { 0.1 });
    adata
        .obsp()
        .add_array(
            "similarity".to_string(),
            IMArrayElement::new(ArrayData::from(similarity_data)),
        )
        .unwrap();

    let obs_indices = vec![0, 5, 10, 15, 19];
    let obs_selection = SelectInfoElem::Index(obs_indices.clone());
    let var_selection = SelectInfoElem::full();

    adata
        .subset_inplace(&[&obs_selection, &var_selection])
        .unwrap();

    let pca = adata.obsm().get_array("X_pca").unwrap();
    let pca_shape = pca.get_shape().unwrap();
    assert_eq!(pca_shape.as_ref(), &[5, 10]);

    let sim = adata.obsp().get_array("similarity").unwrap();
    let sim_shape = sim.get_shape().unwrap();
    assert_eq!(sim_shape.as_ref(), &[5, 5]);
}
