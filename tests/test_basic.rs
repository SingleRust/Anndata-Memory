use anndata::{container::Axis, data::DynCsrMatrix, ArrayData};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use anndata_memory::{IMAnnData, IMArrayElement};

fn create_test_data() -> (ArrayData, Vec<String>, Vec<String>) {
    let nrows = 3;
    let ncols = 3;

    // Create a COO matrix with initial capacity for 4 non-zero entries
    let mut coo_matrix = CooMatrix::new(nrows, ncols);

    // Add some non-zero elements (row, col, value)
    coo_matrix.push(0, 0, 1.0);  // element at (0, 0) = 1.0
    coo_matrix.push(1, 2, 2.0);  // element at (1, 2) = 2.0
    coo_matrix.push(2, 1, 3.0);  // element at (2, 1) = 3.0
    coo_matrix.push(2, 2, 4.0);  // element at (2, 2) = 4.0

    // Optionally, you can convert the COO matrix to a more efficient CSR format
    let csr_matrix: CsrMatrix<f64> = CsrMatrix::from(&coo_matrix);
    
    let matrix = DynCsrMatrix::from(csr_matrix);
    let obs_names = vec!["obs1".to_string(), "obs2".to_string(), "obs3".to_string()];
    let var_names = vec!["var1".to_string(), "var2".to_string(), "var3".to_string()];
    (ArrayData::CsrMatrix(matrix), obs_names, var_names)
}

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