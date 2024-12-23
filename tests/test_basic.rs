use std::ops::Deref;

use anndata::{container::Axis, data::DynCsrMatrix, ArrayData};
use anndata_memory::{IMAnnData, IMArrayElement};
use nalgebra_sparse::{CooMatrix, CsrMatrix};

fn create_test_data() -> (ArrayData, Vec<String>, Vec<String>) {
    let nrows = 3;
    let ncols = 3;

    // Create a COO matrix with initial capacity for 4 non-zero entries
    let mut coo_matrix = CooMatrix::new(nrows, ncols);

    // Add some non-zero elements (row, col, value)
    coo_matrix.push(0, 0, 1.0); // element at (0, 0) = 1.0
    coo_matrix.push(1, 2, 2.0); // element at (1, 2) = 2.0
    coo_matrix.push(2, 1, 3.0); // element at (2, 1) = 3.0
    coo_matrix.push(2, 2, 4.0); // element at (2, 2) = 4.0

    // Optionally, you can convert the COO matrix to a more efficient CSR format
    let csr_matrix: CsrMatrix<f64> = CsrMatrix::from(&coo_matrix);

    let matrix = DynCsrMatrix::from(csr_matrix);
    let obs_names = vec!["obs1".to_string(), "obs2".to_string(), "obs3".to_string()];
    let var_names = vec!["var1".to_string(), "var2".to_string(), "var3".to_string()];
    (ArrayData::CsrMatrix(matrix), obs_names, var_names)
}

#[test]
fn test_convert_matrix_format() {
    // Create test data using CooMatrix
    let coo = CooMatrix::try_from_triplets(
        5,
        4,                                  // 5x4 matrix
        vec![0, 1, 1, 2, 3, 4],             // row indices
        vec![0, 1, 2, 3, 1, 3],             // column indices
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // values
    )
    .unwrap();

    // Convert to CSR format
    let csr = CsrMatrix::from(&coo);
    let array_data = ArrayData::CsrMatrix(DynCsrMatrix::F64(csr));
    let matrix = IMArrayElement::new(array_data);

    // Convert CSR to CSC
    matrix.convert_matrix_format().unwrap();

    // Verify it's now CSC
    {
        let read_guard = matrix.0.read_inner();
        match read_guard.deref() {
            ArrayData::CscMatrix(_) => (),
            _ => panic!("Matrix should be in CSC format"),
        }
    } // read_guard is dropped here

    // Convert CSC back to CSR
    matrix.convert_matrix_format().unwrap();

    // Verify it's back to CSR and check content
    {
        let read_guard = matrix.0.read_inner();
        match read_guard.deref() {
            ArrayData::CsrMatrix(csr) => {
                if let DynCsrMatrix::F64(m) = csr {
                    // Verify the matrix content is preserved
                    assert_eq!(m.nrows(), 5);
                    assert_eq!(m.ncols(), 4);
                    assert_eq!(m.nnz(), 6);

                    // Check specific values
                    assert_eq!(
                        m.triplet_iter()
                            .find(|&(i, j, &_v)| i == 0 && j == 0)
                            .map(|(_, _, &v)| v),
                        Some(1.0)
                    );
                    assert_eq!(
                        m.triplet_iter()
                            .find(|&(i, j, &_v)| i == 1 && j == 1)
                            .map(|(_, _, &v)| v),
                        Some(2.0)
                    );
                    assert_eq!(
                        m.triplet_iter()
                            .find(|&(i, j, &_v)| i == 1 && j == 2)
                            .map(|(_, _, &v)| v),
                        Some(3.0)
                    );
                    assert_eq!(
                        m.triplet_iter()
                            .find(|&(i, j, &_v)| i == 2 && j == 3)
                            .map(|(_, _, &v)| v),
                        Some(4.0)
                    );
                    assert_eq!(
                        m.triplet_iter()
                            .find(|&(i, j, &_v)| i == 3 && j == 1)
                            .map(|(_, _, &v)| v),
                        Some(5.0)
                    );
                    assert_eq!(
                        m.triplet_iter()
                            .find(|&(i, j, &_v)| i == 4 && j == 3)
                            .map(|(_, _, &v)| v),
                        Some(6.0)
                    );
                } else {
                    panic!("Expected F64 matrix");
                }
            }
            _ => panic!("Matrix should be in CSR format"),
        }
    } // read_guard is dropped here
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
