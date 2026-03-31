use anndata::{data::SelectInfoElem, ArrayData, HasShape};
use anndata_memory::IMAnnData;
use sprs::{CsMatI, TriMatI};

#[test]
fn test_type_integrity_i32() -> anyhow::Result<()> {
    let nrows = 5;
    let ncols = 5;
    let mut coo = TriMatI::<i32, u32>::new((nrows, ncols));
    coo.add_triplet(0, 0, 10);
    coo.add_triplet(1, 1, 20);
    
    let csr: CsMatI<i32, u32, u64> = coo.to_csr();
    let matrix = ArrayData::from(csr);
    
    let obs_names: Vec<String> = (0..nrows).map(|i| format!("cell_{}", i)).collect();
    let var_names: Vec<String> = (0..ncols).map(|i| format!("gene_{}", i)).collect();
    
    let adata = IMAnnData::new_basic(matrix, obs_names, var_names)?;
    
    assert_eq!(adata.n_obs(), 5);
    let x_data = adata.x().get_data()?;
    
    if let Ok(extracted) = CsMatI::<i32, u32, u64>::try_from(x_data) {
        assert_eq!(extracted.outer_view(0).unwrap().data()[0], 10);
    } else {
        panic!("Expected i32 CSR matrix");
    }
    
    Ok(())
}

#[test]
fn test_type_integrity_u8() -> anyhow::Result<()> {
    let nrows = 3;
    let ncols = 3;
    let mut coo = TriMatI::<u8, u32>::new((nrows, ncols));
    coo.add_triplet(0, 0, 1);
    coo.add_triplet(2, 2, 1);
    
    let csr: CsMatI<u8, u32, u64> = coo.to_csr();
    let matrix = ArrayData::from(csr);
    
    let obs_names: Vec<String> = (0..nrows).map(|i| format!("cell_{}", i)).collect();
    let var_names: Vec<String> = (0..ncols).map(|i| format!("gene_{}", i)).collect();
    
    let adata = IMAnnData::new_basic(matrix, obs_names, var_names)?;
    
    let x_data = adata.x().get_data()?;
    if let Ok(extracted) = CsMatI::<u8, u32, u64>::try_from(x_data) {
        assert_eq!(extracted.outer_view(0).unwrap().data()[0], 1);
    } else {
        panic!("Expected u8 CSR matrix");
    }
    
    Ok(())
}

#[test]
fn test_non_canonical_ingestion() -> anyhow::Result<()> {
    let nrows = 3;
    let ncols = 3;
    
    // Create a non-canonical matrix by adding duplicate triplets
    let mut coo = TriMatI::<f64, u32>::new((nrows, ncols));
    coo.add_triplet(0, 0, 1.0);
    coo.add_triplet(0, 0, 2.0); // Duplicate entry
    coo.add_triplet(1, 1, 5.0);
    
    // to_csr() handles canonicalization by summing duplicates
    let csr: CsMatI<f64, u32, u64> = coo.to_csr();
    let matrix = ArrayData::from(csr);
    
    let obs_names: Vec<String> = (0..nrows).map(|i| format!("cell_{}", i)).collect();
    let var_names: Vec<String> = (0..ncols).map(|i| format!("gene_{}", i)).collect();
    
    let adata = IMAnnData::new_basic(matrix, obs_names, var_names)?;
    
    let x_data = adata.x().get_data()?;
    if let Ok(extracted) = CsMatI::<f64, u32, u64>::try_from(x_data) {
        // Value at (0,0) should be 3.0 (1.0 + 2.0)
        assert_eq!(extracted.outer_view(0).unwrap().data()[0], 3.0);
    }
    
    Ok(())
}

#[test]
fn test_empty_dimensions() -> anyhow::Result<()> {
    // 0x0 case
    let matrix_0x0 = ArrayData::from(ndarray::Array2::<f64>::zeros((0, 0)));
    let adata_0x0 = IMAnnData::new_basic(matrix_0x0, vec![], vec![])?;
    assert_eq!(adata_0x0.n_obs(), 0);
    assert_eq!(adata_0x0.n_vars(), 0);
    
    // 1000x1000 empty sparse case
    let nrows = 1000;
    let ncols = 1000;
    let coo = TriMatI::<f64, u32>::new((nrows, ncols));
    let csr: CsMatI<f64, u32, u64> = coo.to_csr();
    let matrix = ArrayData::from(csr);
    
    let obs_names: Vec<String> = (0..nrows).map(|i| format!("cell_{}", i)).collect();
    let var_names: Vec<String> = (0..ncols).map(|i| format!("gene_{}", i)).collect();
    
    let adata = IMAnnData::new_basic(matrix, obs_names, var_names)?;
    assert_eq!(adata.n_obs(), 1000);
    assert_eq!(adata.x().get_data()?.shape().as_ref(), &[1000, 1000]);
    
    // Subset from empty
    let selection = SelectInfoElem::from(0..10);
    let subset = adata.subset(&[&selection, &selection])?;
    assert_eq!(subset.n_obs(), 10);
    
    Ok(())
}
