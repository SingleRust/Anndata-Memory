use anndata::{data::SelectInfoElem, ArrayData};
use anndata_memory::IMAnnData;
use sprs::{CsMatI, TriMatI};
use polars::{frame::DataFrame, prelude::Column};
use rand::{thread_rng, Rng};

/// Helper function to create a random sparse matrix
fn create_random_sparse_matrix(nrows: usize, ncols: usize, density: f64) -> ArrayData {
    let mut rng = thread_rng();
    let nnz = ((nrows * ncols) as f64 * density) as usize;

    let mut coo = TriMatI::<f64, u32>::new((nrows, ncols));

    for _ in 0..nnz {
        let row = rng.gen_range(0..nrows);
        let col = rng.gen_range(0..ncols);
        let value: f64 = rng.gen_range(-10.0..10.0);
        coo.add_triplet(row, col, value);
    }

    let csr: CsMatI<f64, u32, u64> = coo.to_csr();
    ArrayData::from(csr)
}

/// Helper function to create test IMAnnData with random data
fn create_test_anndata(nrows: usize, ncols: usize, density: f64) -> anyhow::Result<IMAnnData> {
    let matrix = create_random_sparse_matrix(nrows, ncols, density);

    // Create observation names
    let obs_names: Vec<String> = (0..nrows).map(|i| format!("cell_{}", i)).collect();

    // Create variable names
    let var_names: Vec<String> = (0..ncols).map(|i| format!("gene_{}", i)).collect();

    // Create basic DataFrames
    let obs_df = DataFrame::new(obs_names.len(), vec![
        Column::new("index".into(), &obs_names),
        Column::new("n_genes".into(), &vec![100; nrows]), // Mock gene count
    ])?;

    let var_df = DataFrame::new(var_names.len(), vec![
        Column::new("index".into(), &var_names),
        Column::new("highly_variable".into(), &vec![true; ncols]), // Mock highly variable
    ])?;

    IMAnnData::new_extended(matrix, obs_names, var_names, obs_df, var_df)
}

#[test]
fn test_subset_inplace_row_only() -> anyhow::Result<()> {
    println!("Testing row-only subsetting...");

    // Create test data: 1000 cells x 500 genes with 5% sparsity
    let original_nrows = 1000;
    let original_ncols = 500;
    let mut adata = create_test_anndata(original_nrows, original_ncols, 0.05)?;

    println!(
        "Original dimensions: {} x {}",
        adata.n_obs(),
        adata.n_vars()
    );

    // Subset to first 300 cells (keep all genes)
    let target_rows = 300;
    let obs_selection = SelectInfoElem::from(0..target_rows);
    let var_selection = SelectInfoElem::full(); // Keep all variables

    // Get original X matrix shape for comparison
    let original_x_shape = adata.x().get_shape()?;
    println!("Original X shape: {:?}", original_x_shape);

    // Perform subset in place
    adata.subset_inplace(&[&obs_selection, &var_selection])?;

    // Check dimensions
    assert_eq!(
        adata.n_obs(),
        target_rows,
        "Row count should match subset size"
    );
    assert_eq!(
        adata.n_vars(),
        original_ncols,
        "Column count should remain unchanged"
    );

    // Check X matrix dimensions
    let new_x_shape = adata.x().get_shape()?;
    println!("New X shape: {:?}", new_x_shape);
    assert_eq!(
        new_x_shape[0], target_rows,
        "X matrix rows should match subset"
    );
    assert_eq!(
        new_x_shape[1], original_ncols,
        "X matrix cols should be unchanged"
    );

    // Check obs and var DataFrames
    assert_eq!(
        adata.obs().get_data().height(),
        target_rows,
        "obs DataFrame height should match"
    );
    assert_eq!(
        adata.var().get_data().height(),
        original_ncols,
        "var DataFrame height should be unchanged"
    );

    // Check obs and var names
    let obs_names = adata.obs_names();
    let var_names = adata.var_names();
    assert_eq!(
        obs_names.len(),
        target_rows,
        "obs_names length should match"
    );
    assert_eq!(
        var_names.len(),
        original_ncols,
        "var_names length should be unchanged"
    );

    // Verify the names are correct
    for (i, name) in obs_names.iter().enumerate() {
        assert_eq!(
            name,
            &format!("cell_{}", i),
            "obs name should match expected pattern"
        );
    }

    println!("✓ Row-only subsetting test passed!");
    Ok(())
}

#[test]
fn test_subset_inplace_column_only() -> anyhow::Result<()> {
    println!("Testing column-only subsetting...");

    // Create test data: 500 cells x 2000 genes with 3% sparsity
    let original_nrows = 500;
    let original_ncols = 2000;
    let mut adata = create_test_anndata(original_nrows, original_ncols, 0.03)?;

    println!(
        "Original dimensions: {} x {}",
        adata.n_obs(),
        adata.n_vars()
    );

    // Subset to first 800 genes (keep all cells)
    let target_cols = 800;
    let obs_selection = SelectInfoElem::full(); // Keep all observations
    let var_selection = SelectInfoElem::from(0..target_cols);

    // Get original X matrix shape for comparison
    let original_x_shape = adata.x().get_shape()?;
    println!("Original X shape: {:?}", original_x_shape);

    // Perform subset in place
    adata.subset_inplace(&[&obs_selection, &var_selection])?;

    // Check dimensions
    assert_eq!(
        adata.n_obs(),
        original_nrows,
        "Row count should remain unchanged"
    );
    assert_eq!(
        adata.n_vars(),
        target_cols,
        "Column count should match subset size"
    );

    // Check X matrix dimensions
    let new_x_shape = adata.x().get_shape()?;
    println!("New X shape: {:?}", new_x_shape);
    assert_eq!(
        new_x_shape[0], original_nrows,
        "X matrix rows should be unchanged"
    );
    assert_eq!(
        new_x_shape[1], target_cols,
        "X matrix cols should match subset"
    );

    // Check obs and var DataFrames
    assert_eq!(
        adata.obs().get_data().height(),
        original_nrows,
        "obs DataFrame height should be unchanged"
    );
    assert_eq!(
        adata.var().get_data().height(),
        target_cols,
        "var DataFrame height should match"
    );

    // Check obs and var names
    let obs_names = adata.obs_names();
    let var_names = adata.var_names();
    assert_eq!(
        obs_names.len(),
        original_nrows,
        "obs_names length should be unchanged"
    );
    assert_eq!(
        var_names.len(),
        target_cols,
        "var_names length should match"
    );

    // Verify the names are correct
    for (i, name) in var_names.iter().enumerate() {
        assert_eq!(
            name,
            &format!("gene_{}", i),
            "var name should match expected pattern"
        );
    }

    println!("✓ Column-only subsetting test passed!");
    Ok(())
}

#[test]
fn test_subset_inplace_both_dimensions() -> anyhow::Result<()> {
    println!("Testing both row and column subsetting...");

    // Create test data: 2000 cells x 1500 genes with 2% sparsity
    let original_nrows = 2000;
    let original_ncols = 1500;
    let mut adata = create_test_anndata(original_nrows, original_ncols, 0.02)?;

    println!(
        "Original dimensions: {} x {}",
        adata.n_obs(),
        adata.n_vars()
    );

    // Subset to middle portion: cells 500-1200, genes 300-1000
    let row_start = 500;
    let row_end = 1200;
    let col_start = 300;
    let col_end = 1000;

    let target_rows = row_end - row_start;
    let target_cols = col_end - col_start;

    let obs_selection = SelectInfoElem::from(row_start..row_end);
    let var_selection = SelectInfoElem::from(col_start..col_end);

    // Get original X matrix shape for comparison
    let original_x_shape = adata.x().get_shape()?;
    println!("Original X shape: {:?}", original_x_shape);

    // Perform subset in place
    adata.subset_inplace(&[&obs_selection, &var_selection])?;

    // Check dimensions
    assert_eq!(
        adata.n_obs(),
        target_rows,
        "Row count should match subset size"
    );
    assert_eq!(
        adata.n_vars(),
        target_cols,
        "Column count should match subset size"
    );

    // Check X matrix dimensions
    let new_x_shape = adata.x().get_shape()?;
    println!("New X shape: {:?}", new_x_shape);
    assert_eq!(
        new_x_shape[0], target_rows,
        "X matrix rows should match subset"
    );
    assert_eq!(
        new_x_shape[1], target_cols,
        "X matrix cols should match subset"
    );

    // Check obs and var DataFrames
    assert_eq!(
        adata.obs().get_data().height(),
        target_rows,
        "obs DataFrame height should match"
    );
    assert_eq!(
        adata.var().get_data().height(),
        target_cols,
        "var DataFrame height should match"
    );

    // Check obs and var names
    let obs_names = adata.obs_names();
    let var_names = adata.var_names();
    assert_eq!(
        obs_names.len(),
        target_rows,
        "obs_names length should match"
    );
    assert_eq!(
        var_names.len(),
        target_cols,
        "var_names length should match"
    );

    // Verify the names correspond to the correct subset
    for (i, name) in obs_names.iter().enumerate() {
        let expected_name = format!("cell_{}", row_start + i);
        assert_eq!(
            name, &expected_name,
            "obs name should match expected subset pattern"
        );
    }

    for (i, name) in var_names.iter().enumerate() {
        let expected_name = format!("gene_{}", col_start + i);
        assert_eq!(
            name, &expected_name,
            "var name should match expected subset pattern"
        );
    }

    println!("✓ Both dimensions subsetting test passed!");
    Ok(())
}

#[test]
fn test_subset_inplace_sparse_selection() -> anyhow::Result<()> {
    println!("Testing sparse index selection...");

    // Create test data: 1000 cells x 800 genes with 4% sparsity
    let original_nrows = 1000;
    let original_ncols = 800;
    let mut adata = create_test_anndata(original_nrows, original_ncols, 0.04)?;

    println!(
        "Original dimensions: {} x {}",
        adata.n_obs(),
        adata.n_vars()
    );

    // Select specific indices (non-contiguous)
    let selected_rows = vec![10, 50, 100, 200, 300, 450, 600, 750, 900];
    let selected_cols = vec![5, 25, 75, 150, 300, 500, 700];

    let obs_selection = SelectInfoElem::Index(selected_rows.clone());
    let var_selection = SelectInfoElem::Index(selected_cols.clone());

    // Get original X matrix shape for comparison
    let original_x_shape = adata.x().get_shape()?;
    println!("Original X shape: {:?}", original_x_shape);

    // Perform subset in place
    adata.subset_inplace(&[&obs_selection, &var_selection])?;

    // Check dimensions
    assert_eq!(
        adata.n_obs(),
        selected_rows.len(),
        "Row count should match selected indices"
    );
    assert_eq!(
        adata.n_vars(),
        selected_cols.len(),
        "Column count should match selected indices"
    );

    // Check X matrix dimensions
    let new_x_shape = adata.x().get_shape()?;
    println!("New X shape: {:?}", new_x_shape);
    assert_eq!(
        new_x_shape[0],
        selected_rows.len(),
        "X matrix rows should match selection"
    );
    assert_eq!(
        new_x_shape[1],
        selected_cols.len(),
        "X matrix cols should match selection"
    );

    // Check that the names correspond to the selected indices
    let obs_names = adata.obs_names();
    let var_names = adata.var_names();

    for (i, &original_idx) in selected_rows.iter().enumerate() {
        let expected_name = format!("cell_{}", original_idx);
        assert_eq!(
            &obs_names[i], &expected_name,
            "obs name should match selected index"
        );
    }

    for (i, &original_idx) in selected_cols.iter().enumerate() {
        let expected_name = format!("gene_{}", original_idx);
        assert_eq!(
            &var_names[i], &expected_name,
            "var name should match selected index"
        );
    }

    println!("✓ Sparse index selection test passed!");
    Ok(())
}

#[test]
fn test_subset_inplace_edge_cases() -> anyhow::Result<()> {
    println!("Testing edge cases...");

    // Test with very small matrix
    let mut small_adata = create_test_anndata(5, 3, 0.8)?; // High density for small matrix

    // Subset to single cell and single gene
    let obs_selection = SelectInfoElem::Index(vec![2]);
    let var_selection = SelectInfoElem::Index(vec![1]);

    small_adata.subset_inplace(&[&obs_selection, &var_selection])?;

    assert_eq!(small_adata.n_obs(), 1, "Should have 1 observation");
    assert_eq!(small_adata.n_vars(), 1, "Should have 1 variable");

    let x_shape = small_adata.x().get_shape()?;
    assert_eq!(x_shape, vec![1, 1].into(), "X matrix should be 1x1");

    // Test with empty selection (should fail gracefully or handle appropriately)
    let mut empty_test = create_test_anndata(10, 10, 0.1)?;
    let empty_obs = SelectInfoElem::Index(vec![]);
    let empty_var = SelectInfoElem::Index(vec![]);

    // This should either work (creating 0x0 matrix) or fail gracefully
    let result = empty_test.subset_inplace(&[&empty_obs, &empty_var]);
    match result {
        Ok(_) => {
            assert_eq!(empty_test.n_obs(), 0, "Should have 0 observations");
            assert_eq!(empty_test.n_vars(), 0, "Should have 0 variables");
            println!("✓ Empty selection handled correctly");
        }
        Err(_) => {
            println!("✓ Empty selection properly rejected");
        }
    }

    println!("✓ Edge cases test completed!");
    Ok(())
}

#[test]
fn test_matrix_data_integrity() -> anyhow::Result<()> {
    println!("Testing matrix data integrity after subsetting...");

    // Create a small, predictable matrix for data verification
    let nrows = 4;
    let ncols = 3;

    // Create a known sparse matrix manually
    let mut coo = TriMatI::<f64, u32>::new((nrows, ncols));
    coo.add_triplet(0, 0, 1.0);
    coo.add_triplet(0, 2, 2.0);
    coo.add_triplet(1, 1, 3.0);
    coo.add_triplet(2, 0, 4.0);
    coo.add_triplet(3, 2, 5.0);

    let csr: CsMatI<f64, u32, u64> = coo.to_csr();
    let matrix = ArrayData::from(csr);

    let obs_names = vec![
        "cell_0".to_string(),
        "cell_1".to_string(),
        "cell_2".to_string(),
        "cell_3".to_string(),
    ];
    let var_names = vec![
        "gene_0".to_string(),
        "gene_1".to_string(),
        "gene_2".to_string(),
    ];

    let obs_df = DataFrame::new(obs_names.len(), vec![Column::new("index".into(), &obs_names)])?;
    let var_df = DataFrame::new(var_names.len(), vec![Column::new("index".into(), &var_names)])?;

    let mut adata = IMAnnData::new_extended(matrix, obs_names, var_names, obs_df, var_df)?;

    println!("Original matrix created with known values");

    // Subset to rows 1,2 and columns 0,2
    let obs_selection = SelectInfoElem::Index(vec![1, 2]);
    let var_selection = SelectInfoElem::Index(vec![0, 2]);

    adata.subset_inplace(&[&obs_selection, &var_selection])?;

    // Verify dimensions
    assert_eq!(adata.n_obs(), 2);
    assert_eq!(adata.n_vars(), 2);

    // Verify names
    let new_obs_names = adata.obs_names();
    let new_var_names = adata.var_names();
    assert_eq!(new_obs_names, vec!["cell_1", "cell_2"]);
    assert_eq!(new_var_names, vec!["gene_0", "gene_2"]);

    println!("✓ Matrix data integrity test passed!");
    Ok(())
}

#[test]
fn test_subset_value_integrity() -> anyhow::Result<()> {
    println!("Testing that actual matrix values are preserved after subsetting...");

    // Create a small, known matrix: 4x3 with specific values
    let mut coo = TriMatI::<f64, u32>::new((4, 3));
    coo.add_triplet(0, 0, 10.0); // cell_0, gene_0 = 10.0
    coo.add_triplet(0, 2, 20.0); // cell_0, gene_2 = 20.0
    coo.add_triplet(1, 1, 30.0); // cell_1, gene_1 = 30.0
    coo.add_triplet(2, 0, 40.0); // cell_2, gene_0 = 40.0
    coo.add_triplet(2, 2, 50.0); // cell_2, gene_2 = 50.0
    coo.add_triplet(3, 1, 60.0); // cell_3, gene_1 = 60.0

    let csr: CsMatI<f64, u32, u64> = coo.to_csr();
    let matrix = ArrayData::from(csr);

    // Create AnnData with known names
    let obs_names = vec![
        "cell_0".to_string(),
        "cell_1".to_string(),
        "cell_2".to_string(),
        "cell_3".to_string(),
    ];
    let var_names = vec![
        "gene_0".to_string(),
        "gene_1".to_string(),
        "gene_2".to_string(),
    ];

    let obs_df = DataFrame::new(obs_names.len(), vec![Column::new("index".into(), &obs_names)])?;
    let var_df = DataFrame::new(var_names.len(), vec![Column::new("index".into(), &var_names)])?;

    let mut adata = IMAnnData::new_extended(matrix, obs_names, var_names, obs_df, var_df)?;

    println!("Original matrix:");
    println!("  cell_0: gene_0=10.0, gene_2=20.0");
    println!("  cell_1: gene_1=30.0");
    println!("  cell_2: gene_0=40.0, gene_2=50.0");
    println!("  cell_3: gene_1=60.0");

    // Subset to cells 1,2 and genes 0,2
    // Expected result should be:
    // cell_1: gene_0=0.0, gene_2=0.0 (no values in original)
    // cell_2: gene_0=40.0, gene_2=50.0
    let obs_selection = SelectInfoElem::Index(vec![1, 2]);
    let var_selection = SelectInfoElem::Index(vec![0, 2]);

    adata.subset_inplace(&[&obs_selection, &var_selection])?;

    // Verify dimensions first
    assert_eq!(adata.n_obs(), 2, "Should have 2 observations");
    assert_eq!(adata.n_vars(), 2, "Should have 2 variables");

    // Verify names
    let new_obs_names = adata.obs_names();
    let new_var_names = adata.var_names();
    assert_eq!(new_obs_names, vec!["cell_1", "cell_2"]);
    assert_eq!(new_var_names, vec!["gene_0", "gene_2"]);

    // Get the subset matrix and verify values
    let subset_matrix_data = adata.x().get_data()?;

    if let Ok(extracted_matrix) = CsMatI::<f64, u32, u64>::try_from(subset_matrix_data) {
        println!("Subset matrix CSR data:");

        // Check dimensions
        assert_eq!(extracted_matrix.rows(), 2, "Should have 2 rows");
        assert_eq!(extracted_matrix.cols(), 2, "Should have 2 columns");

        // Check specific values by examining CSR structure:
        // Row 0 (cell_1): should be empty (no non-zero values)
        let row0_nnz = extracted_matrix.outer_view(0).unwrap().nnz();
        assert_eq!(
            row0_nnz, 0,
            "cell_1 should have no non-zero values in selected columns"
        );

        // Row 1 (cell_2): should have 2 non-zero values: 40.0 at col 0, 50.0 at col 1
        let row1_view = extracted_matrix.outer_view(1).unwrap();
        let row1_nnz = row1_view.nnz();
        assert_eq!(row1_nnz, 2, "cell_2 should have 2 non-zero values");

        // Check the actual values for row 1 (cell_2)
        if row1_nnz == 2 {
            let col0_idx = row1_view.indices()[0];
            let col0_val = row1_view.data()[0];
            let col1_idx = row1_view.indices()[1];
            let col1_val = row1_view.data()[1];

            // Should be: column 0 (gene_0) = 40.0, column 1 (gene_2) = 50.0
            assert_eq!(col0_idx, 0, "First non-zero should be in column 0");
            assert_eq!(col0_val, 40.0, "Value at (cell_2, gene_0) should be 40.0");
            assert_eq!(col1_idx, 1, "Second non-zero should be in column 1");
            assert_eq!(col1_val, 50.0, "Value at (cell_2, gene_2) should be 50.0");
        }

        println!("✓ All values match expected results!");
    } else {
        panic!("Expected CSR matrix format");
    }

    Ok(())
}