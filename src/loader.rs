use std::path::Path;

use anndata::backend::AttributeOp;
use anndata::backend::GroupOp;
use anndata::{backend::DataContainer, data::DataFrameIndex, ArrayData, Backend, Data, Readable};
use anndata_hdf5::{H5File, H5};
use polars::frame::DataFrame;

use crate::IMArrayElement;
use crate::IMElement;
use crate::LoadingStrategy;
use crate::{
    utils::{read_dataframe_index},
    IMAnnData, LoadingConfig,
};

pub fn load_h5ad(h5_path: impl AsRef<Path>) -> anyhow::Result<IMAnnData> {
    load_h5ad_with_config(h5_path, LoadingConfig::default())
}

pub fn load_h5ad_fast(h5_path: impl AsRef<Path>) -> anyhow::Result<IMAnnData> {
    let config = LoadingConfig {
        loading_strategy: LoadingStrategy::ForceComplete,
        chunk_size_mb: 256,
        memory_threshold_mb: 4096,
        show_progress: true,
    };

    load_h5ad_with_config(h5_path, config)
}

pub fn load_h5ad_conservative(h5_path: impl AsRef<Path>) -> anyhow::Result<IMAnnData> {
    let config = LoadingConfig {
        loading_strategy: LoadingStrategy::ForceChunked,
        chunk_size_mb: 64,
        memory_threshold_mb: 256,
        show_progress: true,
    };

    load_h5ad_with_config(h5_path, config)
}

pub fn load_h5ad_with_config(
    h5_path: impl AsRef<Path>,
    config: LoadingConfig,
) -> anyhow::Result<IMAnnData> {
    let h5_file = H5::open(h5_path)?;

    if config.show_progress {
        println!("Loading H5AD file directly...");
    }

    // Load core components
    let (obs_df, obs_names) = load_obs(&h5_file, &config)?;
    let (var_df, var_names) = load_var(&h5_file, &config)?;
    let x_data = load_x_matrix(&h5_file, &config)?;

    // Create the main structure
    let imad = IMAnnData::new_extended(x_data, obs_names, var_names, obs_df, var_df)?;

    // Load optional components in parallel if beneficial
    load_axis_arrays(&h5_file, "obsm", imad.obsm(), &config)?;
    load_axis_arrays(&h5_file, "obsp", imad.obsp(), &config)?;
    load_axis_arrays(&h5_file, "varm", imad.varm(), &config)?;
    load_axis_arrays(&h5_file, "varp", imad.varp(), &config)?;
    load_axis_arrays(&h5_file, "layers", imad.layers(), &config)?;
    load_uns(&h5_file, imad.uns(), &config)?;

    if config.show_progress {
        println!("H5AD file loaded successfully");
    }

    Ok(imad)
}

fn load_obs(h5_file: &H5File, config: &LoadingConfig) -> anyhow::Result<(DataFrame, Vec<String>)> {
    if config.show_progress {
        println!("Loading observations...");
    }

    if !h5_file.exists("obs")? {
        return Ok((DataFrame::empty(), vec![]));
    }

    let obs_container = DataContainer::open(h5_file, "obs")?;
    let obs_df: DataFrame = ArrayData::read(&obs_container)?.try_into()?;

    let obs_index = read_dataframe_index(&obs_container)?;
    let obs_names = obs_index.into_vec();

    if config.show_progress {
        println!("  {} observations loaded", obs_names.len());
    }

    Ok((obs_df, obs_names))
}

fn load_var(h5_file: &H5File, config: &LoadingConfig) -> anyhow::Result<(DataFrame, Vec<String>)> {
    if config.show_progress {
        println!("Loading variables...");
    }

    if !h5_file.exists("var")? {
        return Ok((DataFrame::empty(), vec![]));
    }

    let var_container = DataContainer::open(h5_file, "var")?;
    let var_df: DataFrame = ArrayData::read(&var_container)?.try_into()?;

    let var_index = read_dataframe_index(&var_container)?;
    let var_names = var_index.into_vec();

    if config.show_progress {
        println!("  {} variables loaded", var_names.len());
    }

    Ok((var_df, var_names))
}

fn load_x_matrix(
    h5_file: &anndata_hdf5::H5File,
    config: &LoadingConfig,
) -> anyhow::Result<ArrayData> {
    if !h5_file.link_exists("X") {
        if config.show_progress {
            println!("No X matrix found, using empty matrix");
        }
        return Ok(ArrayData::Array(anndata::data::DynArray::from(
            ndarray::Array2::<f64>::zeros((0, 0)),
        )));
    }

    if config.show_progress {
        println!("Loading X matrix...");
    }

    let x_container = DataContainer::open(h5_file, "X")?;

    let matrix_type = x_container.encoding_type()?;
    if config.show_progress {
        match &matrix_type {
            anndata::backend::DataType::CsrMatrix(..) | anndata::backend::DataType::CscMatrix(..) => {
                let group = x_container.as_group()?;
                let shape: Vec<u64> = group.get_attr("shape")?;
                let nnz = group.open_dataset("data")?.shape()[0];
                println!(
                    "  Sparse matrix: {}×{} with {} non-zeros",
                    shape[0], shape[1], nnz
                );
            }
            anndata::backend::DataType::Array(_) => {
                let shape = x_container.as_dataset()?.shape();
                println!("  Dense matrix: {:?}", shape);
            }
            _ => {
                println!("  Matrix type: {:?}", matrix_type);
            }
        }
    }

    if config.show_progress {
        println!("  Using standard loading");
    }
    
    let result = ArrayData::read(&x_container)?;

    if config.show_progress {
        println!("  X matrix loaded successfully");
    }

    Ok(result)
}

fn load_axis_arrays(
    h5_file: &anndata_hdf5::H5File,
    group_name: &str,
    target_arrays: crate::IMAxisArrays,
    config: &LoadingConfig,
) -> anyhow::Result<()> {
    if !h5_file.link_exists(group_name) {
        return Ok(());
    }

    let group = h5_file.open_group(group_name)?;
    let array_names = group.list()?;

    if array_names.is_empty() {
        return Ok(());
    }

    if config.show_progress {
        println!("Loading {} ({} items)...", group_name, array_names.len());
    }
    let arr_n_len = array_names.len();
    for array_name in array_names {
        if config.show_progress && arr_n_len > 3 {
            println!("  Loading {}/{}", group_name, array_name);
        }

        let array_container = DataContainer::open(&group, &array_name)?;

        let array_data = ArrayData::read(&array_container)?;

        let im_array = IMArrayElement::new(array_data);
        target_arrays.add_array(array_name, im_array)?;
    }

    if config.show_progress {
        println!("  {} loaded successfully", group_name);
    }

    Ok(())
}

fn load_uns(
    h5_file: &anndata_hdf5::H5File,
    target_uns: crate::IMElementCollection,
    config: &LoadingConfig,
) -> anyhow::Result<()> {
    if !h5_file.link_exists("uns") {
        return Ok(());
    }

    if config.show_progress {
        println!("Loading unstructured annotations...");
    }

    let uns_group = h5_file.open_group("uns")?;
    let item_names = uns_group.list()?;

    for item_name in item_names.iter() {
        let item_container = DataContainer::open(&uns_group, &item_name)?;
        let data = Data::read(&item_container)?;
        let im_element = IMElement::new(data);
        target_uns.add_data(item_name.clone(), im_element)?;
    }

    if config.show_progress && !item_names.is_empty() {
        println!("  {} items loaded in uns", item_names.len());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{convert_to_in_memory, load_h5ad};
    use anndata::{data::Element, AnnData, HasShape};
    use anndata_hdf5::H5;
    use std::time::Instant;

    // Update this path to point to your H5AD file
    const TEST_H5AD_PATH: &str = "/local/06-24_single-rust-project/performance_eval/datasets/real_world/tahoe/tahoe_100K_cells.h5ad";

    #[test]
    fn test_conversion_vs_direct_loading() -> anyhow::Result<()> {
        // Check if test file exists
        if !std::path::Path::new(TEST_H5AD_PATH).exists() {
            println!("Skipping test: H5AD file not found at {}", TEST_H5AD_PATH);
            println!("Please update TEST_H5AD_PATH to point to your H5AD file");
            return Ok(());
        }

        println!("Testing with file: {}", TEST_H5AD_PATH);

        // Method 1: Load via conversion (H5AD → AnnData → IMAnnData)
        println!("Loading via conversion...");
        let start = Instant::now();
        let h5_file = H5::open(TEST_H5AD_PATH)?;
        let anndata = AnnData::<H5>::open(h5_file)?;
        let converted_imad = convert_to_in_memory(anndata)?;
        let conversion_time = start.elapsed();

        // Method 2: Load directly
        println!("Loading directly...");
        let start = Instant::now();
        let direct_imad = load_h5ad(TEST_H5AD_PATH)?;
        let direct_time = start.elapsed();

        // Compare basic properties
        println!("\n=== Results ===");
        println!("Conversion time: {:?}", conversion_time);
        println!("Direct time: {:?}", direct_time);

        let speedup = conversion_time.as_secs_f64() / direct_time.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);

        // Basic data integrity checks
        assert_eq!(
            converted_imad.n_obs(),
            direct_imad.n_obs(),
            "Number of observations should match"
        );
        assert_eq!(
            converted_imad.n_vars(),
            direct_imad.n_vars(),
            "Number of variables should match"
        );
        assert_eq!(
            converted_imad.obs_names(),
            direct_imad.obs_names(),
            "Observation names should match"
        );
        assert_eq!(
            converted_imad.var_names(),
            direct_imad.var_names(),
            "Variable names should match"
        );

        println!(
            "Dimensions: {} obs × {} vars",
            direct_imad.n_obs(),
            direct_imad.n_vars()
        );
        println!("✓ Basic data integrity checks passed");

        Ok(())
    }

    #[test]
    fn test_loading_presets_comparison() -> anyhow::Result<()> {
        if !std::path::Path::new(TEST_H5AD_PATH).exists() {
            println!("Skipping test: H5AD file not found at {}", TEST_H5AD_PATH);
            return Ok(());
        }

        println!("Testing different loading presets...");

        // Test all three preset functions
        let start = Instant::now();
        let default_imad = load_h5ad(TEST_H5AD_PATH)?;
        let default_time = start.elapsed();

        let start = Instant::now();
        let fast_imad = load_h5ad_fast(TEST_H5AD_PATH)?;
        let fast_time = start.elapsed();

        let start = Instant::now();
        let conservative_imad = load_h5ad_conservative(TEST_H5AD_PATH)?;
        let conservative_time = start.elapsed();

        println!("\n=== Preset Performance ===");
        println!("Default: {:?}", default_time);
        println!("Fast: {:?}", fast_time);
        println!("Conservative: {:?}", conservative_time);

        // All should produce the same basic results
        assert_eq!(default_imad.n_obs(), fast_imad.n_obs());
        assert_eq!(default_imad.n_obs(), conservative_imad.n_obs());
        assert_eq!(default_imad.n_vars(), fast_imad.n_vars());
        assert_eq!(default_imad.n_vars(), conservative_imad.n_vars());

        println!("✓ All presets produce consistent dimensions");

        Ok(())
    }

    #[test]
    fn test_matrix_data_consistency() -> anyhow::Result<()> {
        if !std::path::Path::new(TEST_H5AD_PATH).exists() {
            println!("Skipping test: H5AD file not found at {}", TEST_H5AD_PATH);
            return Ok(());
        }

        println!("Testing matrix data consistency...");

        // Load with both methods
        let h5_file = H5::open(TEST_H5AD_PATH)?;
        let anndata = AnnData::<H5>::open(h5_file)?;
        let converted_imad = convert_to_in_memory(anndata)?;
        let direct_imad = load_h5ad(TEST_H5AD_PATH)?;

        // Check X matrix properties
        let converted_x = converted_imad.x().get_data()?;
        let direct_x = direct_imad.x().get_data()?;

        // Both should be the same type
        assert_eq!(
            std::mem::discriminant(&converted_x),
            std::mem::discriminant(&direct_x),
            "X matrix types should match"
        );

        // Check shapes match
        let converted_shape = converted_x.shape();
        let direct_shape = direct_x.shape();
        assert_eq!(
            converted_shape, direct_shape,
            "X matrix shapes should match"
        );

        println!("X matrix shape: {:?}", direct_shape);
        println!("X matrix type: {:?}", direct_x.data_type());

        // Check if we have layers and compare a few
        let converted_layers = converted_imad.layers().keys();
        let direct_layers = direct_imad.layers().keys();
        assert_eq!(converted_layers, direct_layers, "Layer keys should match");

        if !direct_layers.is_empty() {
            println!("Layers found: {:?}", direct_layers);

            // Test first layer if available
            if let Some(first_layer) = direct_layers.first() {
                let conv_layer = converted_imad.get_layer(first_layer)?;
                let direct_layer = direct_imad.get_layer(first_layer)?;

                let conv_layer_shape = conv_layer.get_shape()?;
                let direct_layer_shape = direct_layer.get_shape()?;
                assert_eq!(
                    conv_layer_shape, direct_layer_shape,
                    "Layer shapes should match"
                );

                println!("Layer '{}' shape: {:?}", first_layer, direct_layer_shape);
            }
        }

        println!("✓ Matrix data consistency checks passed");

        Ok(())
    }

    #[test]
    fn test_subsetting_consistency() -> anyhow::Result<()> {
        if !std::path::Path::new(TEST_H5AD_PATH).exists() {
            println!("Skipping test: H5AD file not found at {}", TEST_H5AD_PATH);
            return Ok(());
        }

        println!("Testing subsetting consistency...");

        // Load data
        let direct_imad = load_h5ad(TEST_H5AD_PATH)?;

        let n_obs = direct_imad.n_obs();
        let n_vars = direct_imad.n_vars();

        // Only test subsetting if we have enough data
        if n_obs < 10 || n_vars < 10 {
            println!("Dataset too small for subsetting test, skipping");
            return Ok(());
        }

        // Create a subset (first 10 obs, first 5 vars)
        let obs_selection = anndata::data::SelectInfoElem::from(0..10.min(n_obs));
        let var_selection = anndata::data::SelectInfoElem::from(0..5.min(n_vars));

        // Test subsetting
        let subset = direct_imad.subset(&[&obs_selection, &var_selection])?;

        assert_eq!(subset.n_obs(), 10.min(n_obs));
        assert_eq!(subset.n_vars(), 5.min(n_vars));

        // Check that subset names match original
        let original_obs_names = direct_imad.obs_names();
        let subset_obs_names = subset.obs_names();

        for (i, subset_name) in subset_obs_names.iter().enumerate() {
            assert_eq!(
                subset_name, &original_obs_names[i],
                "Subset obs names should match"
            );
        }

        println!("Original: {} obs × {} vars", n_obs, n_vars);
        println!("Subset: {} obs × {} vars", subset.n_obs(), subset.n_vars());
        println!("✓ Subsetting consistency checks passed");

        Ok(())
    }
}