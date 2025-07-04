use std::path::Path;

use anndata::{backend::DataContainer, data::DataFrameIndex, ArrayData, Backend, Data, Readable};
use anndata_hdf5::{H5File, H5};
use polars::frame::DataFrame;
use anndata::backend::AttributeOp;
use anndata::backend::GroupOp;

use crate::IMArrayElement;
use crate::IMElement;
use crate::LoadingStrategy;
use crate::{chunked_loader::load_csr_chunked, optimized_loader::load_csr_optimized, utils::{read_dataframe_index, should_use_chunked_loading}, IMAnnData, LoadingConfig};

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

fn load_obs(
    h5_file: &H5File, 
    config: &LoadingConfig
) -> anyhow::Result<(DataFrame, Vec<String>)> {
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

fn load_var(
    h5_file: &H5File, 
    config: &LoadingConfig
) -> anyhow::Result<(DataFrame, Vec<String>)> {
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
    config: &LoadingConfig
) -> anyhow::Result<ArrayData> {
    if !h5_file.link_exists("X") {
        if config.show_progress {
            println!("No X matrix found, using empty matrix");
        }
        return Ok(ArrayData::Array(
            anndata::data::DynArray::from(ndarray::Array2::<f64>::zeros((0, 0)))
        ));
    }
    
    if config.show_progress {
        println!("Loading X matrix...");
    }
    
    let x_container = DataContainer::open(h5_file, "X")?;
    
    let matrix_type = x_container.encoding_type()?;
    if config.show_progress {
        match &matrix_type {
            anndata::backend::DataType::CsrMatrix(_) => {
                let group = x_container.as_group()?;
                let shape: Vec<u64> = group.get_attr("shape")?;
                let nnz = group.open_dataset("data")?.shape()[0];
                println!("  CSR matrix: {}×{} with {} non-zeros", shape[0], shape[1], nnz);
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
    
    let result = match matrix_type {
        anndata::backend::DataType::CsrMatrix(_) => {
            if should_use_chunked_loading(&x_container, config)? {
                if config.show_progress {
                    println!("  Using chunked loading for large matrix");
                }
                load_csr_chunked(&x_container, config)?
            } else {
                if config.show_progress {
                    println!("  Using optimized CSR loading");
                }
                load_csr_optimized(&x_container)?
            }
        }
        _ => {
            if config.show_progress {
                println!("  Using standard loading");
            }
            ArrayData::read(&x_container)?
        }
    };
    
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
        
        let array_data = if group_name == "layers" {
            match array_container.encoding_type()? {
                anndata::backend::DataType::CsrMatrix(_) => {
                    if should_use_chunked_loading(&array_container, config)? {
                        if config.show_progress {
                            println!("    Using chunked loading for layer {}", array_name);
                        }
                        load_csr_chunked(&array_container, config)?
                    } else {
                        load_csr_optimized(&array_container)?
                    }
                }
                _ => ArrayData::read(&array_container)?,
            }
        } else {
            ArrayData::read(&array_container)?
        };
        
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