use std::path::Path;

use anndata::{AnnData, Backend};
use anndata_hdf5::H5;
use anndata_zarr::Zarr;

use crate::converter::convert_to_in_memory_with_options;
use crate::{IMAnnData, LoadingConfig};

pub fn load_h5ad(h5_path: impl AsRef<Path>) -> anyhow::Result<IMAnnData> {
    load_h5ad_with_config(h5_path, LoadingConfig::default())
}

pub fn load_h5ad_fast(h5_path: impl AsRef<Path>) -> anyhow::Result<IMAnnData> {
    let config = LoadingConfig {
        loading_strategy: crate::LoadingStrategy::ForceComplete,
        chunk_size_mb: 256,
        memory_threshold_mb: 4096,
        show_progress: true,
    };

    load_h5ad_with_config(h5_path, config)
}

pub fn load_h5ad_conservative(h5_path: impl AsRef<Path>) -> anyhow::Result<IMAnnData> {
    let config = LoadingConfig {
        loading_strategy: crate::LoadingStrategy::ForceChunked,
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
    let file = H5::open(h5_path)?;
    let adata = AnnData::<H5>::open(file)?;
    convert_to_in_memory_with_options(adata, config.show_progress)
}

pub fn load_zarr(zarr_path: impl AsRef<Path>) -> anyhow::Result<IMAnnData> {
    load_zarr_with_config(zarr_path, LoadingConfig::default())
}

pub fn load_zarr_with_config(
    zarr_path: impl AsRef<Path>,
    config: LoadingConfig,
) -> anyhow::Result<IMAnnData> {
    let store = Zarr::open(zarr_path)?;
    let adata = AnnData::<Zarr>::open(store)?;
    convert_to_in_memory_with_options(adata, config.show_progress)
}
