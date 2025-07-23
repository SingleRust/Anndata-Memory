mod ad;
mod base;
mod converter;
pub(crate) mod utils;
pub(crate) mod chunked_loader;
pub(crate) mod optimized_loader;
mod loader;
mod concat;

pub use ad::IMAnnData;
pub use ad::helpers::IMArrayElement;
pub use ad::helpers::IMDataFrameElement;
pub use ad::helpers::IMElementCollection;
pub use ad::helpers::IMElement;
pub use ad::helpers::IMAxisArrays;
pub use converter::convert_to_in_memory;
pub use converter::convert_to_backed;
pub use converter::convert_to_new_backed_h5;
pub use base::DeepClone;



#[derive(Clone, Debug)]
pub enum LoadingStrategy {
    Auto,           
    ForceComplete,  
    ForceChunked, 
}

#[derive(Clone, Debug)]
pub struct LoadingConfig {
    pub loading_strategy: LoadingStrategy,  
    pub chunk_size_mb: usize,
    pub memory_threshold_mb: usize,
    pub show_progress: bool,
}

impl Default for LoadingConfig {
    fn default() -> Self {
        Self {
            loading_strategy: LoadingStrategy::Auto, 
            chunk_size_mb: 100,
            memory_threshold_mb: 1024,
            show_progress: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ConcatStrategy {
    ConcatObs,
    ConcatVars,
    Union,
    Intersection
}

pub use loader::{load_h5ad, load_h5ad_fast, load_h5ad_conservative, load_h5ad_with_config};