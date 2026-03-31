use std::path::Path;

use crate::{ConcatStrategy, LoadingStrategy};

pub(crate) mod csr;

// TODO!


#[allow(dead_code)]
pub fn concat_datasets_backed_to_memory<P: AsRef<Path>>(_paths: &[P], _load_strategy: Option<LoadingStrategy>, _merge_strategy: Option<ConcatStrategy>) {

    // general approach have dataset:
    // 1. collect statistics: ensure all datasets have same datatype (for sparse datatypes collect num rows, num cols, nnz)
    // 2. delegate to CSR, CSC datatype
    // 3. load sequentially one dataset after another, if chunked approach, use anndata-rs to handle chunking 0





}


