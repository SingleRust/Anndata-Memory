use std::path::Path;

use crate::{ConcatStrategy, LoadingStrategy};

pub(crate) mod csr;

// TODO!


pub fn concat_datasets_backed_to_memory<P: AsRef<Path>>(paths: &[P], load_strategy: Option<LoadingStrategy>, merge_strategy: Option<ConcatStrategy>) {

    // general approach have dataset:
    // 1. collect statistics: ensure all datasets have same datatype (for sparse datatypes collect num rows, num cols, nnz)
    // 2. delegate to CSR, CSC datatype
    // 3. load sequentially one dataset after another, if chunked approach, use anndata-rs to handle chunking 0





}


