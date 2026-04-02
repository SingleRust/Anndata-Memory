# AnnData-Memory

![Version](https://img.shields.io/badge/version-1.0.7-blue)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE.md)

A high-performance, thread-safe, in-memory implementation of the AnnData data structure for the [SingleRust](https://github.com/SingleRust) ecosystem.

## Overview

**AnnData-Memory** provides a thread-safe, high-performance implementation of the AnnData data structure for single-cell genomics data analysis in Rust. It serves as an in-memory companion to the [anndata-rs](https://github.com/kaizhang/anndata-rs) crate, focusing on efficient random access, controlled mutability, and safe concurrent operations.

This library is designed for:
- **High-Performance Analysis**: Accelerate workflows by keeping data in optimized RAM structures.
- **Thread Safety**: Safe multi-threaded access using fine-grained locking.
- **Interoperability**: Seamlessly switch between backed (H5AD/Zarr) and in-memory representations.
- **Lean Data Transfer**: Efficiently ingest data from disk using `take()` and `drain()` semantics when possible.

## Key Features

- **Component-Level Locking**: Built on `parking_lot` for efficient concurrent operations; each AnnData component (X, obs, obsm, etc.) is individually locked.
- **Sparse Matrix Support**: Native integration with [sprs](https://github.com/vbarrielle/sprs) for high-performance sparse matrix operations.
- **H5 & Zarr Interoperability**: Direct loading from `.h5ad` files and Zarr V3 stores.
- **Flexible Subsetting**: Fast in-place and copy-based subsetting operations.
- **Comprehensive Data Model**: Full support for all AnnData components (X, obs, var, layers, obsm, obsp, varm, varp, uns).

## Installation

Add AnnData-Memory to your `Cargo.toml`:

```toml
[dependencies]
anndata-memory = "1.0.7"
```

## Usage

### Creating an AnnData Object

AnnData-Memory uses `sprs` for sparse data. You can easily create an `IMAnnData` object from raw components:

```rust
use anndata::ArrayData;
use anndata_memory::IMAnnData;
use sprs::{CsMatI, TriMatI};

// Create a sparse matrix (CSR format)
let (nrows, ncols) = (3, 3);
let mut coo = TriMatI::<f64, u32>::new((nrows, ncols));
coo.add_triplet(0, 0, 1.0);
coo.add_triplet(1, 2, 2.0);

let csr: CsMatI<f64, u32, u64> = coo.to_csr();
let array_data: ArrayData = csr.into();

// Initialize the AnnData object
let adata = IMAnnData::new_basic(
    array_data,
    vec!["cell1".into(), "cell2".into(), "cell3".into()],
    vec!["gene1".into(), "gene2".into(), "gene3".into()]
).unwrap();
```

### Loading Data (H5AD & Zarr)

You can load data directly into memory from various backends:

```rust
use anndata_memory::{load_h5ad, load_zarr};

// From HDF5 (.h5ad)
let adata_h5 = load_h5ad("data.h5ad").unwrap();

// From Zarr V3
let adata_zarr = load_zarr("data.zarr").unwrap();
```

### In-Memory Conversion

For fine-grained control, you can open a backed AnnData object and convert it. The conversion is optimized to transfer ownership of data buffers where possible.

```rust
use anndata::AnnData;
use anndata_hdf5::H5;
use anndata_memory::convert_to_in_memory;

// Open backed file
let file = H5::open("data.h5ad").unwrap();
let backed = AnnData::<H5>::open(file).unwrap();

// Convert to in-memory representation
let adata = convert_to_in_memory(backed).unwrap();
```

### Concurrent Access

Individual components of `IMAnnData` are protected by `RwLock`. Multiple threads can read different components simultaneously.

```rust
use std::sync::Arc;
use std::thread;

let adata = Arc::new(adata);

let handle = thread::spawn({
    let adata = Arc::clone(&adata);
    move || {
        let x = adata.x().get_data().unwrap();
        // Perform computation on X...
    }
});
```

## Performance Considerations

- **Lean Extraction**: When converting from backed objects, `anndata-memory` attempts to use "take" semantics to move data out of the source object's cache, minimizing memory duplication.
- **Sparse vs Dense**: `X` can be either sparse (CSR/CSC) or dense (ndarray). `sprs` is used for all sparse operations.
- **Subsetting**: In-place subsetting (`subset_inplace`) is generally faster and more memory-efficient than creating a new subset.

## Architecture

AnnData-Memory uses a **Component-Level Locking** strategy:
- `IMAnnData`: The primary container.
- `RwSlot<T>`: A wrapper around `Arc<RwLock<Option<T>>>` used for individual fields.
- `IMArrayElement`, `IMDataFrameElement`, etc.: Thread-safe wrappers for specific AnnData components.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- The [anndata-rs](https://github.com/kaizhang/anndata-rs) team for the foundational AnnData traits and backends.
- The [sprs](https://github.com/vbarrielle/sprs) maintainers for high-performance sparse matrix primitives.
