# AnnData-Memory

![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE.md)

A high-performance, thread-safe, in-memory implementation of the AnnData data structure for the SingleRust ecosystem.

## Overview

AnnData-Memory provides a thread-safe, high-performance implementation of the AnnData data structure for single-cell genomics data analysis in Rust. It serves as a companion to the [anndata-rs](https://github.com/kaizhang/anndata-rs) crate, focusing on efficient in-memory operations with controlled mutability and concurrent access patterns.

This library is designed to:
- Accelerate AnnData operations through optimized in-memory structures
- Enable safe multi-threaded access to AnnData objects
- Provide flexible and efficient data manipulation capabilities
- Seamlessly integrate with the broader SingleRust ecosystem

## Key Features

- **Thread-Safe Data Access**: Built on `parking_lot` locks for efficient concurrent operations
- **Controlled Mutability**: Fine-grained locking mechanisms allow for safe concurrent reads and writes
- **Memory Efficiency**: Optimized data structures to reduce memory overhead
- **Format Conversion**: Seamless conversion between CSR and CSC sparse matrix formats
- **Efficient Subsetting**: Fast subsetting operations (both in-place and copy-based)
- **H5 Interoperability**: Convert between H5-backed AnnData and in-memory structures
- **Comprehensive Data Model**: Full support for AnnData components (X, obs, var, layers, obsm, obsp, varm, varp, uns)

## Installation

Add AnnData-Memory to your `Cargo.toml`:

```toml
[dependencies]
anndata-memory = "1.0.0"
```

## Usage

### Creating an AnnData Object

```rust
use anndata::{ArrayData, data::DynCsrMatrix};
use anndata_memory::{IMAnnData, IMArrayElement};
use nalgebra_sparse::{CooMatrix, CsrMatrix};

// Create a sparse matrix
let mut coo_matrix = CooMatrix::new(nrows, ncols);
coo_matrix.push(0, 0, 1.0);
coo_matrix.push(1, 2, 2.0);
// ... add more entries

let csr_matrix = CsrMatrix::from(&coo_matrix);
let matrix = DynCsrMatrix::from(csr_matrix);
let array_data = ArrayData::CsrMatrix(matrix);

// Create the AnnData object
let adata = IMAnnData::new_basic(
    array_data,
    vec!["cell1".to_string(), "cell2".to_string(), "cell3".to_string()],
    vec!["gene1".to_string(), "gene2".to_string(), "gene3".to_string()]
).unwrap();
```

### Converting from H5-backed AnnData

```rust
use anndata::{AnnData};
use anndata_hdf5::H5;
use anndata_memory::convert_to_in_memory;

// Open an H5-backed AnnData file
let h5_file = H5::open("data.h5ad").unwrap();
let anndata = AnnData::<H5>::open(h5_file).unwrap();

// Convert to in-memory representation
let imanndata = convert_to_in_memory(anndata).unwrap();
```

### Working with Layers

```rust
use anndata_memory::{IMAnnData, IMArrayElement};

// Add a layer to the AnnData object
let layer_name = "normalized".to_string();
adata.add_layer(layer_name.clone(), normalized_data).unwrap();

// Retrieve a layer
let layer = adata.get_layer("normalized").unwrap();
```

### Subsetting Data

```rust
use anndata::data::SelectInfoElem;

// Create selection criteria
let obs_selection = SelectInfoElem::Index(vec![0, 2]); // Select observations 0 and 2
let var_selection = SelectInfoElem::Index(vec![1, 2]); // Select variables 1 and 2

// Create a subset of the data (creating a new object)
let subset = adata.subset(&[&obs_selection, &var_selection]).unwrap();

// Or subset in-place
adata.subset_inplace(&[&obs_selection, &var_selection]).unwrap();
```

### Matrix Format Conversion

```rust
use anndata_memory::IMArrayElement;

// Get the X matrix and convert between CSR and CSC formats
let x = adata.x();
x.convert_matrix_format().unwrap(); // Converts CSR to CSC or vice versa
```

## Thread Safety

AnnData-Memory is designed for safe concurrent access. The `IMAnnData` structure itself isn't wrapped in a lock, but each of its fields (x, obs, var, layers, etc.) is individually wrapped in a thread-safe `RwSlot` that allows multiple readers or a single writer at any time. This provides fine-grained control over concurrency.

```rust
use std::thread;
use std::sync::{Arc, RwLock};
use anndata_memory::IMAnnData;

// For thread-safe access to the whole object, wrap it in Arc<RwLock<>>
let adata = Arc::new(RwLock::new(adata));

// Example 1: Multiple threads accessing individual fields (safer)
let handles: Vec<_> = (0..10).map(|i| {
    let adata_clone = Arc::clone(&adata);
    thread::spawn(move || {
        // Lock the whole object only briefly to get references to fields
        let data = adata_clone.read().unwrap();
        
        // Now work with the thread-safe fields
        let x = data.x(); // Each field is already in a RwSlot
        let shape = x.get_shape().unwrap();
        
        // Process field-specific data...
        println!("Thread {} working with matrix of shape {:?}", i, shape);
    })
}).collect();

// Example 2: When you need to modify the IMAnnData structure itself
let handle = {
    let adata_clone = Arc::clone(&adata);
    thread::spawn(move || {
        // Get write lock on the entire object
        let mut data = adata_clone.write().unwrap();
        
        // Now you can safely modify any aspect of the IMAnnData
        data.subset_inplace(&[&obs_selection, &var_selection]).unwrap();
    })
};

// Wait for all threads to complete
for handle in handles {
    handle.join().unwrap();
}
```

Note: When performing mutations from multiple threads, you need to take extra care to avoid lock races since `IMAnnData` itself isn't thread-safe (only its individual fields are). For multi-threaded write operations, consider wrapping your `IMAnnData` instance in a `RwLock` or `Mutex`, or use the `deep_clone()` method to create independent copies when necessary.

## Performance Considerations

- Use `get_layer_shallow()` for read-only access to layers to avoid unnecessary cloning
- Consider converting between CSR and CSC formats based on your access patterns (row-wise vs. column-wise)
- For multi-threaded applications, balance the granularity of operations to minimize lock contention

## Architecture

AnnData-Memory uses a component-based architecture:

- `IMAnnData`: The main container structure, containing individually thread-safe fields
- `IMArrayElement`: Thread-safe wrapper for array data (using `RwSlot`)
- `IMDataFrameElement`: Thread-safe wrapper for DataFrames with index (using `RwSlot`)
- `IMAxisArrays`: Thread-safe collection of arrays associated with an axis (using `RwSlot`)
- `IMElementCollection`: Thread-safe collection of unstructured annotations (using `RwSlot`)
- `RwSlot`: Basic building block providing controlled access to data with read-write locking

## Limitations

- View support is limited (subsetting creates copies, not views)
- The `IMAnnData` structure itself isn't thread-safe, only its individual fields are
- Some operations may involve lock races when writing to multiple fields from different threads
- Care must be taken with concurrent operations to prevent deadlocks (as noted in the source code comments)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- [anndata-rs](https://github.com/kaizhang/anndata-rs) team for the core AnnData implementation in Rust
- The SingleRust ecosystem contributors