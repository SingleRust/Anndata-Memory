# Changelog


## Version: 0.1.0
Implemented basic functionality for the in-memory version of the anndata package
- Conversion from `anndata-rs` to `IMAnnData` (supports X, var, obs, obsm, obsp, varm, varp, uns, layers)
- Subsetting of the in-memory data object (inplace and copy, no view)
Limitations/Caution:
- Writing to multiple fields from different threads might involve lock races, in-case you write from different threads you have to take care of that, although we plan on implementing a safe version soon