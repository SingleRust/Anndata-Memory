use anndata::{
    AnnData, Backend,
};
use anndata_hdf5::H5;
use anndata_memory::*;

//#[test]
#[allow(dead_code)]
fn test_convert_anndata_to_imanndata() -> anyhow::Result<()> {
    let h5_file = H5::open("/local/bachelor_thesis_ian/single_bench/data/merged_test.h5ad")?;
    let anndata = AnnData::<H5>::open(h5_file)?;

    let imanndata = convert_to_in_memory(anndata)?;
    println!("{}", imanndata);

    Ok(())
}


//#[test]
#[allow(dead_code)]
fn test_iman_length_converted() -> anyhow::Result<()> {
    let h5_file = H5::open("/local/bachelor_thesis_ian/single_bench/data/150k_cellxgene_mmu_changed.h5ad")?;
    let anndata = AnnData::<H5>::open(h5_file)?;

    let imanndata = convert_to_in_memory(anndata)?;
    
    println!("{}", imanndata);
    Ok(())
}
