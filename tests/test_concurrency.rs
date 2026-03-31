use anndata::{data::SelectInfoElem, ArrayData};
use anndata_memory::{IMAnnData, IMArrayElement};
use std::sync::Arc;
use std::thread;
use sprs::{CsMatI, TriMatI};

#[test]
fn test_component_level_concurrency() -> anyhow::Result<()> {
    let nrows = 100;
    let ncols = 100;
    
    let mut coo = TriMatI::<f64, u32>::new((nrows, ncols));
    for i in 0..nrows {
        coo.add_triplet(i, i, 1.0);
    }
    let csr: CsMatI<f64, u32, u64> = coo.to_csr();
    let matrix = ArrayData::from(csr);
    
    let obs_names: Vec<String> = (0..nrows).map(|i| format!("cell_{}", i)).collect();
    let var_names: Vec<String> = (0..ncols).map(|i| format!("gene_{}", i)).collect();
    
    let adata = Arc::new(IMAnnData::new_basic(matrix, obs_names, var_names)?);
    
    let mut handles = vec![];
    
    // Thread 1: Continuously reading from X
    let adata_t1 = Arc::clone(&adata);
    handles.push(thread::spawn(move || {
        for _ in 0..100 {
            let _x = adata_t1.x().get_data().unwrap();
            thread::yield_now();
        }
    }));
    
    // Thread 2: Continuously adding and removing layers
    let adata_t2 = Arc::clone(&adata);
    handles.push(thread::spawn(move || {
        for i in 0..50 {
            let mut coo_layer = TriMatI::<f64, u32>::new((nrows, ncols));
            coo_layer.add_triplet(0, 0, i as f64);
            let layer_data = ArrayData::from(coo_layer.to_csr::<u64>());
            let layer_name = format!("layer_{}", i);
            
            adata_t2.add_layer(layer_name.clone(), IMArrayElement::new(layer_data)).unwrap();
            adata_t2.layers().remove_array(&layer_name).unwrap();
            thread::yield_now();
        }
    }));
    
    // Thread 3: Reading obs and var
    let adata_t3 = Arc::clone(&adata);
    handles.push(thread::spawn(move || {
        for _ in 0..100 {
            let _obs = adata_t3.obs().get_data();
            let _var = adata_t3.var().get_data();
            thread::yield_now();
        }
    }));
    
    // Thread 4: Subsetting obsm (internal to IMAxisArrays)
    let adata_t4 = Arc::clone(&adata);
    handles.push(thread::spawn(move || {
        // First add something to obsm
        let mut coo_obsm = TriMatI::<f64, u32>::new((nrows, 10));
        for i in 0..10 { coo_obsm.add_triplet(i, i, 1.0); }
        let obsm_data = ArrayData::from(coo_obsm.to_csr::<u64>());
        adata_t4.obsm().add_array("X_pca".to_string(), IMArrayElement::new(obsm_data)).unwrap();
        
        let selection = SelectInfoElem::from(0..50);
        for _ in 0..20 {
            // This subsets all arrays in obsm
            adata_t4.obsm().subset_inplace(&[&selection, &SelectInfoElem::full()]).unwrap();
            // Note: in a real scenario we'd need to be careful about shrinking dimensions 
            // but for a lock-stress test this is fine if we don't crash.
            // Reset it back to full size for next iteration if needed, but here we just test locking.
        }
    }));

    for handle in handles {
        handle.join().unwrap();
    }
    
    Ok(())
}
