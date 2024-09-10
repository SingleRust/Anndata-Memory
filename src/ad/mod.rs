use anndata::{
    container::{Axis, Dim},
    data::{DataFrameIndex, SelectInfoElem},
    ArrayData, HasShape,
};
use log::{log, Level};
use helpers::IMAxisArrays;
use polars::{frame::DataFrame, prelude::NamedFrom, series::Series};

use crate::{base::DeepClone, IMArrayElement, IMDataFrameElement, IMElementCollection};

pub(crate) mod helpers;

pub struct IMAnnData {
    /// Number of observations (rows).
    pub(crate) n_obs: Dim,
    /// Number of variables (columns).
    pub(crate) n_vars: Dim,
    /// Data matrix.
    x: IMArrayElement,
    /// Observations metadata.
    obs: IMDataFrameElement,
    /// Observation multi-dimensional annotation.
    obsm: IMAxisArrays,
    /// Observation pairwise annotation.
    obsp: IMAxisArrays,
    /// Variables metadata.
    var: IMDataFrameElement,
    /// Variable multi-dimensional annotation.
    varm: IMAxisArrays,
    /// Variable pairwise annotation.
    varp: IMAxisArrays,
    /// Unstructured annotation.
    uns: IMElementCollection,
    /// Layers of data.
    layers: IMAxisArrays,
}

impl IMAnnData {
    /// Creates a new `IMAnnData` instance.
    ///
    /// # Arguments
    ///
    /// * `x` - Main data matrix.
    /// * `obs` - Observations metadata.
    /// * `var` - Variables metadata.
    ///
    /// # Returns
    ///
    /// Returns `Ok(IMAnnData)` if dimensions match, otherwise returns an `Err`.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions mismatch between `x`, `obs`, and `var`.
    pub fn new(
        x: IMArrayElement,
        obs: IMDataFrameElement,
        var: IMDataFrameElement,
    ) -> anyhow::Result<Self> {
        let n_obs = Dim::new(obs.get_data().height());
        let n_vars = Dim::new(var.get_data().height());
        // Validate dimensions
        let x_shape = x.get_shape()?;
        if x_shape[0] != n_obs.get() || x_shape[1] != n_vars.get() {
            return Err(anyhow::anyhow!("Dimensions mismatch"));
        }
        Ok(Self {
            n_obs: n_obs.clone(),
            n_vars: n_vars.clone(),
            x,
            obs,
            var,
            obsm: IMAxisArrays::new(Axis::Row, n_obs.clone(), None),
            obsp: IMAxisArrays::new(Axis::Pairwise, n_obs.clone(), None),
            varm: IMAxisArrays::new(Axis::Row, n_vars.clone(), None),
            varp: IMAxisArrays::new(Axis::Pairwise, n_vars.clone(), None),
            uns: IMElementCollection::new_empty(),
            layers: IMAxisArrays::new(Axis::RowColumn, n_obs.clone(), Some(n_vars.clone())),
        })
    }

    /// Creates a new basic `IMAnnData` instance from a sparse matrix and index names.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A sparse matrix (CsrArray) containing the main data.
    /// * `obs_names` - Names for the observations (rows).
    /// * `var_names` - Names for the variables (columns).
    ///
    /// # Returns
    ///
    /// Returns `Result<IMAnnData>` if successful, otherwise returns an `Err`.
    ///
    /// # Errors
    ///
    /// Returns an error if there's a mismatch in dimensions or if DataFrame creation fails.
    pub fn new_basic(
        matrix: ArrayData,
        obs_names: Vec<String>,
        var_names: Vec<String>,
    ) -> anyhow::Result<Self> {
        let s = matrix.shape();
        let n_obs = s[0];
        let n_vars = s[1];

        // Validate dimensions
        if n_obs != obs_names.len() || n_vars != var_names.len() {
            return Err(anyhow::anyhow!(
                "Dimensions mismatch between matrix and index names"
            ));
        }

        // Create basic obs DataFrame and IMDataFrameElement
        let obs_df = DataFrame::new(vec![Series::new("index", &obs_names)])?;
        let obs_index: DataFrameIndex = obs_names.into();
        let obs = IMDataFrameElement::new(obs_df, obs_index);

        // Create basic var DataFrame and IMDataFrameElement
        let var_df = DataFrame::new(vec![Series::new("index", &var_names)])?;
        let var_index: DataFrameIndex = var_names.into();
        let var = IMDataFrameElement::new(var_df, var_index);

        // Create the IMAnnData object
        IMAnnData::new(IMArrayElement::new(matrix), obs, var)
    }

    pub fn new_extended(
        matrix: ArrayData,
        obs_names: Vec<String>,
        var_names: Vec<String>,
        obs_df: DataFrame,
        var_df: DataFrame,
    ) -> anyhow::Result<Self> {
        let s = matrix.shape();
        let n_obs = s[0];
        let n_vars = s[1];

        // Validate dimensions
        if n_obs != obs_names.len() || n_vars != var_names.len() {
            return Err(anyhow::anyhow!(
                "Dimensions mismatch between matrix and index names"
            ));
        }

        // Create basic obs DataFrame and IMDataFrameElement
        let obs_index: DataFrameIndex = obs_names.into();
        let obs = IMDataFrameElement::new(obs_df, obs_index);

        // Create basic var DataFrame and IMDataFrameElement
        let var_index: DataFrameIndex = var_names.into();
        let var = IMDataFrameElement::new(var_df, var_index);

        // Create the IMAnnData object
        IMAnnData::new(IMArrayElement::new(matrix), obs, var)
    }

    /// Returns the number of observations.
    pub fn n_obs(&self) -> usize {
        self.n_obs.get()
    }

    /// Returns the number of variables.
    pub fn n_vars(&self) -> usize {
        self.n_vars.get()
    }

    pub fn obs_names(&self) -> Vec<String> {
        self.obs.get_index().into_vec()
    }

    pub fn var_names(&self) -> Vec<String> {
        self.var.get_index().into_vec()
    }

    /// Returns a shallow clone of the main data matrix.
    ///
    /// # Notes
    ///
    /// This method returns a new `IMArrayElement` that shares the same underlying data with the original.
    /// Modifications to the returned `IMArrayElement` will affect the original data.
    pub fn x(&self) -> IMArrayElement {
        self.x.clone()
    }

    /// Returns a shallow clone of the observations metadata.
    ///
    /// # Notes
    ///
    /// This method returns a new `IMDataFrameElement` that shares the same underlying data with the original.
    /// Modifications to the returned `IMDataFrameElement` will affect the original data.
    pub fn obs(&self) -> IMDataFrameElement {
        self.obs.clone()
    }

    /// Returns a shallow clone of the variable DataFrame.
    ///
    /// # Notes
    ///
    /// This method returns a new `IMDataFrameElement` that shares the same underlying data with the original.
    /// Modifications to the returned `IMDataFrameElement` will affect the original data.
    pub fn var(&self) -> IMDataFrameElement {
        self.var.clone()
    }

    /// Adds a new layer to the `layers` field.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the new layer.
    /// * `data` - Data for the new layer.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the layer was successfully added, otherwise returns an `Err`.
    ///
    /// # Errors
    ///
    /// Returns an error if a layer with the same name already exists.
    pub fn add_layer(&mut self, name: String, data: IMArrayElement) -> anyhow::Result<()> {
        self.layers.add_array(name, data)
    }

    /// Retrieves a deep clone of a layer by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the layer to retrieve.
    ///
    /// # Returns
    ///
    /// Returns `Ok(IMArrayElement)` if the layer was found, otherwise returns an `Err`.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer is not found.
    pub fn get_layer(&self, name: &str) -> anyhow::Result<IMArrayElement> {
        self.layers.get_array(name)
    }

    /// Retrieves a shallow clone of a layer by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the layer to retrieve.
    ///
    /// # Returns
    ///
    /// Returns `Ok(IMArrayElement)` if the layer was found, otherwise returns an `Err`.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer is not found.
    pub fn get_layer_shallow(&self, name: &str) -> anyhow::Result<IMArrayElement> {
        self.layers.get_array_shallow(name)
    }

    /// Removes a layer by name and returns it.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the layer to remove.
    ///
    /// # Returns
    ///
    /// Returns `Ok(IMArrayElement)` with the removed layer if found, otherwise returns an `Err`.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer is not found.
    pub fn remove_layer(&mut self, name: &str) -> anyhow::Result<IMArrayElement> {
        self.layers.remove_array(name)
    }

    /// Updates an existing layer with new data.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the layer to update.
    /// * `data` - New data for the layer.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the layer was successfully updated, otherwise returns an `Err`.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer is not found.
    pub fn update_array(&mut self, name: &str, data: IMArrayElement) -> anyhow::Result<()> {
        self.layers.update_array(name, data)
    }

    /// Returns a shallow clone of the observation multi-dimensional annotation.
    ///
    /// # Returns
    ///
    /// Returns an `IMAxisArrays` instance that shares the same underlying data with the original.
    ///
    /// # Notes
    ///
    /// This method performs a shallow clone, meaning the returned `IMAxisArrays` shares the same
    /// Arc pointer to the RwLock containing the data. Any modifications made through this clone
    /// will affect the original data in the `IMAnnData` instance.
    pub fn obsm(&self) -> IMAxisArrays {
        self.obsm.clone()
    }

    /// Returns a shallow clone of the observation pairwise annotation.
    ///
    /// # Returns
    ///
    /// Returns an `IMAxisArrays` instance that shares the same underlying data with the original.
    ///
    /// # Notes
    ///
    /// This method performs a shallow clone, meaning the returned `IMAxisArrays` shares the same
    /// Arc pointer to the RwLock containing the data. Any modifications made through this clone
    /// will affect the original data in the `IMAnnData` instance.
    pub fn obsp(&self) -> IMAxisArrays {
        self.obsp.clone()
    }

    /// Returns a shallow clone of the variable multi-dimensional annotation.
    ///
    /// # Returns
    ///
    /// Returns an `IMAxisArrays` instance that shares the same underlying data with the original.
    ///
    /// # Notes
    ///
    /// This method performs a shallow clone, meaning the returned `IMAxisArrays` shares the same
    /// Arc pointer to the RwLock containing the data. Any modifications made through this clone
    /// will affect the original data in the `IMAnnData` instance.
    pub fn varm(&self) -> IMAxisArrays {
        self.varm.clone()
    }

    /// Returns a shallow clone of the variable pairwise annotation.
    ///
    /// # Returns
    ///
    /// Returns an `IMAxisArrays` instance that shares the same underlying data with the original.
    ///
    /// # Notes
    ///
    /// This method performs a shallow clone, meaning the returned `IMAxisArrays` shares the same
    /// Arc pointer to the RwLock containing the data. Any modifications made through this clone
    /// will affect the original data in the `IMAnnData` instance.
    pub fn varp(&self) -> IMAxisArrays {
        self.varp.clone()
    }

    /// Returns a shallow clone of the unstructured annotation.
    ///
    /// # Returns
    ///
    /// Returns an `IMElementCollection` instance that shares the same underlying data with the original.
    ///
    /// # Notes
    ///
    /// This method performs a shallow clone, meaning the returned `IMElementCollection` shares the same
    /// Arc pointer to the RwLock containing the data. Any modifications made through this clone
    /// will affect the original data in the `IMAnnData` instance.
    pub fn uns(&self) -> IMElementCollection {
        self.uns.clone()
    }

    /// Returns a shallow clone of the layers of data.
    ///
    /// # Returns
    ///
    /// Returns an `IMAxisArrays` instance that shares the same underlying data with the original.
    ///
    /// # Notes
    ///
    /// This method performs a shallow clone, meaning the returned `IMAxisArrays` shares the same
    /// Arc pointer to the RwLock containing the data. Any modifications made through this clone
    /// will affect the original data in the `IMAnnData` instance.
    pub fn layers(&self) -> IMAxisArrays {
        self.layers.clone()
    }
    // !!!!! THIS IS VERY UNSAFE as it might allow for lock races, requires wrapping IMAnnData into a RwLock in order to prevent that, or transition to async data running of functions !!!!!
    pub fn subset_inplace(&mut self, selection: &[&SelectInfoElem]) -> anyhow::Result<()> {
        log!(Level::Debug, "Staring subsetting inplace");
        if selection.len() != 2 {
            return Err(anyhow::anyhow!("Invalid selection, only 2-dimensional selections are supported on the in-memory anndata object!"));
        }

        let obs_sel = selection[0];
        let var_sel = selection[1];

        // check if these changes are valid
        log!(Level::Debug, "Performing boundchecks");
        obs_sel.bound_check(self.n_obs())?;
        var_sel.bound_check(self.n_vars())?;

        log!(Level::Debug, "Subsetting X");
        self.x.subset_inplace(selection)?;
        log!(Level::Debug, "Subsetting obs");
        self.obs.subset_inplace(obs_sel)?;
        log!(Level::Debug, "Subsetting var");
        self.var.subset_inplace(var_sel)?;
        log!(Level::Debug, "Subsetting layers");
        self.layers.subset_inplace(selection)?;
        log!(Level::Debug, "Subsetting obsm");
        self.obsm
            .subset_inplace(vec![&obs_sel.clone(), &SelectInfoElem::full()].as_slice())?;
        log!(Level::Debug, "Subsetting obsp");
        self.obsp
            .subset_inplace(vec![&obs_sel.clone(), &obs_sel.clone()].as_slice())?;
        log!(Level::Debug, "Subsetting varm");
        self.varm
            .subset_inplace(vec![&var_sel.clone(), &SelectInfoElem::full()].as_slice())?;
        log!(Level::Debug, "Subsetting varp");
        self.varp
            .subset_inplace(vec![&var_sel.clone(), &var_sel.clone()].as_slice())?;

        Ok(())
    }

    pub fn subset(&self, selection: &[&SelectInfoElem]) -> anyhow::Result<Self> {
        if selection.len() != 2 {
            return Err(anyhow::anyhow!("Invalid selection, only 2-dimensional selections are supported on the in-memory anndata object!"));
        }

        let obs_sel = selection[0];
        let var_sel = selection[1];

        // check if these changes are valid
        obs_sel.bound_check(self.n_obs())?;
        var_sel.bound_check(self.n_vars())?;

        
        let obs = self.obs.subset(obs_sel)?;
        let var = self.var.subset(var_sel)?;
        let layers = self.layers.subset(selection)?;
        let obsm = self.obsm.subset(vec![&obs_sel.clone(), &SelectInfoElem::full()].as_slice())?;
        let obsp = self.obsp.subset(vec![&obs_sel.clone(), &obs_sel.clone()].as_slice())?;
        let varm = self.varm.subset(vec![&var_sel.clone(), &SelectInfoElem::full()].as_slice())?;
        let varp = self.varp.subset(vec![&var_sel.clone(), &var_sel.clone()].as_slice())?;

        let x = self.x.subset(selection)?;

        Ok(IMAnnData {
            n_obs: Dim::new(obs.get_data().height()),
            n_vars: Dim::new(var.get_data().height()),
            x,
            obs,
            obsm,
            obsp,
            var,
            varm,
            varp,
            uns: self.uns.clone(),
            layers,
        })
    }

}

use std::fmt;

impl fmt::Display for IMAnnData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IMAnnData Object")?;
        writeln!(f, "-----------------")?;
        writeln!(
            f,
            "Dimensions: {} observations x {} variables",
            self.n_obs(),
            self.n_vars()
        )?;

        // X matrix info
        let x_shape = self.x().get_shape().map_err(|_| fmt::Error)?;
        writeln!(
            f,
            "X: {:?} {}",
            x_shape,
            self.x().get_type().map_err(|_| fmt::Error)?
        )?;

        // Layers info
        let layer_keys = self.layers().keys();
        writeln!(
            f,
            "Layers: {} - {}",
            layer_keys.len(),
            layer_keys.join(", ")
        )?;

        // Obs and Var info
        writeln!(
            f,
            "Obs DataFrame Shape: {:?}",
            self.obs().get_data().shape()
        )?;
        writeln!(
            f,
            "Var DataFrame Shape: {:?}",
            self.var().get_data().shape()
        )?;

        // Obsm, Obsp, Varm, Varp info
        writeln!(f, "Obsm keys: {}", self.obsm().keys().join(", "))?;
        writeln!(f, "Obsp keys: {}", self.obsp().keys().join(", "))?;
        writeln!(f, "Varm keys: {}", self.varm().keys().join(", "))?;
        writeln!(f, "Varp keys: {}", self.varp().keys().join(", "))?;

        // Uns info

        Ok(())
    }
}

impl DeepClone for IMAnnData {
    fn deep_clone(&self) -> Self {
        Self {
            n_obs: self.n_obs.clone(),
            n_vars: self.n_vars.clone(),
            x: self.x.deep_clone(),
            obs: self.obs.deep_clone(),
            obsm: self.obsm.deep_clone(),
            obsp: self.obsp.deep_clone(),
            var: self.var.deep_clone(),
            varm: self.varm.deep_clone(),
            varp: self.varp.deep_clone(),
            uns: self.uns.deep_clone(),
            layers: self.layers.deep_clone(),
        }
    }
}
