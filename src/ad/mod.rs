use anndata::{container::{Axis, Dim}, data::DataFrameIndex, ArrayData, HasShape};
use helpers::IMAxisArrays;
use polars::{frame::DataFrame, prelude::NamedFrom, series::Series};

use crate::{IMArrayElement, IMDataFrameElement, IMElementCollection};

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
            return Err(anyhow::anyhow!("Dimensions mismatch between matrix and index names"));
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

    /// Returns the number of observations.
    pub fn n_obs(&self) -> usize {
        self.n_obs.get()
    }

    /// Returns the number of variables.
    pub fn n_vars(&self) -> usize {
        self.n_vars.get()
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
}