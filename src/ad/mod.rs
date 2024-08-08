use anndata::container::{Axis, Dim};
use helpers::IMAxisArrays;

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

    pub fn n_obs(&self) -> usize {
        self.n_obs.get()
    }

    pub fn n_vars(&self) -> usize {
        self.n_vars.get()
    }

    pub fn x(&self) -> IMArrayElement {
        self.x
    }

    pub fn obs(&self) -> IMDataFrameElement {
        self.obs
    }

    pub fn var(&self) -> IMDataFrameElement {
        self.var
    }

    pub fn add_layer(&mut self, name: String, data: IMArrayElement) -> anyhow::Result<()> {
        self.layers.add_array(name, data)
    }

    pub fn get_layer(&self, name: &str) -> anyhow::Result<IMArrayElement> {
        self.layers.get_array(name)
    }

    pub fn get_layer_shallow(&self, name: &str) -> anyhow::Result<IMArrayElement> {
        self.layers.get_array_shallow(name)
    }
    
    pub fn remove_layer(&mut self, name: &str) -> anyhow::Result<IMArrayElement> {
        self.layers.remove_array(name)
    }

    pub fn update_array(&mut self, name: &str, data: IMArrayElement) -> anyhow::Result<()> {
        self.layers.update_array(name, data)
    }

    
}
