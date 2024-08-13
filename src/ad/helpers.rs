use std::{
    collections::HashMap,
    fmt,
    ops::{Deref, DerefMut},
};

use anndata::{
    backend::DataType,
    container::{Axis, Dim},
    data::{DataFrameIndex, SelectInfoElem, Shape},
    ArrayData, ArrayOp, Data, HasShape, WriteData,
};
use polars::{
    frame::DataFrame,
    prelude::{IdxCa, NamedFrom},
    series::Series,
};

use crate::base::RwSlot;

pub struct IMArrayElement(pub RwSlot<ArrayData>);

impl IMArrayElement {
    pub fn new(data: ArrayData) -> Self {
        IMArrayElement(RwSlot::new(data))
    }

    pub fn get_type(&self) -> anyhow::Result<DataType> {
        Ok(self.0.read_inner().data_type())
    }

    pub fn get_shape(&self) -> anyhow::Result<Shape> {
        Ok(self.0.read_inner().shape())
    }

    pub fn get_data(&self) -> anyhow::Result<ArrayData> {
        Ok(self.0.read_inner().clone())
    }

    pub fn set_data(&self, data: ArrayData) -> anyhow::Result<()> {
        let mut write_guard = self.0.lock_write();
        let d = write_guard.deref_mut();
        *d = Some(data);
        Ok(())
    }

    pub fn subset_inplace(&self, s: &[&SelectInfoElem]) -> anyhow::Result<()> {
        let mut write_guard = self.0.write_inner();
        let d = write_guard.deref_mut();

        // Perform the selection operation directly on d
        *d = d.select(s);

        Ok(())
    }

    pub fn subset(&self, s: &[&SelectInfoElem]) -> anyhow::Result<Self> {
        let read_guard = self.0.read_inner();
        let d = read_guard.deref();

        // Return a new ArrayData by selecting from d
        Ok(IMArrayElement::new(d.select(s)))
    }

    pub fn deep_clone(&self) -> anyhow::Result<Self> {
        Ok(IMArrayElement(self.0.deep_clone()))
    }

    pub fn deep_clone_content(&self) -> anyhow::Result<ArrayData> {
        Ok(self.0.read_inner().clone())
    }
}

impl Clone for IMArrayElement {
    fn clone(&self) -> Self {
        IMArrayElement(self.0.clone())
    }
}

pub struct IMDataFrameElement(RwSlot<InnerIMDataFrame>);

pub struct InnerIMDataFrame {
    df: DataFrame,
    pub index: DataFrameIndex,
}

impl Clone for InnerIMDataFrame {
    fn clone(&self) -> Self {
        InnerIMDataFrame {
            df: self.df.clone(),
            index: self.index.clone(),
        }
    }
}

impl Clone for IMDataFrameElement {
    /// Shallow clone of the IMDataFrameElement
    fn clone(&self) -> Self {
        IMDataFrameElement(self.0.clone())
    }
}

impl IMDataFrameElement {
    pub fn new(df: DataFrame, index: DataFrameIndex) -> Self {
        if df.height() == 0 {
            let tmp_df =
                DataFrame::new(vec![Series::new("index", &index.clone().into_vec())]).unwrap();
            return IMDataFrameElement(RwSlot::new(InnerIMDataFrame { df: tmp_df, index }));
        }
        if df.height() != index.len() {
            panic!("Length of index does not match length of DataFrame");
        }
        IMDataFrameElement(RwSlot::new(InnerIMDataFrame { df, index }))
    }

    pub fn get_data(&self) -> DataFrame {
        self.0.read_inner().df.clone()
    }

    pub fn get_index(&self) -> DataFrameIndex {
        self.0.read_inner().index.clone()
    }

    pub fn set_both(&self, df: DataFrame, index: DataFrameIndex) -> anyhow::Result<()> {
        let mut write_guard = self.0.lock_write();
        let d = write_guard.as_mut();
        match d {
            Some(data) => {
                if data.df.height() != df.height()
                    || data.index.len() != index.len()
                    || data.df.height() != index.len()
                {
                    return Err(anyhow::anyhow!(
                        "Length of index does not match length of DataFrame"
                    ));
                }

                data.df = df;
                data.index = index;
                Ok(())
            }
            None => Err(anyhow::anyhow!("DataFrame is not initialized")),
        }
    }

    pub fn set_data(&self, df: DataFrame) -> anyhow::Result<()> {
        let mut write_guard = self.0.lock_write();
        let d = write_guard.as_mut();
        match d {
            Some(data) => {
                if data.index.len() != df.height() || data.df.height() != df.height() {
                    return Err(anyhow::anyhow!(
                        "Length of index does not match length of DataFrame"
                    ));
                }
                data.df = df;
                Ok(())
            }
            None => Err(anyhow::anyhow!("DataFrame is not initialized")),
        }
    }

    pub fn set_index(&self, index: DataFrameIndex) -> anyhow::Result<()> {
        let mut write_guard = self.0.lock_write();
        let d = write_guard.as_mut();
        match d {
            Some(data) => {
                if data.df.height() != index.len() || data.index.len() != index.len() {
                    return Err(anyhow::anyhow!(
                        "Length of index does not match length of DataFrame"
                    ));
                }

                data.index = index;
                Ok(())
            }
            None => Err(anyhow::anyhow!("DataFrame is not initialized")),
        }
    }

    pub fn attach_column_to_df(&self, column: Series) -> anyhow::Result<()> {
        let mut write_guard = self.0.lock_write();
        let d = write_guard.as_mut();
        match d {
            Some(data) => {
                if data.df.height() != column.len() {
                    return Err(anyhow::anyhow!(
                        "Length of column does not match length of DataFrame"
                    ));
                }
                data.df.with_column(column)?;
                Ok(())
            }
            None => Err(anyhow::anyhow!("DataFrame is not initialized")),
        }
    }

    pub fn remove_column_from_df(&self, column_name: &str) -> anyhow::Result<()> {
        let mut write_guard = self.0.lock_write();
        let d = write_guard.as_mut();
        match d {
            Some(data) => {
                let _ = data.df.drop_in_place(column_name)?;
                Ok(())
            }
            None => Err(anyhow::anyhow!("DataFrame is not initialized")),
        }
    }

    pub fn get_column_from_df(&self, column_name: &str) -> anyhow::Result<Series> {
        let read_guard = self.0.lock_read();
        let d = read_guard.as_ref();
        match d {
            Some(data) => match data.df.column(column_name) {
                Ok(series) => Ok(series.clone()),
                Err(e) => Err(anyhow::anyhow!("Column not found: {}", e)),
            },
            None => Err(anyhow::anyhow!("DataFrame is not initialized")),
        }
    }

    pub fn set_column_in_df(&self, column_name: &str, column: Series) -> anyhow::Result<()> {
        let mut write_guard = self.0.lock_write();
        let d = write_guard.as_mut();
        match d {
            Some(data) => {
                data.df.replace(column_name, column)?;
                Ok(())
            }
            None => Err(anyhow::anyhow!("DataFrame is not initialized")),
        }
    }

    pub fn deep_clone(&self) -> anyhow::Result<Self> {
        Ok(IMDataFrameElement(self.0.deep_clone()))
    }

    pub fn subset_inplace(&self, s: &SelectInfoElem) -> anyhow::Result<()> {
        let mut write_guard = self.0.write_inner();
        let d = write_guard.deref_mut();
        let indices = crate::utils::select_info_elem_to_indices(s, d.index.len())?;
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let idx = IdxCa::new("idx", &indices_u32);
        // Perform the selection operation directly on d
        let ind = d.index.clone().into_vec();
        let ind_subset: Vec<String> = indices.iter().map(|&i| ind[i].clone()).collect();
        let df_subset = d.df.take(&idx)?;
        self.set_both(df_subset, DataFrameIndex::from(ind_subset))
    }

    pub fn subset(&self, s: &SelectInfoElem) -> anyhow::Result<Self> {
        let read_guard = self.0.lock_read();
        let d = read_guard.as_ref().unwrap();
        let indices = crate::utils::select_info_elem_to_indices(s, d.index.len())?;
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let idx = IdxCa::new("idx", &indices_u32);
        let ind = d.index.clone().into_vec();
        let ind_subset: Vec<String> = indices.iter().map(|&i| ind[i].clone()).collect();
        let df_subset = d.df.take(&idx)?;
        Ok(Self::new(df_subset, DataFrameIndex::from(ind_subset)))
    }
}

pub struct IMAxisArrays(pub RwSlot<InnerIMAxisArray>);

impl Clone for IMAxisArrays {
    fn clone(&self) -> Self {
        IMAxisArrays(self.0.clone())
    }
}

pub struct InnerIMAxisArray {
    pub axis: Axis,
    pub(crate) dim1: Dim,
    pub(crate) dim2: Option<Dim>,
    data: HashMap<String, IMArrayElement>,
}

impl Clone for InnerIMAxisArray {
    fn clone(&self) -> Self {
        InnerIMAxisArray {
            axis: self.axis,
            dim1: self.dim1.clone(),
            dim2: self.dim2.clone(),
            data: self.data.clone(),
        }
    }
}

impl fmt::Display for IMAxisArrays {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let read_guard = self.0.read_inner();

        writeln!(f, "IMAxisArrays {{")?;
        writeln!(f, "    Axis: {:?}", read_guard.axis)?;
        writeln!(f, "    Dim1: {}", read_guard.dim1)?;
        if let Some(dim2) = &read_guard.dim2 {
            writeln!(f, "    Dim2: {}", dim2)?;
        }
        writeln!(f, "    Arrays: {{")?;
        for (key, value) in &read_guard.data {
            let shape = value.get_shape().map_err(|_| fmt::Error)?;
            writeln!(f, "        {}: {:?}", key, shape)?;
        }
        writeln!(f, "    }}")?;
        write!(f, "}}")
    }
}

impl IMAxisArrays {
    // Create a new IMAxisArrays
    pub fn new(axis: Axis, dim1: Dim, dim2: Option<Dim>) -> Self {
        let inner = InnerIMAxisArray {
            axis,
            dim1,
            dim2,
            data: HashMap::new(),
        };
        IMAxisArrays(RwSlot::new(inner))
    }

    pub fn new_from(
        axis: Axis,
        dim1: Dim,
        dim2: Option<Dim>,
        data: HashMap<String, IMArrayElement>,
    ) -> Self {
        let inner = InnerIMAxisArray {
            axis,
            dim1,
            dim2,
            data,
        };
        IMAxisArrays(RwSlot::new(inner))
    }

    pub fn add_array(&self, key: String, element: IMArrayElement) -> anyhow::Result<()> {
        let mut write_guard = self.0.write_inner();
        let imarray = write_guard.deref_mut();
        // Check if the key already exists
        if imarray.data.contains_key(&key) {
            return Err(anyhow::anyhow!("Key already exists"));
        }

        // Get the shape of the input element
        let shape = element.get_shape()?;
        let dim1 = imarray.dim1.get();
        let dim2 = imarray.dim2.clone().unwrap_or(Dim::new(0)).get();

        // Perform dimensionality checks based on the axis type
        match imarray.axis {
            Axis::Row => {
                if shape[0] != dim1 {
                    return Err(anyhow::anyhow!(
                        "Data shape {:?} does not match expected row dimension {}",
                        shape,
                        dim1
                    ));
                }
            }
            Axis::RowColumn => {
                if shape[0] != dim1 || shape[1] != dim2 {
                    return Err(anyhow::anyhow!(
                        "Data shape {:?} does not match expected dimensions ({}, {})",
                        shape,
                        dim1,
                        dim2
                    ));
                }
            }
            Axis::Pairwise => {
                if shape[0] != dim1 || shape[1] != dim1 {
                    return Err(anyhow::anyhow!(
                        "Data shape {:?} does not match expected pairwise dimensions ({}, {})",
                        shape,
                        dim1,
                        dim1
                    ));
                }
            }
        }

        // If all checks pass, insert the element
        imarray.data.insert(key, element);
        Ok(())
    }

    // Get an array element (returns a deep clone to avoid holding the read lock)
    pub fn get_array(&self, key: &str) -> anyhow::Result<IMArrayElement> {
        let read_guard = self.0.read_inner();
        read_guard
            .data
            .get(key)
            .map(|element| element.deep_clone().unwrap())
            .ok_or_else(|| anyhow::anyhow!("Key not found"))
    }

    // New method: Get an array element (returns a shallow clone)
    pub fn get_array_shallow(&self, key: &str) -> anyhow::Result<IMArrayElement> {
        let read_guard = self.0.read_inner();
        read_guard
            .data
            .get(key)
            .cloned() // This performs a shallow clone
            .ok_or_else(|| anyhow::anyhow!("Key not found"))
    }

    // Remove an array element
    pub fn remove_array(&self, key: &str) -> anyhow::Result<IMArrayElement> {
        let mut write_guard = self.0.write_inner();
        write_guard
            .data
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("Key not found"))
    }

    // Get the number of arrays
    pub fn len(&self) -> usize {
        let read_guard = self.0.read_inner();
        read_guard.data.len()
    }

    // Check if there are any arrays
    pub fn is_empty(&self) -> bool {
        let read_guard = self.0.read_inner();
        read_guard.data.is_empty()
    }

    // Get all keys
    pub fn keys(&self) -> Vec<String> {
        let read_guard = self.0.read_inner();
        read_guard.data.keys().cloned().collect()
    }

    // Get the axis
    pub fn axis(&self) -> Axis {
        let read_guard = self.0.read_inner();
        read_guard.axis
    }

    // Get dimensions
    pub fn dimensions(&self) -> (Dim, Option<Dim>) {
        let read_guard = self.0.read_inner();
        (read_guard.dim1.clone(), read_guard.dim2.clone())
    }

    // Update an existing array element
    pub fn update_array(&self, key: &str, new_element: IMArrayElement) -> anyhow::Result<()> {
        let mut write_guard = self.0.write_inner();
        if let Some(element) = write_guard.data.get_mut(key) {
            *element = new_element;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Key not found"))
        }
    }

    pub fn subset_inplace(&self, s: &[&SelectInfoElem]) -> anyhow::Result<()> {
        let mut write_guard = self.0.write_inner();
        let imarray = write_guard.deref_mut();
        let dim1_indices = crate::utils::select_info_elem_to_indices(s[0], imarray.dim1.get())?;
        imarray.dim1.try_set(dim1_indices.len())?;

        if let Some(dim2) = &mut imarray.dim2 {
            if s.len() < 2 {
                return Err(anyhow::anyhow!(
                    "Subset operation requires two selection elements"
                ));
            }
            let dim2_indices = crate::utils::select_info_elem_to_indices(s[1], dim2.get())?;
            dim2.try_set(dim2_indices.len())?;
        }

        for element in imarray.data.values_mut() {
            element.subset_inplace(s)?;
        }

        Ok(())
    }

    pub fn subset(
        &self,
        s: &[&SelectInfoElem]
    ) -> anyhow::Result<Self> {
        let read_guard = self.0.read_inner();
        let imarray = read_guard.deref();
        let dim1_indices = crate::utils::select_info_elem_to_indices(s[0], imarray.dim1.get())?;
        let new_dim1 = Dim::new(dim1_indices.len());
        let mut new_dim2: Option<Dim> = None;
        if imarray.dim2.is_some() {
            if s.len() < 2 {
                return Err(anyhow::anyhow!(
                    "Subset operation requires two selection elements"
                ));
            }
            let dim2_indices = crate::utils::select_info_elem_to_indices(s[1], imarray.dim2.clone().unwrap().get())?;
            new_dim2 = Some(Dim::new(dim2_indices.len()));
        }
        let mut new_data = HashMap::new();
        for (key, element) in &imarray.data {
            new_data.insert(key.clone(), element.subset(s)?);
        }

        Ok(IMAxisArrays::new_from(
            imarray.axis,
            new_dim1,
            new_dim2,
            new_data,
        ))
    }

    // Perform an operation on all arrays
    pub fn map<F>(&self, f: F) -> anyhow::Result<()>
    where
        F: Fn(&mut IMArrayElement) -> anyhow::Result<()>,
    {
        let mut write_guard = self.0.write_inner();
        for element in write_guard.data.values_mut() {
            f(element)?;
        }
        Ok(())
    }

    pub fn deep_clone(&self) -> anyhow::Result<Self> {
        Ok(IMAxisArrays(self.0.deep_clone()))
    }
}

pub struct Element(pub RwSlot<Data>);

impl Clone for Element {
    fn clone(&self) -> Self {
        Element(self.0.clone())
    }
}

impl Element {
    pub fn new(data: Data) -> Self {
        Element(RwSlot::new(data))
    }

    pub fn get_data(&self) -> anyhow::Result<Data> {
        Ok(self.0.read_inner().clone())
    }

    pub fn set_data(&self, data: Data) -> anyhow::Result<()> {
        let mut write_guard = self.0.lock_write();
        let d = write_guard.deref_mut();
        *d = Some(data);
        Ok(())
    }

    pub fn deep_clone(&self) -> anyhow::Result<Self> {
        Ok(Element(self.0.deep_clone()))
    }
}

pub struct IMElementCollection(pub RwSlot<HashMap<String, Element>>);

impl Clone for IMElementCollection {
    fn clone(&self) -> Self {
        IMElementCollection(self.0.clone())
    }
}

impl IMElementCollection {
    pub fn new_empty() -> Self {
        IMElementCollection(RwSlot::new(HashMap::new()))
    }

    pub fn add_data(&self, key: String, element: Element) -> anyhow::Result<()> {
        let mut write_guard = self.0.write_inner();
        let collection = write_guard.deref_mut();
        if collection.contains_key(&key) {
            return Err(anyhow::anyhow!("Key already exists"));
        }
        collection.insert(key, element);
        Ok(())
    }

    pub fn remove_data(&self, key: &str) -> anyhow::Result<Element> {
        let mut write_guard = self.0.write_inner();
        write_guard
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("Key not found"))
    }

    pub fn get_data(&self, key: &str) -> anyhow::Result<Element> {
        let read_guard = self.0.read_inner();
        read_guard
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Key not found"))
    }

    pub fn get_data_deep(&self, key: &str) -> anyhow::Result<Element> {
        let read_guard = self.0.read_inner();
        read_guard
            .get(key)
            .map(|element| element.deep_clone().unwrap())
            .ok_or_else(|| anyhow::anyhow!("Key not found"))
    }
}
