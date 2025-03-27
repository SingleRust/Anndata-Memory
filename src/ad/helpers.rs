use std::{
    collections::HashMap,
    fmt,
    ops::{Deref, DerefMut},
};

use anndata::{
    backend::DataType,
    container::{Axis, Dim},
    data::{DataFrameIndex, DynArray, DynCscMatrix, DynCsrMatrix, Element, SelectInfoElem, Shape},
    ArrayData, Data, HasShape, Selectable,
};
use anyhow::bail;

use nalgebra_sparse::CsrMatrix;
use ndarray::Array2;
use polars::{
    frame::DataFrame,
    prelude::{Column, IdxCa, NamedFrom},
    series::Series,
};

use crate::base::DeepClone;
use crate::base::RwSlot;

impl DeepClone for ArrayData {
    fn deep_clone(&self) -> Self {
        self.clone()
    }
}

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

    pub fn convert_matrix_format(&self) -> anyhow::Result<()> {
        let mut write_guard = self.0.write_inner();
        let d = write_guard.deref_mut();

        // Create a placeholder that we can swap with - use an empty dense array as it's likely the smallest
        let ddata: Array2<f64> = Array2::zeros((0, 0));
        let placeholder = ArrayData::Array(DynArray::from(ddata));

        // Take ownership using replace
        let matrix_data = std::mem::replace(d, placeholder);

        let converted = match matrix_data {
            ArrayData::CsrMatrix(dyn_csr_matrix) => {
                let csc = match dyn_csr_matrix {
                    DynCsrMatrix::F64(m) => DynCscMatrix::F64(m.transpose_as_csc()),
                    DynCsrMatrix::F32(m) => DynCscMatrix::F32(m.transpose_as_csc()),
                    DynCsrMatrix::I64(m) => DynCscMatrix::I64(m.transpose_as_csc()),
                    DynCsrMatrix::I32(m) => DynCscMatrix::I32(m.transpose_as_csc()),
                    DynCsrMatrix::I16(m) => DynCscMatrix::I16(m.transpose_as_csc()),
                    DynCsrMatrix::I8(m) => DynCscMatrix::I8(m.transpose_as_csc()),
                    DynCsrMatrix::U64(m) => DynCscMatrix::U64(m.transpose_as_csc()),
                    DynCsrMatrix::U32(m) => DynCscMatrix::U32(m.transpose_as_csc()),
                    DynCsrMatrix::U16(m) => DynCscMatrix::U16(m.transpose_as_csc()),
                    DynCsrMatrix::U8(m) => DynCscMatrix::U8(m.transpose_as_csc()),
                    DynCsrMatrix::Bool(m) => DynCscMatrix::Bool(m.transpose_as_csc()),
                    DynCsrMatrix::String(m) => DynCscMatrix::String(m.transpose_as_csc()),
                };
                ArrayData::CscMatrix(csc)
            }
            ArrayData::CscMatrix(dyn_csc_matrix) => {
                let csr = match dyn_csc_matrix {
                    DynCscMatrix::F64(m) => DynCsrMatrix::F64(m.transpose_as_csr()),
                    DynCscMatrix::F32(m) => DynCsrMatrix::F32(m.transpose_as_csr()),
                    DynCscMatrix::I64(m) => DynCsrMatrix::I64(m.transpose_as_csr()),
                    DynCscMatrix::I32(m) => DynCsrMatrix::I32(m.transpose_as_csr()),
                    DynCscMatrix::I16(m) => DynCsrMatrix::I16(m.transpose_as_csr()),
                    DynCscMatrix::I8(m) => DynCsrMatrix::I8(m.transpose_as_csr()),
                    DynCscMatrix::U64(m) => DynCsrMatrix::U64(m.transpose_as_csr()),
                    DynCscMatrix::U32(m) => DynCsrMatrix::U32(m.transpose_as_csr()),
                    DynCscMatrix::U16(m) => DynCsrMatrix::U16(m.transpose_as_csr()),
                    DynCscMatrix::U8(m) => DynCsrMatrix::U8(m.transpose_as_csr()),
                    DynCscMatrix::Bool(m) => DynCsrMatrix::Bool(m.transpose_as_csr()),
                    DynCscMatrix::String(m) => DynCsrMatrix::String(m.transpose_as_csr()),
                };
                ArrayData::CsrMatrix(csr)
            }
            _ => {
                // Put back the original value since we're erroring
                *d = matrix_data;
                bail!("This datatype is not supported, only CSC and CSR matrices are supported.")
            }
        };

        *d = converted;
        Ok(())
    }

    pub fn subset(&self, s: &[&SelectInfoElem]) -> anyhow::Result<Self> {
        let read_guard = self.0.read_inner();
        let d = read_guard.deref();

        // Return a new ArrayData by selecting from d
        Ok(IMArrayElement::new(d.select(s)))
    }

    pub fn deep_clone_content(&self) -> anyhow::Result<ArrayData> {
        Ok(self.0.read_inner().clone())
    }

    // pub fn change_matrix_type<T>(&self) -> anyhow::Result<()> {
    //     let mut write_guard = self.0.write_inner();
    //     let d = write_guard.deref_mut();
    //
    //     // Create a placeholder that we can swap with - use an empty dense array as it's likely the smallest
    //     let ddata: Array2<f64> = Array2::zeros((0, 0));
    //     let placeholder = ArrayData::Array(DynArray::from(ddata));
    //
    //     // Take ownership using replace
    //     let matrix_data = std::mem::replace(d, placeholder);
    //
    //     let converted_matrix = match matrix_data {
    //         ArrayData::Array(dyn_array) => todo!(),
    //         ArrayData::CsrMatrix(dyn_csr_matrix) => {
    //             let csr_matrix: CsrMatrix<T> = match dyn_csr_matrix {
    //                 DynCsrMatrix::I8(csr_matrix) => csr_matrix.try_into()?,
    //                 DynCsrMatrix::I16(csr_matrix) => todo!(),
    //                 DynCsrMatrix::I32(csr_matrix) => todo!(),
    //                 DynCsrMatrix::I64(csr_matrix) => todo!(),
    //                 DynCsrMatrix::U8(csr_matrix) => todo!(),
    //                 DynCsrMatrix::U16(csr_matrix) => todo!(),
    //                 DynCsrMatrix::U32(csr_matrix) => todo!(),
    //                 DynCsrMatrix::U64(csr_matrix) => todo!(),
    //                 DynCsrMatrix::F32(csr_matrix) => todo!(),
    //                 DynCsrMatrix::F64(csr_matrix) => todo!(),
    //                 DynCsrMatrix::Bool(csr_matrix) => todo!(),
    //                 DynCsrMatrix::String(csr_matrix) => todo!(),
    //             };
    //             ArrayData::from(csr_matrix)
    //         }
    //         ArrayData::CsrNonCanonical(dyn_csr_non_canonical) => todo!(),
    //         ArrayData::CscMatrix(dyn_csc_matrix) => todo!(),
    //         ArrayData::DataFrame(data_frame) => todo!(),
    //     };
    //     * d = converted_matrix;
    //     Ok(())
    // }
}

impl DeepClone for IMArrayElement {
    fn deep_clone(&self) -> Self {
        IMArrayElement(self.0.deep_clone())
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

impl DeepClone for InnerIMDataFrame {
    fn deep_clone(&self) -> Self {
        self.clone()
    }
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
                DataFrame::new(vec![Column::new("index".into(), &index.clone().into_vec())])
                    .unwrap();
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
                if index.len() != df.height() {
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

    pub fn get_column_from_df(&self, column_name: &str) -> anyhow::Result<Column> {
        let read_guard = self.0.lock_read();
        let d = read_guard.as_ref();
        match d {
            Some(data) => match data.df.column(column_name) {
                Ok(column) => Ok(column.clone()),
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

    pub fn subset_inplace(&self, s: &SelectInfoElem) -> anyhow::Result<()> {
        let read_guard = self.0.lock_read();
        let d = read_guard.as_ref().unwrap();
        let indices = crate::utils::select_info_elem_to_indices(s, d.index.len())?;
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let idx = IdxCa::new("idx".into(), &indices_u32);
        let ind = d.index.clone().into_vec();
        let ind_subset: Vec<String> = indices.iter().map(|&i| ind[i].clone()).collect();
        let df_subset = d.df.take(&idx)?;
        drop(read_guard);
        self.set_both(df_subset, DataFrameIndex::from(ind_subset))
    }

    pub fn subset(&self, s: &SelectInfoElem) -> anyhow::Result<Self> {
        let read_guard = self.0.lock_read();
        let d = read_guard.as_ref().unwrap();
        let indices = crate::utils::select_info_elem_to_indices(s, d.index.len())?;
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let idx = IdxCa::new("idx".into(), &indices_u32);
        let ind = d.index.clone().into_vec();
        let ind_subset: Vec<String> = indices.iter().map(|&i| ind[i].clone()).collect();
        let df_subset = d.df.take(&idx)?;
        Ok(Self::new(df_subset, DataFrameIndex::from(ind_subset)))
    }
}

impl DeepClone for IMDataFrameElement {
    fn deep_clone(&self) -> Self {
        IMDataFrameElement(self.0.deep_clone())
    }
}

pub struct IMAxisArrays(pub RwSlot<InnerIMAxisArray>);

impl Clone for IMAxisArrays {
    fn clone(&self) -> Self {
        IMAxisArrays(self.0.clone())
    }
}

impl DeepClone for IMAxisArrays {
    fn deep_clone(&self) -> Self {
        IMAxisArrays(self.0.deep_clone())
    }
}

pub struct InnerIMAxisArray {
    pub axis: Axis,
    pub(crate) dim1: Dim,
    pub(crate) dim2: Option<Dim>,
    data: HashMap<String, IMArrayElement>,
}

impl DeepClone for InnerIMAxisArray {
    fn deep_clone(&self) -> Self {
        InnerIMAxisArray {
            axis: self.axis,
            dim1: self.dim1.clone(),
            dim2: self.dim2.clone(),
            data: self
                .data
                .iter()
                .map(|(k, v)| (k.clone(), v.deep_clone()))
                .collect(),
        }
    }
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
            .map(|element| element.deep_clone())
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
        imarray.dim1 = Dim::new(dim1_indices.len());

        if let Some(dim2) = &mut imarray.dim2 {
            if s.len() < 2 {
                return Err(anyhow::anyhow!(
                    "Subset operation requires two selection elements"
                ));
            }
            let dim2_indices = crate::utils::select_info_elem_to_indices(s[1], dim2.get())?;
            *dim2 = Dim::new(dim2_indices.len());
        }

        for element in imarray.data.values_mut() {
            element.subset_inplace(s)?;
        }

        Ok(())
    }

    pub fn subset(&self, s: &[&SelectInfoElem]) -> anyhow::Result<Self> {
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
            let dim2_indices = crate::utils::select_info_elem_to_indices(
                s[1],
                imarray.dim2.clone().unwrap().get(),
            )?;
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
}

pub struct IMElement(pub RwSlot<Data>);

impl DeepClone for Data {
    fn deep_clone(&self) -> Self {
        self.clone()
    }
}

impl DeepClone for IMElement {
    fn deep_clone(&self) -> Self {
        IMElement(self.0.deep_clone())
    }
}

impl Clone for IMElement {
    fn clone(&self) -> Self {
        IMElement(self.0.clone())
    }
}

impl IMElement {
    pub fn new(data: Data) -> Self {
        IMElement(RwSlot::new(data))
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
}

pub struct IMElementCollection(pub RwSlot<HashMap<String, IMElement>>);

impl DeepClone for IMElementCollection {
    fn deep_clone(&self) -> Self {
        let temp_data = self.0.read_inner();
        let data = temp_data.deref();
        let mut new_data = HashMap::new();
        for (key, value) in data.iter() {
            new_data.insert(key.clone(), value.deep_clone());
        }
        IMElementCollection(RwSlot::new(new_data))
    }
}

impl Clone for IMElementCollection {
    fn clone(&self) -> Self {
        IMElementCollection(self.0.clone())
    }
}

impl IMElementCollection {
    pub fn new_empty() -> Self {
        IMElementCollection(RwSlot::new(HashMap::new()))
    }

    pub fn add_data(&self, key: String, element: IMElement) -> anyhow::Result<()> {
        let mut write_guard = self.0.write_inner();
        let collection = write_guard.deref_mut();
        if collection.contains_key(&key) {
            return Err(anyhow::anyhow!("Key already exists"));
        }
        collection.insert(key, element);
        Ok(())
    }

    pub fn remove_data(&self, key: &str) -> anyhow::Result<IMElement> {
        let mut write_guard = self.0.write_inner();
        write_guard
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("Key not found"))
    }

    pub fn get_data(&self, key: &str) -> anyhow::Result<IMElement> {
        let read_guard = self.0.read_inner();
        read_guard
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Key not found"))
    }

    pub fn get_data_deep(&self, key: &str) -> anyhow::Result<IMElement> {
        let read_guard = self.0.read_inner();
        read_guard
            .get(key)
            .map(|element| element.deep_clone())
            .ok_or_else(|| anyhow::anyhow!("Key not found"))
    }
}
