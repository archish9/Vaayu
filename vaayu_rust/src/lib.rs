use memmap2::MmapOptions;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use safetensors::{tensor::Dtype, SafeTensors};
use std::fs::File;

fn dtype_str(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::BOOL => "BOOL",
        Dtype::U8   => "U8",
        Dtype::I8   => "I8",
        Dtype::I16  => "I16",
        Dtype::U16  => "U16",
        Dtype::F16  => "F16",
        Dtype::BF16 => "BF16",
        Dtype::I32  => "I32",
        Dtype::U32  => "U32",
        Dtype::F32  => "F32",
        Dtype::F64  => "F64",
        Dtype::I64  => "I64",
        Dtype::U64  => "U64",
        _           => "F32",
    }
}

/// Load a .safetensors file via mmap. GIL released during I/O.
///
/// Returns `Dict[str, Tuple[np.ndarray, str, List[int]]]`:
///   - ndarray : raw uint8 bytes of the tensor (one copy from mmap)
///   - str     : dtype name ("F16", "BF16", "F32", ...)
///   - List[int]: shape
#[pyfunction]
fn fast_load(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let path_owned = path.to_owned();

    // Release the GIL for the entire I/O + parse phase.
    let tensors: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> =
        py.allow_threads(|| -> Result<_, String> {
            let file = File::open(&path_owned)
                .map_err(|e| format!("open {}: {}", path_owned, e))?;

            // MAP_POPULATE tells the kernel to pre-fault all pages immediately,
            // saturating NVMe readahead while Rust parses the header.
            let mmap = unsafe {
                MmapOptions::new()
                    .populate()
                    .map(&file)
                    .map_err(|e| format!("mmap: {}", e))?
            };

            let st = SafeTensors::deserialize(&mmap)
                .map_err(|e| format!("parse: {}", e))?;

            let mut out = Vec::with_capacity(st.len());
            for (name, view) in st.tensors() {
                out.push((
                    name.to_string(),
                    dtype_str(view.dtype()),
                    view.shape().to_vec(),
                    view.data().to_vec(), // one copy: mmap region → owned Vec
                ));
            }
            Ok(out)
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let result = PyDict::new_bound(py);
    for (name, dtype, shape, data) in tensors {
        let arr = PyArray1::from_vec_bound(py, data);
        let shape_list = PyList::new_bound(py, shape.iter());
        let entry = PyTuple::new_bound(
            py,
            &[
                arr.into_py(py),
                dtype.into_py(py),
                shape_list.into_py(py),
            ],
        );
        result.set_item(&name, entry)?;
    }
    Ok(result.into())
}

#[pymodule]
fn vaayu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_load, m)?)?;
    Ok(())
}
