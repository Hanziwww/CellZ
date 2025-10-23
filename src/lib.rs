mod bitstream;
mod varint;
mod golomb_rice;
mod hybrid_value;
mod matrix;
mod compress;
mod decompress;

use pyo3::prelude::*;

#[pymodule]
fn _cellz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress::compress_csc_to_cellz, m)?)?;
    m.add_function(wrap_pyfunction!(decompress::decompress_cellz_to_csr, m)?)?;
    Ok(())
}
