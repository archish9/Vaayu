use io_uring::{opcode, types, IoUring};
use pyo3::prelude::*;
use std::fs::File;
use std::os::unix::io::AsRawFd;

/// Read a file region directly into a caller-provided buffer using io_uring.
///
/// # Arguments
/// * `path`        - Path to the .safetensors file
/// * `offset`      - Byte offset into the file where tensor data starts (after header)
/// * `length`      - Total bytes to read
/// * `buf_ptr`     - Raw pointer to the destination buffer (Python's pinned memory)
/// * `queue_depth` - Number of parallel io_uring SQEs (e.g. 32)
///
/// # Safety
/// The caller must ensure `buf_ptr` points to at least `length` bytes of valid,
/// writable memory that remains alive for the duration of this call.
#[pyfunction]
fn io_uring_read_into(
    py: Python<'_>,
    path: &str,
    offset: u64,
    length: usize,
    buf_ptr: usize,
    queue_depth: u32,
) -> PyResult<()> {
    let path = path.to_owned();

    py.allow_threads(move || {
        let file = File::open(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(
                format!("open {}: {}", path, e)))?;

        eprintln!("[vaayu_rust] open buffered  path={}  offset={}  length={}", path, offset, length);  // → O_DIRECT after impl

        let fd = types::Fd(file.as_raw_fd());
        let ptr = buf_ptr as *mut u8;
        let qd = queue_depth.min(64) as usize;
        let chunk_size = (length + qd - 1) / qd;
        let chunk_size = (chunk_size + 4095) & !4095;
        let total_chunks = (length + chunk_size - 1) / chunk_size;
        eprintln!("[vaayu_rust] qd={}  chunk_size={}  total_chunks={}  total_bytes={}", qd, chunk_size, total_chunks, length);

        let mut ring = IoUring::new(qd as u32)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("io_uring_setup: {}", e)))?;

        // Submit all read SQEs
        let mut submitted = 0usize;
        let mut file_off = offset;
        let mut buf_off = 0usize;

        while buf_off < length {
            let this_len = chunk_size.min(length - buf_off);
            let read_op = opcode::Read::new(fd, unsafe { ptr.add(buf_off) }, this_len as u32)
                .offset(file_off)
                .build()
                .user_data(submitted as u64);

            unsafe { ring.submission().push(&read_op) }
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SQ full"))?;

            file_off += this_len as u64;
            buf_off += this_len;
            submitted += 1;
        }

        eprintln!("[vaayu_rust] submitted {} SQEs", submitted);

        ring.submit_and_wait(submitted)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("io_uring submit: {}", e)))?;

        let cq = ring.completion();
        let mut completed = 0usize;
        for cqe in cq {
            if cqe.result() < 0 {
                eprintln!("[vaayu_rust] CQE error  idx={}  result={}  errno={}",
                          cqe.user_data(), cqe.result(), -cqe.result());
                return Err(pyo3::exceptions::PyIOError::new_err(
                    format!("io_uring read error  CQE[{}]  result={}  errno={}",
                            cqe.user_data(), cqe.result(), -cqe.result())));
            }
            completed += 1;
        }

        eprintln!("[vaayu_rust] done {}/{} CQEs ok", completed, submitted);

        if completed != submitted {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("incomplete: {}/{} CQEs", completed, submitted)));
        }

        Ok(())
    })
}

/// Fallback: single pread without io_uring (for testing/compatibility)
#[pyfunction]
fn pread_into(
    py: Python<'_>,
    path: &str,
    offset: u64,
    length: usize,
    buf_ptr: usize,
) -> PyResult<()> {
    let path = path.to_owned();
    py.allow_threads(move || {
        let file = std::fs::File::open(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        let ptr = buf_ptr as *mut u8;
        let buf = unsafe { std::slice::from_raw_parts_mut(ptr, length) };
        let n = unsafe {
            libc::pread(file.as_raw_fd(), buf.as_mut_ptr() as *mut _, length, offset as i64)
        };
        if n < 0 {
            return Err(pyo3::exceptions::PyIOError::new_err("pread failed"));
        }
        Ok(())
    })
}

#[pymodule]
fn vaayu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(io_uring_read_into, m)?)?;
    m.add_function(wrap_pyfunction!(pread_into, m)?)?;
    Ok(())
}
