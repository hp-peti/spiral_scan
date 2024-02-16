use numpy::ndarray::Array2;
use numpy::{IntoPyArray, Ix2, PyArray2};

use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn spiral_scan<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    //#[pyo3(name = "generate_2d_matrix")]
    fn generate_2d_matrix<'py>(py: Python<'py>, n: usize, m: usize) -> &'py PyArray2<i64> {
        let mut result = Array2::<i64>::zeros(Ix2(n, m));
        let mut k = 0;

        spiral(n, m, |i: usize, j: usize| {
            result[[i, j]] = k;
            k += 1;
        });

        result.into_pyarray(py)
    }

    #[pyfn(m)]
    //#[pyo3(name = "generate_2d_indices")]
    fn generate_2d_indices<'py>(py: Python<'py>, n: usize, m: usize) -> &'py PyArray2<usize> {
        let mut result = Array2::<usize>::zeros(Ix2(n * m, 2));
        let mut k = 0usize;
        spiral(n, m, |i: usize, j: usize| {
            result[[k, 0]] = i;
            result[[k, 1]] = j;
            k += 1;
        });

        result.into_pyarray(py)
    }

    Ok(())
}

/// Template function to generate spiral scan coordonates
fn spiral<F: FnMut(usize, usize)>(n: usize, m: usize, mut collect: F) {
    let (mut start_i, mut stop_i) = (0usize, n);
    let (mut start_j, mut stop_j) = (0usize, m);

    while start_i < stop_i || start_j < stop_j {
        // →→→
        if start_i < stop_i {
            let i = start_i;
            for j in start_j..stop_j {
                collect(i, j)
            }
            start_i += 1;
        }
        // ↓↓↓
        if start_j < stop_j {
            let j = stop_j - 1;
            for i in start_i..stop_i {
                collect(i, j);
            }
            stop_j -= 1;
        }
        // ←←←
        if start_i < stop_i {
            let i = stop_i - 1;
            for j in start_j..stop_j {
                collect(i, j);
            }
            stop_i -= 1;
        }
        // ↑↑↑
        if start_j < stop_j {
            let j = start_j;
            for i in (start_i..stop_i).rev() {
                collect(i, j);
            }
            start_j += 1;
        }
    }
}
