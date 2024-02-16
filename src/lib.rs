use numpy::ndarray::Array2;
use numpy::{IntoPyArray, Ix2, PyArray2};

use pyo3::exceptions::PyValueError;
use pyo3::pyclass;
use pyo3::{pymethods, pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn spiral_scan<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    m.add_class::<SpiralOrdering>()?;

    fn check_proper_size(n: usize, m: usize) -> PyResult<()> {
        if (u64::try_from(n)? * u64::try_from(m)?) > u64::try_from(i32::MAX)? {
            return Err(PyValueError::new_err("dimensions too large"));
        }
        Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "generate_2d_matrix", signature=(n, m, ordering=SpiralOrdering::TopLeftCW, reversed=false))]
    fn generate_2d_matrix<'py>(
        py: Python<'py>,
        n: usize,
        m: usize,
        ordering: SpiralOrdering,
        reversed: bool,
    ) -> PyResult<&'py PyArray2<i32>> {
        check_proper_size(n, m)?;

        let mut result = Array2::<i32>::uninit(Ix2(n, m));

        if !reversed {
            let mut k = 0i32;
            spiral(n, m, ordering, |i: usize, j: usize| {
                result[[i, j]].write(k);
                k += 1;
            });
        } else {
            let mut k = i32::try_from(n * m)?;
            spiral(n, m, ordering, |i: usize, j: usize| {
                k -= 1;
                result[[i, j]].write(k);
            });
        }

        unsafe { Ok(result.assume_init().into_pyarray(py)) }
    }

    #[pyfn(m)]
    #[pyo3(name = "generate_2d_indices", signature=(n, m, ordering=SpiralOrdering::TopLeftCW, reversed=false))]
    fn generate_2d_indices<'py>(
        py: Python<'py>,
        n: usize,
        m: usize,
        ordering: SpiralOrdering,
        reversed: bool,
    ) -> PyResult<&'py PyArray2<usize>> {
        check_proper_size(n, m)?;

        let mut result = Array2::<usize>::uninit(Ix2(n * m, 2));
        if !reversed {
            let mut k = 0usize;
            spiral(n, m, ordering, |i: usize, j: usize| {
                result[[k, 0]].write(i);
                result[[k, 1]].write(j);
                k += 1;
            });
        } else {
            let mut k = n * m;
            spiral(n, m, ordering, |i: usize, j: usize| {
                k -= 1;
                result[[k, 0]].write(i);
                result[[k, 1]].write(j);
            });
        }

        Ok((unsafe { result.assume_init() }).into_pyarray(py))
    }

    Ok(())
}

pub struct SpiralScan<Collect: FnMut(usize, usize)> {
    start_i: usize,
    stop_i: usize,
    start_j: usize,
    stop_j: usize,
    collect: Collect,
}

impl<Collect: FnMut(usize, usize)> SpiralScan<Collect> {
    pub fn new(n: usize, m: usize, collect: Collect) -> SpiralScan<Collect> {
        SpiralScan {
            start_i: 0,
            stop_i: n,
            start_j: 0,
            stop_j: m,
            collect,
        }
    }
    pub fn has_more(&self) -> bool {
        self.start_i < self.stop_i || self.start_j < self.stop_j
    }

    pub fn go_top_right(&mut self) {
        // →→→
        if self.start_i < self.stop_i {
            let i = self.start_i;
            for j in self.start_j..self.stop_j {
                (self.collect)(i, j)
            }
            self.start_i += 1;
        }
    }

    pub fn go_rightside_down(&mut self) {
        // ↓↓↓
        if self.start_j < self.stop_j {
            let j = self.stop_j - 1;
            for i in self.start_i..self.stop_i {
                (self.collect)(i, j);
            }
            self.stop_j -= 1;
        }
    }

    pub fn go_bottom_left(&mut self) {
        // ←←←
        if self.start_i < self.stop_i {
            let i = self.stop_i - 1;
            for j in (self.start_j..self.stop_j).rev() {
                (self.collect)(i, j);
            }
            self.stop_i -= 1;
        }
    }
    pub fn go_leftside_up(&mut self) {
        // ↑↑↑
        if self.start_j < self.stop_j {
            let j = self.start_j;
            for i in (self.start_i..self.stop_i).rev() {
                (self.collect)(i, j);
            }
            self.start_j += 1;
        }
    }

    // -----

    pub fn go_top_left(&mut self) {
        // ←←←
        if self.start_i < self.stop_i {
            let i = self.start_i;
            for j in (self.start_j..self.stop_j).rev() {
                (self.collect)(i, j)
            }
            self.start_i += 1;
        }
    }

    pub fn go_rightside_up(&mut self) {
        // ↑↑↑
        if self.start_j < self.stop_j {
            let j = self.stop_j - 1;
            for i in (self.start_i..self.stop_i).rev() {
                (self.collect)(i, j);
            }
            self.stop_j -= 1;
        }
    }

    pub fn go_bottom_right(&mut self) {
        // →→→
        if self.start_i < self.stop_i {
            let i = self.stop_i - 1;
            for j in self.start_j..self.stop_j {
                (self.collect)(i, j);
            }
            self.stop_i -= 1;
        }
    }

    pub fn go_leftside_down(&mut self) {
        // ↓↓↓
        if self.start_j < self.stop_j {
            let j = self.start_j;
            for i in self.start_i..self.stop_i {
                (self.collect)(i, j);
            }
            self.start_j += 1;
        }
    }
}

#[pyclass]
#[derive(Clone, Copy)]
enum SpiralOrdering {
    TopLeftCW,
    TopRightCW,
    BottomRightCW,
    BottomLeftCW,
    TopLeftCCW,
    BottomLeftCCW,
    BottomRightCCW,
    TopRightCCW,
}

#[pymethods]
#[rustfmt::skip]
impl SpiralOrdering {
    fn identity(&self) -> SpiralOrdering {
        match self {
            SpiralOrdering::TopLeftCW       => SpiralOrdering::TopLeftCW     ,
            SpiralOrdering::TopRightCW      => SpiralOrdering::TopRightCW    ,
            SpiralOrdering::BottomRightCW   => SpiralOrdering::BottomRightCW ,
            SpiralOrdering::BottomLeftCW    => SpiralOrdering::BottomLeftCW  ,
            SpiralOrdering::TopLeftCCW      => SpiralOrdering::TopLeftCCW    ,
            SpiralOrdering::BottomLeftCCW   => SpiralOrdering::BottomLeftCCW ,
            SpiralOrdering::BottomRightCCW  => SpiralOrdering::BottomRightCCW,
            SpiralOrdering::TopRightCCW     => SpiralOrdering::TopRightCCW   ,
        }
    }
    fn flip_rot(&self) -> SpiralOrdering {
        match self {
            SpiralOrdering::TopLeftCW       => SpiralOrdering::TopLeftCCW     ,
            SpiralOrdering::TopRightCW      => SpiralOrdering::TopRightCCW    ,
            SpiralOrdering::BottomRightCW   => SpiralOrdering::BottomRightCCW ,
            SpiralOrdering::BottomLeftCW    => SpiralOrdering::BottomLeftCCW  ,
            SpiralOrdering::TopLeftCCW      => SpiralOrdering::TopLeftCW    ,
            SpiralOrdering::BottomLeftCCW   => SpiralOrdering::BottomLeftCW ,
            SpiralOrdering::BottomRightCCW  => SpiralOrdering::BottomRightCW,
            SpiralOrdering::TopRightCCW     => SpiralOrdering::TopRightCW   ,
        }
    }
    fn rotate_cw(&self) -> SpiralOrdering {
        match self {
            SpiralOrdering::TopLeftCW       => SpiralOrdering::TopRightCW    ,
            SpiralOrdering::TopRightCW      => SpiralOrdering::BottomRightCW ,
            SpiralOrdering::BottomRightCW   => SpiralOrdering::BottomLeftCW  ,
            SpiralOrdering::BottomLeftCW    => SpiralOrdering::TopLeftCW     ,
            SpiralOrdering::TopLeftCCW      => SpiralOrdering::TopRightCCW   ,
            SpiralOrdering::BottomLeftCCW   => SpiralOrdering::TopLeftCCW    ,
            SpiralOrdering::BottomRightCCW  => SpiralOrdering::BottomLeftCCW ,
            SpiralOrdering::TopRightCCW     => SpiralOrdering::BottomRightCCW,
        }
    }
    fn rotate_ccw(&self) -> SpiralOrdering {
        match self {
            SpiralOrdering::TopLeftCW       => SpiralOrdering::BottomLeftCW  ,
            SpiralOrdering::TopRightCW      => SpiralOrdering::TopLeftCW     ,
            SpiralOrdering::BottomRightCW   => SpiralOrdering::TopRightCW    ,
            SpiralOrdering::BottomLeftCW    => SpiralOrdering::BottomRightCW ,
            SpiralOrdering::TopLeftCCW      => SpiralOrdering::BottomLeftCCW ,
            SpiralOrdering::BottomLeftCCW   => SpiralOrdering::BottomRightCCW,
            SpiralOrdering::BottomRightCCW  => SpiralOrdering::TopRightCCW   ,
            SpiralOrdering::TopRightCCW     => SpiralOrdering::TopLeftCCW    ,
        }
    }
    fn rotate_180(&self) -> SpiralOrdering {
        return self.rotate_cw().rotate_cw();
    }

    fn reverse(&self) -> SpiralOrdering {
        return self.rotate_180().flip_rot()
    }

}

/// Template function to generate spiral scan coordonates
fn spiral<F: FnMut(usize, usize)>(n: usize, m: usize, ordering: SpiralOrdering, collect: F) {
    let mut scan = SpiralScan::new(n, m, collect);

    match ordering {
        SpiralOrdering::TopLeftCW => {
            while scan.has_more() {
                scan.go_top_right();
                scan.go_rightside_down();
                scan.go_bottom_left();
                scan.go_leftside_up();
            }
        }
        SpiralOrdering::TopRightCW => {
            while scan.has_more() {
                scan.go_rightside_down();
                scan.go_bottom_left();
                scan.go_leftside_up();
                scan.go_top_right();
            }
        }
        SpiralOrdering::BottomRightCW => {
            while scan.has_more() {
                scan.go_bottom_left();
                scan.go_leftside_up();
                scan.go_top_right();
                scan.go_rightside_down();
            }
        }
        SpiralOrdering::BottomLeftCW => {
            while scan.has_more() {
                scan.go_leftside_up();
                scan.go_top_right();
                scan.go_rightside_down();
                scan.go_bottom_left();
            }
        }
        SpiralOrdering::TopLeftCCW => {
            while scan.has_more() {
                scan.go_leftside_down();
                scan.go_bottom_right();
                scan.go_rightside_up();
                scan.go_top_left();
            }
        }
        SpiralOrdering::BottomLeftCCW => {
            while scan.has_more() {
                scan.go_bottom_right();
                scan.go_rightside_up();
                scan.go_top_left();
                scan.go_leftside_down();
            }
        }
        SpiralOrdering::BottomRightCCW => {
            while scan.has_more() {
                scan.go_rightside_up();
                scan.go_top_left();
                scan.go_leftside_down();
                scan.go_bottom_right();
            }
        }
        SpiralOrdering::TopRightCCW => {
            while scan.has_more() {
                scan.go_top_left();
                scan.go_leftside_down();
                scan.go_bottom_right();
                scan.go_rightside_up();
            }
        }
    }
}
