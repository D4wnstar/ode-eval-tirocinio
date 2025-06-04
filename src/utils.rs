/// Rust does not support vector operations, so this trait implements a few helpers to make
/// vector algebra easier.
pub trait VecOperations {
    /// Run a function `map` over each index from `0` to `n - 1`.
    fn vec(&self, n: usize, map: impl Fn(usize) -> f64) -> Vec<f64> {
        (0..n).map(|i| map(i)).collect()
    }

    /// Calculate the square euclidean norm of the vector.
    fn norm_sq(&self, vec: Vec<f64>) -> f64 {
        vec.iter().fold(0.0, |acc, num| acc + num.powi(2))
    }

    /// Calculate the euclidean norm of the vector.
    fn norm(&self, vec: Vec<f64>) -> f64 {
        self.norm_sq(vec).sqrt()
    }
}

/// A function from R^N to R.
pub type ScalarField = dyn Fn(&[f64]) -> f64;

/// Tolerances for solvers.
#[derive(Debug, Clone, Copy)]
pub struct Tolerances {
    pub absolute: f64,
    pub relative: f64,
}

impl Tolerances {
    pub fn new(atol: f64, rtol: f64) -> Self {
        Self {
            absolute: atol,
            relative: rtol,
        }
    }
}
