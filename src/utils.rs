//! Utility code that does not quite belong to any other module in particular.

use num::{Num, complex::Complex64};

/// Helper trait for numerical operations needed by solvers
pub trait SolverNum: Copy + From<f64> + Num {}

impl<T> SolverNum for T where T: Copy + From<f64> + Num {}

/// A function from R^N to R or C^N to C.
pub type ScalarField<T> = dyn Fn(&[T]) -> T;

/// Trait for vector norm operations.
pub trait VectorNorm<T> {
    /// Calculate the square norm of the vector.
    fn norm_sq(vec: &[T]) -> f64;

    /// Calculate the norm of the vector.
    fn norm(vec: &[T]) -> f64 {
        Self::norm_sq(vec).sqrt()
    }
}

impl VectorNorm<f64> for f64 {
    fn norm_sq(vec: &[f64]) -> f64 {
        vec.iter().fold(0.0, |acc, &num| acc + num * num)
    }
}

impl VectorNorm<Complex64> for Complex64 {
    fn norm_sq(vec: &[Complex64]) -> f64 {
        vec.iter().fold(0.0, |acc, num| acc + num.norm_sqr())
    }
}

/// Trait for vector operations.
pub trait VecOperations<T> {
    /// Create a vector by mapping over indices.
    fn vec(&self, n: usize, map: impl Fn(usize) -> T) -> Vec<T> {
        (0..n).map(map).collect()
    }
}

/// Tolerances for solvers.
#[derive(Debug, Clone, Copy)]
pub struct Tolerances<T> {
    pub absolute: T,
    pub relative: T,
}

impl<T> Tolerances<T> {
    pub fn new(atol: T, rtol: T) -> Self {
        Self {
            absolute: atol,
            relative: rtol,
        }
    }
}
