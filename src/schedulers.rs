//! Methods to adapt stepsize dynamically during integration based on some metric, typically
//! some form of "error" or difference to some desired metric.
//!
//! The name "scheduler" is inspired by "learning rate schedulers" in machine learning,
//! which largely do the same thing (adapt learning rate dynamically based on some metric
//! during training).

use crate::utils::{SolverNum, Tolerances, VecOperations, VectorNorm};

/// Trait for stepsize schedulers. A scheduler must be able to convert an error estimate
/// into a more tractable form, accept or reject a step and estimate the size of the next
/// step.
pub trait StepsizeScheduler<T: SolverNum> {
    /// Calculate the error at the current location `x_curr` using some difference metric
    /// `delta`, weighed by the given tolerances `tol`.
    fn error(&self, x_curr: &[T], delta: &[T], tol: &Tolerances<f64>) -> f64;

    /// Boolean function to determine whether the step should be accepted or not based on
    /// the value of the error calculated by `.error()`.
    fn accept(&self, error: f64) -> bool;

    /// Calculate the next stepsize based on the current one `step` and the current error
    /// `error`.
    fn next(&self, step: f64, error: f64) -> f64;
}

/// A stepsize scheduler that uses the difference between two evaluations (delta) to
/// compute a measure of error and then adapt stepsize accordingly.
#[derive(Clone, Copy)]
pub struct DeltaScheduler {
    one_over_dim: f64,
    safety_factor: f64,
    min_scaling: f64,
    max_scaling: f64,
}

impl DeltaScheduler {
    /// Create a new scheduler.
    pub fn new(num_of_odes: u32, safety_factor: f64, min_scaling: f64, max_scaling: f64) -> Self {
        Self {
            one_over_dim: 1.0 / num_of_odes as f64,
            safety_factor,
            min_scaling,
            max_scaling,
        }
    }

    /// Create a scheduler with default values for a system of the given dimension.
    pub fn with_dimension(num_of_odes: u32) -> Self {
        Self {
            one_over_dim: 1.0 / num_of_odes as f64,
            safety_factor: 0.9,
            min_scaling: 0.2,
            max_scaling: 5.0,
        }
    }
}

// Implement VecOperations for DeltaScheduler with generic T
impl<T: SolverNum> VecOperations<T> for DeltaScheduler {}

impl<T: SolverNum> StepsizeScheduler<T> for DeltaScheduler {
    fn error(&self, x_curr: &[T], delta: &[T], tol: &Tolerances<f64>) -> f64 {
        let delta_over_scale = self.vec(x_curr.len(), |i| {
            let scale_i = tol.absolute + VectorNorm::<T>::norm(&[x_curr[i]]) * tol.relative;
            VectorNorm::<T>::norm(&[delta[i]]) / scale_i
        });
        (self.one_over_dim * VectorNorm::<f64>::norm_sq(&delta_over_scale)).sqrt()
    }

    fn accept(&self, error: f64) -> bool {
        error <= 1.0
    }

    fn next(&self, step: f64, error: f64) -> f64 {
        let scale;
        if error == 0.0 {
            scale = self.max_scaling;
        } else {
            let maybe_scale = self.safety_factor / error.powf(0.2);
            if maybe_scale > self.max_scaling {
                scale = self.max_scaling;
            } else if maybe_scale < self.min_scaling {
                scale = self.min_scaling;
            } else {
                scale = maybe_scale;
            }
        }

        return step * scale;
    }
}
