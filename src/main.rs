use examples::{
    exponential_decay, exponential_decay_adaptive, harmonic_oscillator, method_comparison,
};
use methods::{DormandPrince54, RungeKutta4};
use schedulers::DeltaScheduler;

use crate::examples::{
    elastic_pendulum_comparison, harmonic_oscillator_adaptive, harmonic_oscillator_interpolation,
    simple_pendulum_adaptive, simple_pendulum_against_small_swings, simple_pendulum_comparison,
};

pub mod examples;
pub mod methods;
pub mod schedulers;
pub mod solvers;
pub mod utils;

/// Main function to run all examples.
fn main() {
    exponential_decay(RungeKutta4::default());
    exponential_decay_adaptive(
        DormandPrince54::default(),
        DeltaScheduler::with_dimension(1),
    );
    method_comparison();
    harmonic_oscillator(1.0, 2.0, RungeKutta4::default());
    harmonic_oscillator_adaptive(
        1.0,
        2.0,
        DormandPrince54::default(),
        DeltaScheduler::with_dimension(2),
    );
    harmonic_oscillator_interpolation(
        1.0,
        2.0,
        DormandPrince54::default(),
        DeltaScheduler::with_dimension(2),
    );
    simple_pendulum_adaptive(
        1.0,
        9.8,
        2.0,
        DormandPrince54::default(),
        DeltaScheduler::with_dimension(2),
    );
    simple_pendulum_against_small_swings(
        1.0,
        9.8,
        1.0,
        DormandPrince54::default(),
        DeltaScheduler::with_dimension(2),
    );
    simple_pendulum_comparison(
        1.0,
        9.8,
        1.0,
        DormandPrince54::default(),
        DeltaScheduler::with_dimension(2),
    );
    elastic_pendulum_comparison(
        1.0,
        9.8,
        1.0,
        5.0,
        DormandPrince54::default(),
        DeltaScheduler::with_dimension(4),
    );
}
