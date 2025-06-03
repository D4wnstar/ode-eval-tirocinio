use examples::{
    exponential_decay, exponential_decay_adaptive, harmonic_oscillator, method_comparison,
};
use methods::{DormandPrince54, RungeKutta4};
use schedulers::DeltaScheduler;

use crate::examples::harmonic_oscillator_adaptive;

pub mod examples;
pub mod methods;
pub mod schedulers;
pub mod solvers;
pub mod utils;

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
        DeltaScheduler::with_dimension(1),
    );
}
