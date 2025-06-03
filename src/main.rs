use examples::{
    exponential_decay, exponential_decay_adaptive, harmonic_oscillator, method_comparison,
};
use methods::{DormandPrince54, RungeKutta4};
use schedulers::DormandPrince54Scheduler;

pub mod examples;
pub mod methods;
pub mod schedulers;
pub mod solvers;

fn main() {
    exponential_decay(RungeKutta4::default());
    exponential_decay_adaptive(
        DormandPrince54::default(),
        DormandPrince54Scheduler::new(0.98, 1),
    );
    method_comparison();
    harmonic_oscillator(1.0, 2.0, RungeKutta4::default());
}
