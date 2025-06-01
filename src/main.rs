use examples::{exponential_decay, harmonic_oscillator, method_comparison};
use solvers::RungeKutta4;

pub mod examples;
pub mod solvers;

fn main() {
    exponential_decay(RungeKutta4::default());
    method_comparison();
    harmonic_oscillator(1.0, 2.0, RungeKutta4::default());
}
