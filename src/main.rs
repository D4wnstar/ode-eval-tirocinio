use examples::{exponential_decay, method_comparison};
use solvers::RungeKutta4;

pub mod examples;
pub mod solvers;

fn main() {
    exponential_decay(RungeKutta4::default());
    method_comparison();
}
