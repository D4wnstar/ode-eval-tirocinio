use crate::solvers::Tolerances;

pub trait StepsizeScheduler {
    fn error(&self, x_curr: &[f64], delta: &[f64], tol: &Tolerances) -> f64;
    fn accept(&self, error: f64) -> bool;
    fn next(&self, step: f64, error: f64) -> f64;
}

#[derive(Clone, Copy)]
pub struct DormandPrince54Scheduler {
    safety_factor: f64,
    num_of_odes: u32,
}

impl DormandPrince54Scheduler {
    pub fn new(safety_factor: f64, num_of_odes: u32) -> Self {
        Self {
            safety_factor,
            num_of_odes,
        }
    }
}

impl StepsizeScheduler for DormandPrince54Scheduler {
    fn error(&self, x_curr: &[f64], delta: &[f64], tol: &Tolerances) -> f64 {
        let x_norm = x_curr.iter().fold(0.0, |acc, x_i| acc + x_i.powi(2)).sqrt();
        let scale = tol.absolute + x_norm * tol.relative;
        let norm_sq = delta
            .iter()
            .fold(0.0, |acc, delta_i| acc + (delta_i / scale).powi(2));
        let one_over_n = 1.0 / self.num_of_odes as f64;
        // println!("> x_norm={x_norm}; scale={scale}; norm_sq={norm_sq}; one_over_n={one_over_n}");
        (one_over_n * norm_sq).sqrt()
    }

    fn accept(&self, error: f64) -> bool {
        error <= 1.0
    }

    fn next(&self, step: f64, error: f64) -> f64 {
        self.safety_factor * step / error.powf(0.2)
    }
}
