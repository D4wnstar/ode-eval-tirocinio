// I am using dx/dt = f(x), where x=x(t)
// Numerical Recipes instead uses dy/dx = f(y), where y=y(x)

use std::rc::Rc;

use crate::{
    methods::{AdaptiveIntegrationMethod, IntegrationMethod},
    schedulers::StepsizeScheduler,
    utils::ScalarField,
};

/// An autonomous system of first-order ODEs, alongside their starting conditions
/// and the starting independent variable.
///
/// If the system has N equations, it is represented by `ẋ = f(x)`, where `x=x(t)` and `ẋ=ẋ(t)` are
/// N dimensional vector and `f(x)` is an R^N -> R^N vector field. Then
/// - `t_start` is the (scalar) value of `t` at the start of integration;
/// - `x_start` is the (N-dimensional vector) value of `x` at `t_start`, i.e. `x(t_start) = x_start`;
/// - `derivatives` is the N-dimensional vector of all `f_i(x)` components of `f(x)`, each being an
///    R^N -> R scalar field.
#[derive(Clone)]
pub struct System {
    t_start: f64,
    x_start: Vec<f64>,
    derivatives: Vec<Rc<ScalarField>>,
}

impl System {
    pub fn new(start: f64, initial_values: &[f64], derivatives: &[Rc<ScalarField>]) -> Self {
        if initial_values.len() != derivatives.len() {
            panic!(
                "The number of derivative functions must be the same as the number of initial values"
            )
        }

        // initial_value can be anything, but it must be the value of the derivative at start
        // initial_value = derivative(start)
        Self {
            derivatives: derivatives.iter().cloned().collect(),
            t_start: start,
            x_start: initial_values.into_iter().cloned().collect(),
        }
    }
}

/// A simple ODE solver. It will evaluate the `system` with a static `stepsize`
/// using any given `IntegrationMethod`.
pub struct Solver<M: IntegrationMethod> {
    system: System,
    method: M,
    stepsize: f64,
}

impl<M: IntegrationMethod> Solver<M> {
    pub fn new(system: System, method: M, stepsize: f64) -> Self {
        Self {
            system,
            stepsize,
            method,
        }
    }

    /// Run the solver until `t_end`. It will return an array of evaluated points. Each point is a tuple
    /// `(t, x)`, where `t` is a scalar and `x` is an N dimensional vector.
    ///
    /// Since the stepsize is not adaptive, the last evaluated point `t_last` is not exactly on `t_end`,
    /// but rather `t_last < t_end`.
    pub fn solve(&self, t_end: f64) -> Vec<(f64, Vec<f64>)> {
        // Solve between start and end, with constant stepsize
        let mut output = vec![(self.system.t_start, self.system.x_start.clone())];
        let mut t_curr = self.system.t_start;
        let mut x_curr = self.system.x_start.clone();

        // Loop until close enough to endpoint. Remove a stepsize-derived amount to avoid overshooting
        // due to floating point imprecision
        while t_curr < t_end - self.stepsize / 10.0 {
            t_curr += self.stepsize;
            x_curr = self
                .method
                .next(&x_curr, &self.system.derivatives, self.stepsize);
            output.push((t_curr, x_curr.clone()));
        }

        return output;
    }
}

/// An ODE solver with adaptive stepsize.  It will evaluate the `system` with a variable stepsize
/// using any given `AdaptiveIntegrationMethod` and `StepsizeScheduler`.
pub struct AdaptiveSolver<M, S>
where
    M: AdaptiveIntegrationMethod,
    S: StepsizeScheduler,
{
    system: System,
    method: M,
    scheduler: S,
    tolerances: Tolerances,
}

impl<M, S> AdaptiveSolver<M, S>
where
    M: AdaptiveIntegrationMethod,
    S: StepsizeScheduler,
{
    const MAX_STEPS: u32 = 50_000;
}

impl<M, S> AdaptiveSolver<M, S>
where
    M: AdaptiveIntegrationMethod,
    S: StepsizeScheduler,
{
    pub fn new(system: System, method: M, scheduler: S, tolerances: Tolerances) -> Self {
        AdaptiveSolver {
            system,
            method,
            scheduler,
            tolerances,
        }
    }

    pub fn solve(&self, t_end: f64, starting_stepsize: f64) -> Vec<(f64, Vec<f64>)> {
        let mut output = vec![(self.system.t_start, self.system.x_start.clone())];
        let mut t_curr = self.system.t_start + starting_stepsize;
        let mut x_curr = self.system.x_start.clone();
        let mut stepsize = starting_stepsize;
        let mut step_count = 1;
        let mut end = false;

        while step_count < Self::MAX_STEPS {
            // Make sure last step is exactly at t_end to avoid an empty interval at the end
            if t_curr > t_end {
                stepsize -= (t_curr - t_end).abs();
                t_curr = t_end;
                end = true;
            }

            step_count += 1;
            let result = self
                .method
                .next(&x_curr, &self.system.derivatives, stepsize);

            let error = self
                .scheduler
                .error(&x_curr, &result.delta, &self.tolerances);
            let accepted = self.scheduler.accept(error);

            if accepted {
                // If the step is accepted, update x and stepsize and save the result
                x_curr = result.x_good.clone();
                output.push((t_curr, x_curr.clone()));

                stepsize = self.scheduler.next(stepsize, error);
                t_curr += stepsize;
            } else {
                // If the step is rejected, update the stepsize but do not advance the
                // step counter. This forces the current step to be redone until it is
                // within accepted error margins
                let new_stepsize = self.scheduler.next(stepsize, error);

                // Stepsize must only ever decrease on reject
                stepsize = new_stepsize.min(stepsize);
            }

            if end {
                break;
            }
        }

        return output;
    }
}

#[derive(Clone, Copy)]
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
