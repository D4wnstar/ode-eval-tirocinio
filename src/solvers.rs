//! User-oriented framework for making use of the solvers and schedulers defined elsewhere.
//! The `System` object encapsulates the idea of a system of autonomous first-order ODEs.
//! The `Solver` structs handle the actual numerical integration methods with an easy to
//! use API.

// Notation: I am using dx/dt = f(x), where x=x(t). Numerical Recipes instead uses dy/dx = f(y), where y=y(x)

use std::rc::Rc;

use crate::{
    methods::{AdaptiveIntegrationMethod, IntegrationMethod},
    schedulers::StepsizeScheduler,
    utils::{ScalarField, Tolerances},
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
pub struct OdeSystem {
    t_start: f64,
    x_start: Vec<f64>,
    derivatives: Vec<Rc<ScalarField>>,
}

impl OdeSystem {
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
pub struct OdeSolver<M: IntegrationMethod> {
    system: OdeSystem,
    method: M,
    // TODO: Move this as an argument of `solve`
    stepsize: f64,
}

impl<M: IntegrationMethod> OdeSolver<M> {
    pub fn new(system: OdeSystem, method: M, stepsize: f64) -> Self {
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
pub struct OdeAdaptiveSolver<M, S>
where
    M: AdaptiveIntegrationMethod,
    S: StepsizeScheduler,
{
    system: OdeSystem,
    method: M,
    scheduler: S,
    tolerances: Tolerances,
    to_interpolate: Option<Vec<f64>>,
}

impl<M, S> OdeAdaptiveSolver<M, S>
where
    M: AdaptiveIntegrationMethod,
    S: StepsizeScheduler,
{
    const MAX_STEPS: u32 = 50_000;
}

impl<M, S> OdeAdaptiveSolver<M, S>
where
    M: AdaptiveIntegrationMethod,
    S: StepsizeScheduler,
{
    pub fn new(system: OdeSystem, method: M, scheduler: S, tolerances: Tolerances) -> Self {
        OdeAdaptiveSolver {
            system,
            method,
            scheduler,
            tolerances,
            to_interpolate: None,
        }
    }

    /// Also interpolate at the given points, as a vector of values of the independent variable.
    /// **The vector must be monotonically increasing, so sort it before use**.
    pub fn at_points(mut self, to_interpolate: Vec<f64>) -> Self {
        self.to_interpolate = Some(to_interpolate);
        self
    }

    /// Run the solver until `t_end`. It will return an `OdeSolution` object. The last evaluated point
    /// is guaranteed to be at `t_end`. If an array of values of the independent variable is passed
    /// with `.at_points()` before running this function, the solver will also interpolate the
    /// solution at those points.
    pub fn solve(&self, t_end: f64, starting_stepsize: f64) -> OdeSolution {
        let mut points = vec![(self.system.t_start, self.system.x_start.clone())];
        let mut interp_points = if self.to_interpolate.is_some() {
            Some(vec![])
        } else {
            None
        };
        let mut interp_index = 0;
        let mut accepted_steps = 0;
        let mut rejected_steps = 0;

        let mut t_next = self.system.t_start + starting_stepsize;
        let mut x_curr = self.system.x_start.clone();
        let mut stepsize = starting_stepsize;
        let mut step_count = 1;
        let mut end = false;

        while step_count < Self::MAX_STEPS {
            step_count += 1;

            // Make sure last step is exactly at t_end to avoid an empty interval at the end
            if t_next > t_end {
                stepsize -= (t_next - t_end).abs();
                t_next = t_end;
                end = true;
            }

            let result = self
                .method
                .next(&x_curr, &self.system.derivatives, stepsize);
            let error = self
                .scheduler
                .error(&x_curr, &result.delta, &self.tolerances);
            let accepted = self.scheduler.accept(error);

            if accepted {
                accepted_steps += 1;
                // If the step is accepted, update x and stepsize and save the result
                let x_next = result.x_good;
                points.push((t_next, x_next.clone()));

                // If the user requires specific points, interpolate the ones found in
                // the current step
                if let Some(to_interp) = &self.to_interpolate {
                    while interp_index <= to_interp.len() - 1
                        && to_interp[interp_index] >= t_next - stepsize
                        && to_interp[interp_index] <= t_next
                    {
                        let t_interp = to_interp[interp_index];
                        let x_interp = self.method.interpolate(
                            t_interp,
                            t_next - stepsize,
                            stepsize,
                            &result.interp_coeffs,
                        );
                        interp_points.as_mut().unwrap().push((t_interp, x_interp));
                        interp_index += 1;
                    }
                }

                stepsize = self.scheduler.next(stepsize, error);
                t_next += stepsize;
                x_curr = x_next;
            } else {
                rejected_steps += 1;
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

        return OdeSolution {
            points,
            interp_points,
            accepted_steps,
            rejected_steps,
        };
    }
}

pub struct PdeSolver<M: IntegrationMethod + Clone> {
    ode_method: M,
    stepsize: f64,
    init_condition: Rc<dyn Fn(f64) -> f64>,
    //                        state,  dx   i
    discretization: Rc<dyn Fn(&[f64], f64, usize) -> f64>,
}

impl<M: IntegrationMethod + Clone> PdeSolver<M> {
    pub fn new(
        ode_method: M,
        stepsize: f64,
        init_condition: Rc<dyn Fn(f64) -> f64>,
        discretization: Rc<dyn Fn(&[f64], f64, usize) -> f64>,
    ) -> Self {
        PdeSolver {
            ode_method,
            stepsize,
            init_condition,
            discretization,
        }
    }

    /// Solve the PDE numerically with the method of lines.
    pub fn solve(
        &self,
        t_start: f64,
        t_end: f64,
        x_start: f64,
        x_end: f64,
        grid_points: usize,
    ) -> Vec<(f64, Vec<f64>)> {
        let mut init_conditions = Vec::with_capacity(grid_points);
        let mut odes: Vec<Rc<ScalarField>> = Vec::with_capacity(grid_points);
        let dx = (x_end - x_start) / (grid_points - 1) as f64;

        for i in 0..grid_points {
            // Calculate initial time conditions
            let x = x_start + dx * i as f64;
            init_conditions.push((self.init_condition)(x));

            // Define an array of ODEs, one for each spatial grid point
            let disc = self.discretization.clone();
            let ode = Rc::new(move |state: &[f64]| (disc)(state, dx, i));
            odes.push(ode);
        }

        // Solve the grid ODEs simultaneously
        let ode_system = OdeSystem::new(t_start, &init_conditions, &odes);
        let points =
            OdeSolver::new(ode_system, self.ode_method.clone(), self.stepsize).solve(t_end);

        return points;
    }
}

/// The solution to an ODE problem, including optional interpolations and some metadata.
/// Each point, interpolated or not, is the tuple (t, x), where t is the value of the
/// independent variable and x is the state vector at that point.
#[derive(Debug, Clone)]
pub struct OdeSolution {
    pub points: Vec<(f64, Vec<f64>)>,
    pub interp_points: Option<Vec<(f64, Vec<f64>)>>,
    pub accepted_steps: u32,
    pub rejected_steps: u32,
}
