//! Complex implementations of existing structs and methods. Kept in a separate file
//! since in the future the existing code will hopefully be made generic in numerical
//! type and all of this can be deprecated.

use std::rc::Rc;

use num::complex::Complex64;

/* UTILS */
/// A function from C^N to C.
pub type ScalarFieldComplex = dyn Fn(&[Complex64]) -> Complex64;

/* SOLVERS */
/// Like `OdeSystem`, but using `Complex64` instead.
#[derive(Clone)]
pub struct OdeSystemComplex {
    t_start: f64,
    x_start: Vec<Complex64>,
    derivatives: Vec<Rc<ScalarFieldComplex>>,
}

impl OdeSystemComplex {
    pub fn new(
        start: f64,
        initial_values: &[Complex64],
        derivatives: &[Rc<ScalarFieldComplex>],
    ) -> Self {
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

/// Like `OdeSolver`, but using `Complex64` instead.
pub struct OdeSolverComplex<M: IntegrationMethodComplex> {
    system: OdeSystemComplex,
    method: M,
    // TODO: Move this as an argument of `solve`
    stepsize: f64,
}

impl<M: IntegrationMethodComplex> OdeSolverComplex<M> {
    pub fn new(system: OdeSystemComplex, method: M, stepsize: f64) -> Self {
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
    pub fn solve(&self, t_end: f64) -> Vec<(f64, Vec<Complex64>)> {
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

/// Like `PdeSolver`, but using `Complex64`.
pub struct PdeSolverComplex<M: IntegrationMethodComplex + Clone> {
    ode_method: M,
    stepsize: f64,
    init_condition: Rc<dyn Fn(f64, usize) -> Complex64>,
    //                          state,      x_i  dx   i
    discretization: Rc<dyn Fn(&[Complex64], f64, f64, usize) -> Complex64>,
}

impl<M: IntegrationMethodComplex + Clone> PdeSolverComplex<M> {
    pub fn new(
        ode_method: M,
        stepsize: f64,
        init_condition: Rc<dyn Fn(f64, usize) -> Complex64>,
        discretization: Rc<dyn Fn(&[Complex64], f64, f64, usize) -> Complex64>,
    ) -> Self {
        PdeSolverComplex {
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
    ) -> PdeSolutionComplex {
        let mut init_conditions = Vec::with_capacity(grid_points);
        let mut odes: Vec<Rc<ScalarFieldComplex>> = Vec::with_capacity(grid_points);
        let dx = (x_end - x_start).abs() / (grid_points - 1) as f64;

        for i in 0..grid_points {
            // Calculate initial time conditions
            let x = x_start + dx * i as f64;
            init_conditions.push((self.init_condition)(x, i));

            // Define an array of ODEs, one for each spatial grid point
            let disc = self.discretization.clone();
            let ode = Rc::new(move |state: &[Complex64]| (disc)(state, x, dx, i));
            odes.push(ode);
        }

        // Solve the grid ODEs simultaneously
        let ode_system = OdeSystemComplex::new(t_start, &init_conditions, &odes);
        let points =
            OdeSolverComplex::new(ode_system, self.ode_method.clone(), self.stepsize).solve(t_end);

        return PdeSolutionComplex {
            points,
            x_start,
            dx,
            grid_points,
        };
    }
}

pub struct PdeSolutionComplex {
    pub points: Vec<(f64, Vec<Complex64>)>,
    x_start: f64,
    dx: f64,
    grid_points: usize,
}

impl PdeSolutionComplex {
    pub fn into_square_norms(&self) -> Vec<Vec<Vec<f64>>> {
        let mut data: Vec<Vec<Vec<f64>>> = vec![Vec::new(); self.grid_points];
        for (t, z) in &self.points {
            for i in 0..data.len() {
                data[i].push(vec![*t, self.x_start + self.dx * i as f64, z[i].norm_sqr()]);
            }
        }
        return data;
    }
}

/* METHODS */
/// Trait for all non-adaptive complex numerical integration methods.
pub trait IntegrationMethodComplex {
    /// Predict the next point from the current one `x`, using the functions `f` at a
    /// step `stepsize`.
    fn next(&self, x: &[Complex64], f: &[Rc<ScalarFieldComplex>], stepsize: f64) -> Vec<Complex64>;
}

/// Complex implementation of RK4.
#[derive(Default, Clone, Copy)]
pub struct RungeKutta4Complex {}

impl RungeKutta4Complex {
    /// Run a function `map` over each index from `0` to `n - 1`.
    fn vec(&self, n: usize, map: impl Fn(usize) -> Complex64) -> Vec<Complex64> {
        (0..n).map(|i| map(i)).collect()
    }
}

impl IntegrationMethodComplex for RungeKutta4Complex {
    fn next(&self, x: &[Complex64], f: &[Rc<ScalarFieldComplex>], stepsize: f64) -> Vec<Complex64> {
        let n = x.len();

        let k1 = self.vec(n, |i| f[i](x) * stepsize);
        let trial1 = self.vec(n, |i| x[i] + k1[i] / 2.0);

        let k2 = self.vec(n, |i| f[i](&trial1) * stepsize);
        let trial2 = self.vec(n, |i| x[i] + k2[i] / 2.0);

        let k3 = self.vec(n, |i| f[i](&trial2) * stepsize);
        let trial3 = self.vec(n, |i| x[i] + k3[i]);

        let k4 = self.vec(n, |i| f[i](&trial3) * stepsize);

        self.vec(n, |i| {
            x[i] + (1.0 / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        })
    }
}
