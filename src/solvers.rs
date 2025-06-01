// I am using dx/dt = f(x), where x=x(t)
// Numerical Recipes instead uses dy/dx = f(y), where y=y(x)

use std::rc::Rc;

type ScalarField = dyn Fn(&[f64]) -> f64;

#[derive(Clone)]
pub struct System {
    derivatives: Vec<Rc<ScalarField>>,
    t_start: f64,
    x_start: Vec<f64>,
}

impl System {
    pub fn new(start: f64, initial_values: &[f64], derivatives: &[Rc<ScalarField>]) -> Self {
        if initial_values.len() != derivatives.len() {
            panic!(
                "The number of derivative functions must be the same as the number of initial conditions"
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

pub struct Solver<M: IntegrationMethod> {
    system: System,
    t_end: f64,
    stepsize: f64,
    method: M,
}

impl<M: IntegrationMethod> Solver<M> {
    pub fn new(system: System, end: f64, stepsize: f64, method: M) -> Self {
        Self {
            system,
            t_end: end,
            stepsize,
            method,
        }
    }

    pub fn solve(&self) -> Vec<(f64, Vec<f64>)> {
        // Solve between start and end, with constant stepsize
        let mut output = vec![(self.system.t_start, self.system.x_start.clone())];

        let mut t_curr = self.system.t_start;
        let mut x_curr = self.system.x_start.clone();
        // Loop until close enough to endpoint. Remove a stepsize-derived amount to avoid overshooting
        while t_curr < self.t_end - self.stepsize / 10.0 {
            t_curr += self.stepsize;
            x_curr = self
                .method
                .next(&x_curr, &self.system.derivatives, self.stepsize);
            output.push((t_curr, x_curr.clone()));
        }

        return output;
    }
}

pub trait IntegrationMethod {
    fn next(&self, x: &[f64], f: &[Rc<ScalarField>], stepsize: f64) -> Vec<f64>;

    fn vec(&self, n: usize, map: impl Fn(usize) -> f64) -> Vec<f64> {
        (0..n).map(|i| map(i)).collect()
    }
}

#[derive(Default, Clone, Copy)]
pub struct Euler {}

impl IntegrationMethod for Euler {
    fn next(&self, x: &[f64], f: &[Rc<ScalarField>], stepsize: f64) -> Vec<f64> {
        self.vec(x.len(), |i| x[i] + f[i](x) * stepsize)
    }
}

#[derive(Default, Clone, Copy)]
pub struct Midpoint {}

impl IntegrationMethod for Midpoint {
    fn next(&self, x: &[f64], f: &[Rc<ScalarField>], stepsize: f64) -> Vec<f64> {
        let trial = self.vec(x.len(), |i| x[i] + f[i](x) * stepsize);

        self.vec(x.len(), |i| {
            x[i] + 0.5 * (f[i](x) + f[i](&trial)) * stepsize
        })
    }
}

#[derive(Default, Clone, Copy)]
pub struct RungeKutta4 {}

impl IntegrationMethod for RungeKutta4 {
    fn next(&self, x: &[f64], f: &[Rc<ScalarField>], stepsize: f64) -> Vec<f64> {
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
