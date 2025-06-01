// I am using dx/dt = f(x), where x=x(t)
// Numerical Recipes instead uses dy/dx = f(y), where y=y(x)

use std::rc::Rc;

type Fn1d = dyn Fn(f64) -> f64;

#[derive(Clone)]
pub struct System {
    derivative: Rc<Fn1d>,
    t_start: f64,
    x_start: f64,
}

impl System {
    pub fn new(derivative: impl Fn(f64) -> f64 + 'static, start: f64, initial_value: f64) -> Self {
        // initial_value can be anything, but it must be the value of the ODE at start
        // initial_value = ode(start)
        Self {
            derivative: Rc::new(derivative),
            t_start: start,
            x_start: initial_value,
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

    pub fn solve(&self) -> Vec<Vec<f64>> {
        // Solve between start and end, with constant stepsize
        let mut t_curr = self.system.t_start;
        let mut x_curr = self.system.x_start;
        let mut output = vec![vec![t_curr, x_curr]];

        while t_curr < self.t_end {
            x_curr = self
                .method
                .next(x_curr, self.stepsize, &*self.system.derivative);
            t_curr += self.stepsize;
            output.push(vec![t_curr, x_curr]);
        }

        return output;
    }
}

pub trait IntegrationMethod {
    fn next(&self, curr_x: f64, stepsize: f64, derivative: &Fn1d) -> f64;
}

#[derive(Default, Clone, Copy)]
pub struct Euler {}

impl IntegrationMethod for Euler {
    fn next(&self, curr_x: f64, stepsize: f64, derivative: &Fn1d) -> f64 {
        curr_x + derivative(curr_x) * stepsize
    }
}

#[derive(Default, Clone, Copy)]
pub struct Midpoint {}

impl IntegrationMethod for Midpoint {
    fn next(&self, curr_x: f64, stepsize: f64, derivative: &Fn1d) -> f64 {
        let trial_x = curr_x + derivative(curr_x) * stepsize;
        curr_x + 0.5 * (derivative(curr_x) + derivative(trial_x)) * stepsize
    }
}

#[derive(Default, Clone, Copy)]
pub struct RungeKutta4 {}

impl IntegrationMethod for RungeKutta4 {
    fn next(&self, curr_x: f64, stepsize: f64, derivative: &Fn1d) -> f64 {
        let k1 = derivative(curr_x) * stepsize;
        let k2 = derivative(curr_x + 0.5 * k1) * stepsize;
        let k3 = derivative(curr_x + 0.5 * k2) * stepsize;
        let k4 = derivative(curr_x + k3) * stepsize;

        curr_x + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    }
}
