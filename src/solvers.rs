// I am using dx/dt = f(x), where x=x(t)
// Numerical Recipes instead uses dy/dx = f(y), where y=y(x)

type Ode = dyn Fn(f64) -> f64;

pub struct System {
    ode: Box<Ode>,
    t_start: f64,
    x_start: f64,
}

impl System {
    pub fn new(ode: Box<Ode>, start: f64, initial_value: f64) -> Self {
        // initial_value = ode(start)
        // initial_value can be anything, but it must be the value of the ODE at start
        Self {
            ode,
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

    pub fn solve(self: &Self) -> Vec<Vec<f64>> {
        // Solve between start and end, with constant stepsize
        let mut t_curr = self.system.t_start;
        let mut x_curr = self.system.x_start;
        let mut output = vec![vec![t_curr, x_curr]];

        while t_curr < self.t_end {
            x_curr = self.method.next(x_curr, self.stepsize, &self.system.ode);
            t_curr += self.stepsize;
            output.push(vec![t_curr, x_curr]);
        }

        return output;
    }
}

pub trait IntegrationMethod {
    fn next(&self, curr_x: f64, stepsize: f64, ode: &Ode) -> f64;
}

#[derive(Default)]
pub struct Euler {}

impl IntegrationMethod for Euler {
    fn next(&self, curr_x: f64, stepsize: f64, ode: &Ode) -> f64 {
        curr_x + ode(curr_x) * stepsize
    }
}

// pub struct AdaptiveSolver<S> {
//     initial_values: Vec<f64>,
//     start: f64,
//     end: f64,
//     initial_stepsize: f64,
//     min_stepsize: f64,
//     stepper: S,
// }

// pub struct Stepper {}
