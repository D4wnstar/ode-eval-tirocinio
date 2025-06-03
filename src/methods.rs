use std::rc::Rc;

use crate::solvers::ScalarField;

/// Common trait between all numerical integration methods. The `next` function calculates the next
/// step from the current step.
pub trait IntegrationMethod {
    /// Calculate the next step from the current step `x` given the derivative vector field `f`.
    fn next(&self, x: &[f64], f: &[Rc<ScalarField>], stepsize: f64) -> Vec<f64>;

    /// Rust does not support vector operations, so this function is a helper to run a function `map`
    /// over `n` components.
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

pub trait DoubleIntegrationMethod {
    fn next(&self, x: &[f64], f: &[Rc<ScalarField>], stepsize: f64) -> DoubleIntegrationStep;

    /// Rust does not support vector operations, so this function is a helper to run a function `map`
    /// over `n` components.
    fn vec(&self, n: usize, map: impl Fn(usize) -> f64) -> Vec<f64> {
        (0..n).map(|i| map(i)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct DoubleIntegrationStep {
    pub x_good: Vec<f64>,
    pub x_bad: Vec<f64>,
    pub delta: Vec<f64>,
}

#[derive(Default, Clone, Copy)]
pub struct DormandPrince54 {}

impl DormandPrince54 {
    // Butcher tableau coefficients (a_ij)
    const A: [[f64; 6]; 7] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
        [
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
        ],
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
        ],
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ];

    // 5th order weights (b_i)
    const B: [f64; 7] = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ];

    // 4th order weights (b*_i)
    const B_STAR: [f64; 7] = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ];

    // Nodes (c_i) if non-autonomous
    #[allow(unused)]
    const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
}

impl DoubleIntegrationMethod for DormandPrince54 {
    fn next(&self, x: &[f64], f: &[Rc<ScalarField>], stepsize: f64) -> DoubleIntegrationStep {
        let n = x.len();
        let a = &Self::A;
        let b = &Self::B;
        let b_star = &Self::B_STAR;

        // Calculate the common parameters
        let k1 = self.vec(n, |i| f[i](x) * stepsize);
        let trial1 = self.vec(n, |i| x[i] + a[1][0] * k1[i]);

        let k2 = self.vec(n, |i| f[i](&trial1) * stepsize);
        let trial2 = self.vec(n, |i| x[i] + a[2][0] * k1[i] + a[2][1] * k2[i]);

        let k3 = self.vec(n, |i| f[i](&trial2) * stepsize);
        let trial3 = self.vec(n, |i| {
            x[i] + a[3][0] * k1[i] + a[3][1] * k2[i] + a[3][2] * k3[i]
        });

        let k4 = self.vec(n, |i| f[i](&trial3) * stepsize);
        let trial4 = self.vec(n, |i| {
            x[i] + a[4][0] * k1[i] + a[4][1] * k2[i] + a[4][2] * k3[i] + a[4][3] * k4[i]
        });

        let k5 = self.vec(n, |i| f[i](&trial4) * stepsize);
        let trial5 = self.vec(n, |i| {
            x[i] + a[5][0] * k1[i]
                + a[5][1] * k2[i]
                + a[5][2] * k3[i]
                + a[5][3] * k4[i]
                + a[5][4] * k5[i]
        });

        let k6 = self.vec(n, |i| f[i](&trial5) * stepsize);
        let trial6 = self.vec(n, |i| {
            x[i] + a[6][0] * k1[i]
                + a[6][1] * k2[i]
                + a[6][2] * k3[i]
                + a[6][3] * k4[i]
                + a[6][4] * k5[i]
                + a[6][5] * k6[i]
        });

        let k7 = self.vec(n, |i| f[i](&trial6) * stepsize);

        // Run both RK formulas
        let x_order5 = self.vec(n, |i| {
            x[i] + b[0] * k1[i]
                + b[1] * k2[i]
                + b[2] * k3[i]
                + b[3] * k4[i]
                + b[4] * k5[i]
                + b[5] * k6[i]
                + b[6] * k7[i]
        });

        let x_order4 = self.vec(n, |i| {
            x[i] + b_star[0] * k1[i]
                + b_star[1] * k2[i]
                + b_star[2] * k3[i]
                + b_star[3] * k4[i]
                + b_star[4] * k5[i]
                + b_star[5] * k6[i]
                + b_star[6] * k7[i]
        });

        // Find the difference between the estimations
        let delta = self.vec(n, |i| x_order5[i] - x_order4[i]);

        return DoubleIntegrationStep {
            x_good: x_order5,
            x_bad: x_order4,
            delta,
        };
    }
}
