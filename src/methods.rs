//! Numerical integration methods, both with static stepsize (Euler, midpoint, RK4) and
//! adaptive stepsize (Dormand-Prince 5(4)). Also includes built-in interpolation if the
//! method supports it.

use std::rc::Rc;

use crate::utils::{ScalarField, SolverNum, VecOperations};

/// Trait for all non-adaptive numerical integration methods.
pub trait IntegrationMethod<T: SolverNum> {
    /// Predict the next point from the current one `x`, using the functions `f` at a
    /// step `stepsize`.
    fn next(&self, x: &[T], f: &[Rc<ScalarField<T>>], stepsize: T) -> Vec<T>;
}

/// The traditional Euler method.
#[derive(Default, Clone, Copy)]
pub struct Euler {}

impl<T: SolverNum> VecOperations<T> for Euler {}

impl<T: SolverNum> IntegrationMethod<T> for Euler {
    fn next(&self, x: &[T], f: &[Rc<ScalarField<T>>], stepsize: T) -> Vec<T> {
        self.vec(x.len(), |i| x[i] + f[i](x) * stepsize)
    }
}

/// The improved Euler method, using the average of the derivatives at the start
/// and at the point guessed by the usual Euler method. Technically a second-order
/// Runge-Kutta method.
#[derive(Default, Clone, Copy)]
pub struct Midpoint {}

impl<T: SolverNum> VecOperations<T> for Midpoint {}

impl<T: SolverNum> IntegrationMethod<T> for Midpoint {
    fn next(&self, x: &[T], f: &[Rc<ScalarField<T>>], stepsize: T) -> Vec<T> {
        let trial = self.vec(x.len(), |i| x[i] + f[i](x) * stepsize);

        self.vec(x.len(), |i| {
            x[i] + 0.5.into() * (f[i](x) + f[i](&trial)) * stepsize
        })
    }
}

/// The classic fourth-order Runge-Kutta method (RK4).
#[derive(Default, Clone, Copy)]
pub struct RungeKutta4 {}

impl<T: SolverNum> VecOperations<T> for RungeKutta4 {}

impl<T: SolverNum> IntegrationMethod<T> for RungeKutta4 {
    fn next(&self, x: &[T], f: &[Rc<ScalarField<T>>], stepsize: T) -> Vec<T> {
        let n = x.len();

        let k1 = self.vec(n, |i| f[i](x) * stepsize);
        let trial1 = self.vec(n, |i| x[i] + k1[i] / 2.0.into());

        let k2 = self.vec(n, |i| f[i](&trial1) * stepsize);
        let trial2 = self.vec(n, |i| x[i] + k2[i] / 2.0.into());

        let k3 = self.vec(n, |i| f[i](&trial2) * stepsize);
        let trial3 = self.vec(n, |i| x[i] + k3[i]);

        let k4 = self.vec(n, |i| f[i](&trial3) * stepsize);

        self.vec(n, |i| {
            x[i] + (k1[i] + 2.0.into() * k2[i] + 2.0.into() * k3[i] + k4[i]) / 6.0.into()
        })
    }
}

/// Trait for integration methods that return both an estimate for the next point and
/// an estimate of the level of imprecision of the point. Meant to be used alongside
/// an adaptive `StepsizeScheduler`.
pub trait AdaptiveIntegrationMethod<T: SolverNum> {
    /// Predict the next point from the current one `x`, using the functions `f` at a
    /// step `stepsize`.
    fn next(&self, x: &[T], f: &[Rc<ScalarField<T>>], stepsize: T) -> AdaptiveIntegrationStep<T>;

    /// Interpolate the solution at `t`, which is expected to be between `t_curr` and `t_curr + stepsize`.
    fn interpolate(&self, t: T, t_curr: T, stepsize: T, coeffs: &[Vec<T>]) -> Vec<T>;
}

#[derive(Debug, Clone)]
pub struct AdaptiveIntegrationStep<T> {
    pub x_good: Vec<T>,
    pub x_bad: Vec<T>,
    pub delta: Vec<T>,
    pub interp_coeffs: Vec<Vec<T>>,
}

/// The Dormand-Prince 5(4) embedded Runge-Kutta method. Supports interpolation.
#[derive(Default, Clone, Copy)]
pub struct DormandPrince54 {}

impl DormandPrince54 {
    /// Butcher tableau coefficients (a_ij)
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

    /// 5th order weights (b_i)
    const B: [f64; 7] = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ];

    /// 4th order weights (b*_i)
    const B_STAR: [f64; 7] = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ];

    /// Nodes (c_i) if non-autonomous
    #[allow(unused)]
    const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

    /// Interpolation coefficients (d_i)
    const D: [f64; 7] = [
        -12715105075.0 / 11282082432.0,
        0.0,
        87487479700.0 / 32700410799.0,
        -10690763975.0 / 1880347072.0,
        701980252875.0 / 199316789632.0,
        -1453857185.0 / 822651844.0,
        69997945.0 / 29380423.0,
    ];
}

impl<T: SolverNum> VecOperations<T> for DormandPrince54 {}

impl<T: SolverNum> AdaptiveIntegrationMethod<T> for DormandPrince54 {
    fn next(&self, x: &[T], f: &[Rc<ScalarField<T>>], stepsize: T) -> AdaptiveIntegrationStep<T> {
        let n = x.len();
        let a = &Self::A;
        let b = &Self::B;
        let b_star = &Self::B_STAR;
        let d = &Self::D;

        // Calculate the common parameters
        let k1 = self.vec(n, |i| f[i](x) * stepsize);

        let trial2 = self.vec(n, |i| x[i] + a[1][0].into() * k1[i]);
        let k2 = self.vec(n, |i| f[i](&trial2) * stepsize);

        let trial3 = self.vec(n, |i| {
            x[i] + a[2][0].into() * k1[i] + a[2][1].into() * k2[i]
        });
        let k3 = self.vec(n, |i| f[i](&trial3) * stepsize);

        let trial4 = self.vec(n, |i| {
            x[i] + a[3][0].into() * k1[i] + a[3][1].into() * k2[i] + a[3][2].into() * k3[i]
        });
        let k4 = self.vec(n, |i| f[i](&trial4) * stepsize);

        let trial5 = self.vec(n, |i| {
            x[i] + a[4][0].into() * k1[i]
                + a[4][1].into() * k2[i]
                + a[4][2].into() * k3[i]
                + a[4][3].into() * k4[i]
        });
        let k5 = self.vec(n, |i| f[i](&trial5) * stepsize);

        let trial6 = self.vec(n, |i| {
            x[i] + a[5][0].into() * k1[i]
                + a[5][1].into() * k2[i]
                + a[5][2].into() * k3[i]
                + a[5][3].into() * k4[i]
                + a[5][4].into() * k5[i]
        });
        let k6 = self.vec(n, |i| f[i](&trial6) * stepsize);

        let trial7 = self.vec(n, |i| {
            x[i] + a[6][0].into() * k1[i]
                + a[6][1].into() * k2[i]
                + a[6][2].into() * k3[i]
                + a[6][3].into() * k4[i]
                + a[6][4].into() * k5[i]
                + a[6][5].into() * k6[i]
        });
        let k7 = self.vec(n, |i| f[i](&trial7) * stepsize);

        // Run both RK formulas
        let x_order5 = self.vec(n, |i| {
            x[i] + b[0].into() * k1[i]
                + b[1].into() * k2[i]
                + b[2].into() * k3[i]
                + b[3].into() * k4[i]
                + b[4].into() * k5[i]
                + b[5].into() * k6[i]
                + b[6].into() * k7[i]
        });

        let x_order4 = self.vec(n, |i| {
            x[i] + b_star[0].into() * k1[i]
                + b_star[1].into() * k2[i]
                + b_star[2].into() * k3[i]
                + b_star[3].into() * k4[i]
                + b_star[4].into() * k5[i]
                + b_star[5].into() * k6[i]
                + b_star[6].into() * k7[i]
        });

        // Find the difference between the estimations
        let delta = self.vec(n, |i| x_order5[i] - x_order4[i]);

        // Calculate the interpolation coefficients
        let r1 = x.to_vec();
        let r2 = self.vec(n, |i| x_order5[i] - x[i]);
        let r3 = self.vec(n, |i| k1[i] - r2[i]);
        let r4 = self.vec(n, |i| r2[i] - r3[i] - k7[i]);
        let r5 = self.vec(n, |i| {
            d[0].into() * k1[i]
                + d[2].into() * k3[i]
                + d[3].into() * k4[i]
                + d[4].into() * k5[i]
                + d[5].into() * k6[i]
                + d[6].into() * k7[i]
        });

        return AdaptiveIntegrationStep {
            x_good: x_order5,
            x_bad: x_order4,
            delta,
            interp_coeffs: vec![r1, r2, r3, r4, r5],
        };
    }

    fn interpolate(&self, t: T, t_curr: T, stepsize: T, r: &[Vec<T>]) -> Vec<T> {
        let n = r[0].len();
        // Normalize t to be between 0 and 1
        let s = (t - t_curr) / stepsize;
        let s_sq = s * s;
        let one_minus_s = 1.0.into() - s;
        let one_minus_s_sq = one_minus_s * one_minus_s;

        return self.vec(n, |i| {
            r[0][i]
                + s * r[1][i]
                + s * one_minus_s * r[2][i]
                + s_sq * one_minus_s * r[3][i]
                + s_sq * one_minus_s_sq * r[4][i]
        });
    }
}
