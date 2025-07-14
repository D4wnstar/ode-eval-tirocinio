//! Several examples of integration on well-known physical systems to showcase the rest
//! of the library.

use std::{f64::consts::PI, fs, rc::Rc, time::Instant};

use charming::{
    Chart, HtmlRenderer, ImageFormat, ImageRenderer,
    component::{Axis, Axis3D, Grid, Grid3D, Legend, Title},
    element::{Tooltip, Trigger},
    series::{Line, Scatter},
};
use num::complex::Complex64;

use crate::{
    complex::{PdeSolverComplex, RungeKutta4Complex},
    methods::{AdaptiveIntegrationMethod, Euler, IntegrationMethod, Midpoint, RungeKutta4},
    schedulers::StepsizeScheduler,
    solvers::{OdeAdaptiveSolver, PdeSolver},
};
use crate::{
    solvers::{OdeSolver, OdeSystem},
    utils::Tolerances,
};

const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

/// Simple example of the exponential decay ODE ẋ = -x over a variety of initial values.
/// Integration method can be chosen.
pub fn exponential_decay(method: impl IntegrationMethod + Copy) {
    let ode = Rc::new(|x: &[f64]| -x[0]);
    let t_start = 0.0;
    let t_end = 5.0;
    let stepsize = 0.1;

    let system2 = OdeSystem::new(t_start, &[2.0], &[ode.clone()]);
    let system1 = OdeSystem::new(t_start, &[1.0], &[ode.clone()]);
    let system0 = OdeSystem::new(t_start, &[0.0], &[ode.clone()]);
    let systemm1 = OdeSystem::new(t_start, &[-1.0], &[ode.clone()]);
    let systemm2 = OdeSystem::new(t_start, &[-2.0], &[ode]);

    let points2 = OdeSolver::new(system2, method, stepsize).solve(t_end);
    let points1 = OdeSolver::new(system1, method, stepsize).solve(t_end);
    let points0 = OdeSolver::new(system0, method, stepsize).solve(t_end);
    let pointsm1 = OdeSolver::new(systemm1, method, stepsize).solve(t_end);
    let pointsm2 = OdeSolver::new(systemm2, method, stepsize).solve(t_end);

    let chart = Chart::new()
        .title(Title::new().text("ẋ = -x for different initial values"))
        .background_color("white")
        .tooltip(Tooltip::new().trigger(Trigger::Item))
        .x_axis(Axis::new().name("t"))
        .y_axis(Axis::new().name("x"))
        .series(Line::new().data(tuple_to_vec(&points2, 0)))
        .series(Line::new().data(tuple_to_vec(&points1, 0)))
        .series(Line::new().data(tuple_to_vec(&points0, 0)))
        .series(Line::new().data(tuple_to_vec(&pointsm1, 0)))
        .series(Line::new().data(tuple_to_vec(&pointsm2, 0)));

    save_chart(&chart, "exponential_decay", 1000, 800);
}

/// Simple example of the exponential decay ODE ẋ = -x over a variety of initial values.
/// Unlike `exponential_decay`, this function uses an integration method with an adaptive
/// stepsize, and thus also requires a `StepsizeScheduler`.
pub fn exponential_decay_adaptive(
    method: impl AdaptiveIntegrationMethod + Copy,
    scheduler: impl StepsizeScheduler + Copy,
) {
    let ode = Rc::new(|x: &[f64]| -x[0]);
    let t_start = 0.0;
    let t_end = 5.0;
    let guess_stepsize = 0.1;
    let tolerances = Tolerances::new(1e-6, 1e-6);

    let system2 = OdeSystem::new(t_start, &[2.0], &[ode.clone()]);
    let system1 = OdeSystem::new(t_start, &[1.0], &[ode.clone()]);
    let system0 = OdeSystem::new(t_start, &[0.0], &[ode.clone()]);
    let systemm1 = OdeSystem::new(t_start, &[-1.0], &[ode.clone()]);
    let systemm2 = OdeSystem::new(t_start, &[-2.0], &[ode]);

    let points2 = OdeAdaptiveSolver::new(system2, method, scheduler, tolerances)
        .solve(t_end, guess_stepsize)
        .points;
    let points1 = OdeAdaptiveSolver::new(system1, method, scheduler, tolerances)
        .solve(t_end, guess_stepsize)
        .points;
    let points0 = OdeAdaptiveSolver::new(system0, method, scheduler, tolerances)
        .solve(t_end, guess_stepsize)
        .points;
    let pointsm1 = OdeAdaptiveSolver::new(systemm1, method, scheduler, tolerances)
        .solve(t_end, guess_stepsize)
        .points;
    let pointsm2 = OdeAdaptiveSolver::new(systemm2, method, scheduler, tolerances)
        .solve(t_end, guess_stepsize)
        .points;

    let chart = Chart::new()
        .title(Title::new().text("ẋ = -x for different initial values (adaptive stepsize)"))
        .background_color("white")
        .tooltip(Tooltip::new().trigger(Trigger::Item))
        .x_axis(Axis::new().name("t"))
        .y_axis(Axis::new().name("x"))
        .series(Line::new().data(tuple_to_vec(&points2, 0)))
        .series(Line::new().data(tuple_to_vec(&points1, 0)))
        .series(Line::new().data(tuple_to_vec(&points0, 0)))
        .series(Line::new().data(tuple_to_vec(&pointsm1, 0)))
        .series(Line::new().data(tuple_to_vec(&pointsm2, 0)));

    save_chart(&chart, "exponential_decay_adaptive", 1000, 800);
}

/// A comparison of several integration methods on the exponential decay ODE ẋ = -x.
/// Uses constant stepsize.
pub fn method_comparison() {
    let ode = Rc::new(|x: &[f64]| -x[0]);
    let t_start = 0.0;
    let t_end = 5.0;
    let system = OdeSystem::new(t_start, &[2.0], &[ode]);

    // A low stepsize is recommended to make the the errors more visually obvious.
    let stepsize = 0.5;
    let points_euler = OdeSolver::new(system.clone(), Euler::default(), stepsize).solve(t_end);
    let points_midpoint =
        OdeSolver::new(system.clone(), Midpoint::default(), stepsize).solve(t_end);
    let points_rk4 = OdeSolver::new(system.clone(), RungeKutta4::default(), stepsize).solve(t_end);

    // Smaller step size to see how methods scale
    let stepsize = 0.1;
    let points_euler_dense =
        OdeSolver::new(system.clone(), Euler::default(), stepsize).solve(t_end);
    let points_midpoint_dense =
        OdeSolver::new(system.clone(), Midpoint::default(), stepsize).solve(t_end);
    let points_rk4_dense = OdeSolver::new(system, RungeKutta4::default(), stepsize).solve(t_end);

    // Exact analytical solution
    let solution = |t: f64, x0: f64| x0 * (-t).exp();
    let points_exact: Vec<Vec<f64>> = (0..=50)
        .map(|i| {
            let t = i as f64 * 0.1;
            vec![t, solution(t, 2.0)]
        })
        .collect();

    let mut chart = Chart::new()
        .title(
            Title::new()
                .text("Method comparison (ẋ = -x; initial value x0 = 2.0; stepsize = 0.5 [left] and 0.1 [right])")
                .left("center"),
        )
        .tooltip(Tooltip::new().trigger(Trigger::Item))
        .legend(
            Legend::new()
                .data(vec![
                    "Euler",
                    "Midpoint",
                    "Fourth-order Runge-Kutta",
                    "Exact solution (x(t) = x0 * e^(-t))",
                ])
                .top("bottom"),
        )
        .grid(Grid::new().width("42%").left("5%"))
        .grid(Grid::new().width("42%").right("5%"))
        .background_color("white");

    // Left plot
    chart = chart
        .x_axis(Axis::new().name("t").max(5.0))
        .y_axis(Axis::new().name("x"))
        .series(
            Line::new()
                .data(tuple_to_vec(&points_euler, 0))
                .name("Euler"),
        )
        .series(
            Line::new()
                .data(tuple_to_vec(&points_midpoint, 0))
                .name("Midpoint"),
        )
        .series(
            Line::new()
                .data(tuple_to_vec(&points_rk4, 0))
                .name("Fourth-order Runge-Kutta"),
        )
        .series(
            Line::new()
                .data(points_exact.clone())
                .name("Exact solution (x(t) = x0 * e^(-t))"),
        );

    // Right plot
    chart = chart
        .x_axis(Axis::new().name("t").grid_index(1).max(5.0))
        .y_axis(Axis::new().name("x").grid_index(1))
        .series(
            Line::new()
                .data(tuple_to_vec(&points_euler_dense, 0))
                .name("Euler")
                .x_axis_index(1)
                .y_axis_index(1),
        )
        .series(
            Line::new()
                .data(tuple_to_vec(&points_midpoint_dense, 0))
                .name("Midpoint")
                .x_axis_index(1)
                .y_axis_index(1),
        )
        .series(
            Line::new()
                .data(tuple_to_vec(&points_rk4_dense, 0))
                .name("Fourth-order Runge-Kutta")
                .x_axis_index(1)
                .y_axis_index(1),
        )
        .series(
            Line::new()
                .data(points_exact)
                .name("Exact solution (x(t) = x0 * e^(-t))")
                .x_axis_index(1)
                .y_axis_index(1),
        );

    save_chart(&chart, "method_comparison", 2000, 800);
}

/// A simple one-dimensional harmonic oscillator with the given mass and angular frequency.
pub fn harmonic_oscillator(mass: f64, freq: f64, method: impl IntegrationMethod) {
    // x = (q, p), so x[0] = q and x[1] = p
    let q_dot = Rc::new(move |x: &[f64]| x[1] / mass);
    let p_dot = Rc::new(move |x: &[f64]| -mass * freq.powi(2) * x[0]);

    let q_start = 0.0;
    let p_start = 1.0;

    let t_start = 0.0;
    let t_end = 5.0;
    let stepsize = 0.1;

    let system = OdeSystem::new(t_start, &[q_start, p_start], &[q_dot, p_dot]);
    let points = OdeSolver::new(system, method, stepsize).solve(t_end);

    let chart = process_harmonic_oscillator(points, q_start, p_start, mass, freq);
    save_chart(&chart, "harmonic_oscillator", 1000, 1400);
}

/// A simple one-dimensional harmonic oscillator with the given mass and angular frequency.
/// Uses an adaptive stepsize.
pub fn harmonic_oscillator_adaptive(
    mass: f64,
    freq: f64,
    method: impl AdaptiveIntegrationMethod + Copy,
    scheduler: impl StepsizeScheduler + Copy,
) {
    // x = (q, p), so x[0] = q and x[1] = p
    let q_dot = Rc::new(move |x: &[f64]| x[1] / mass);
    let p_dot = Rc::new(move |x: &[f64]| -mass * freq.powi(2) * x[0]);

    let q_start = 0.0;
    let p_start = 1.0;

    let t_start = 0.0;
    let t_end = 5.0;
    let guess_stepsize = 0.1;
    let tolerances = Tolerances::new(1e-6, 1e-6);

    let system = OdeSystem::new(t_start, &[q_start, p_start], &[q_dot, p_dot]);
    let solution =
        OdeAdaptiveSolver::new(system, method, scheduler, tolerances).solve(t_end, guess_stepsize);

    let chart = process_harmonic_oscillator(solution.points, q_start, p_start, mass, freq);
    save_chart(&chart, "harmonic_oscillator_adaptive", 1000, 1400);
}

/// A simple one-dimensional harmonic oscillator with the given mass and angular frequency.
/// Uses an adaptive stepsize and will interpolate at the given points.
pub fn harmonic_oscillator_interpolation(
    mass: f64,
    freq: f64,
    method: impl AdaptiveIntegrationMethod,
    scheduler: impl StepsizeScheduler,
) {
    // x = (q, p), so x[0] = q and x[1] = p
    let q_dot = Rc::new(move |x: &[f64]| x[1] / mass);
    let p_dot = Rc::new(move |x: &[f64]| -mass * freq.powi(2) * x[0]);

    let q_start = 0.0;
    let p_start = 1.0;

    let t_start = 0.0;
    let t_end = 5.0;
    let guess_stepsize = 0.1;
    let tolerances = Tolerances::new(1e-6, 1e-6);

    // Interpolate every 0.1 step of t
    let n_to_interp = f64::floor((t_end - t_start) / 0.1) as usize + 1;
    let to_interpolate = [0.1]
        .repeat(n_to_interp)
        .iter()
        .enumerate()
        .map(|(i, t)| i as f64 * t)
        .collect();

    let system = OdeSystem::new(t_start, &[q_start, p_start], &[q_dot, p_dot]);
    let solution = OdeAdaptiveSolver::new(system, method, scheduler, tolerances)
        .at_points(to_interpolate)
        .solve(t_end, guess_stepsize);

    // let mut with_interp = solution.interp_points.unwrap();
    // with_interp.append(&mut solution.points.clone());
    // with_interp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let chart = Chart::new()
        .title(Title::new().text(format!("Harmonic oscillator (q0 = {q_start}, p0 = {p_start}, mass = {mass}, frequency = {freq})")))
        .background_color("white")
        .tooltip(Tooltip::new().trigger(Trigger::Item))
        .x_axis(Axis::new().name("t"))
        .y_axis(Axis::new().name("q"))
        .series(Scatter::new().data(tuple_to_vec(&solution.points, 0)).name("q solved"))
        .series(Line::new().data(tuple_to_vec(&solution.interp_points.unwrap(), 0)).name("q interpolated"))
        .legend(
            Legend::new().data(vec!["q solved", "q interpolated"]).top("bottom")
        );
    save_chart(&chart, "harmonic_oscillator_interpolated", 1000, 800);
}

/// Convenience function to process harmonic oscillator simulations.
fn process_harmonic_oscillator(
    points: Vec<(f64, Vec<f64>)>,
    q_start: f64,
    p_start: f64,
    mass: f64,
    freq: f64,
) -> Chart {
    // Uncomment to see comparison with exact solution
    // Commented because the solutions superimpose and you can't see the numerical one
    // let q = move |t: f64| p_start / (mass * freq) * f64::sin(freq * t);
    // let p = move |t: f64| p_start * f64::cos(freq * t);
    // let exact_q_points: Vec<Vec<f64>> = (0..=50)
    //     .map(|i| vec![stepsize * i as f64, q(stepsize * i as f64)])
    //     .collect();
    // let exact_p_points: Vec<Vec<f64>> = (0..=50)
    //     .map(|i| vec![stepsize * i as f64, p(stepsize * i as f64)])
    //     .collect();

    let mut chart = Chart::new()
        .title(Title::new().text(format!(
            "Harmonic oscillator (q0 = {q_start}, p0 = {p_start}, mass = {mass}, frequency = {freq})"
        )))
        .background_color("white")
        .tooltip(Tooltip::new().trigger(Trigger::Item));

    // Top plot
    let kinetic_over_time: Vec<Vec<f64>> = points
        .iter()
        .map(|(t, qp)| vec![*t, qp[1].powi(2) / (2.0 * mass)])
        .collect();
    let potential_over_time: Vec<Vec<f64>> = points
        .iter()
        .map(|(t, qp)| vec![*t, 0.5 * mass * freq.powi(2) * qp[0].powi(2)])
        .collect();
    let energy_over_time: Vec<Vec<f64>> = kinetic_over_time
        .iter()
        .zip(&potential_over_time)
        .map(|(kinetic, potential)| vec![kinetic[0], kinetic[1] + potential[1]])
        .collect();

    chart = chart
        .grid(Grid::new().top("5%").height("42%"))
        .x_axis(Axis::new().name("t"))
        .y_axis(Axis::new())
        .series(Line::new().data(tuple_to_vec(&points, 0)).name("q"))
        .series(Line::new().data(tuple_to_vec(&points, 1)).name("p"))
        // .series(Line::new().data(exact_q_points).name("q exact"))
        // .series(Line::new().data(exact_p_points).name("p exact"))
        .series(Line::new().data(kinetic_over_time).name("kinetic energy"))
        .series(
            Line::new()
                .data(potential_over_time)
                .name("potential energy"),
        )
        .series(Line::new().data(energy_over_time).name("total energy"));

    // Bottom plot
    let p_over_q: Vec<Vec<f64>> = points.iter().map(|(_, qp)| vec![qp[0], qp[1]]).collect();

    return chart
        .grid(Grid::new().bottom("5%").height("42%"))
        .x_axis(Axis::new().name("q").grid_index(1).min(-1.2).max(1.2))
        .y_axis(Axis::new().name("p").grid_index(1).min(-1.2).max(1.2))
        .series(
            Line::new()
                .data(p_over_q)
                .name("phase trajectory")
                .x_axis_index(1)
                .y_axis_index(1),
        )
        .legend(
            Legend::new()
                .data(vec![
                    "q",
                    "p",
                    "q exact",
                    "p exact",
                    "kinetic energy",
                    "potential energy",
                    "total energy",
                    "phase trajectory",
                ])
                .top("center"),
        );
}

/// A simple pendulum with the given mass `m`, gravitational acceleration `g` and length `l`.
/// Will use an adaptive stepsize. This is a proper pendulum, without small swing approximation.
pub fn simple_pendulum_adaptive(
    m: f64,
    g: f64,
    l: f64,
    method: impl AdaptiveIntegrationMethod,
    scheduler: impl StepsizeScheduler,
) {
    // x = (theta, p_theta)
    let theta_dot = Rc::new(move |x: &[f64]| x[1] / (m * l.powi(2)));
    let p_dot = Rc::new(move |x: &[f64]| -m * g * l * f64::sin(x[0]));

    let theta_start = 0.0;
    let p_start = 1.0;

    let t_start = 0.0;
    let t_end = 5.0;
    let guess_stepsize = 0.1;
    let tolerances = Tolerances::new(1e-7, 1e-7);

    let system = OdeSystem::new(t_start, &[theta_start, p_start], &[theta_dot, p_dot]);
    let solution =
        OdeAdaptiveSolver::new(system, method, scheduler, tolerances).solve(t_end, guess_stepsize);
    let points = solution.points;

    let mut chart = Chart::new()
        .title(Title::new().text(format!("Simple pendulum (θ0 = {theta_start}, p0 = {p_start}, mass = {m}, length = {l}, g = {g})")))
        .background_color("white")
        .tooltip(Tooltip::new().trigger(Trigger::Item));

    // Top plot
    let kinetic_over_time: Vec<Vec<f64>> = points
        .iter()
        .map(|(t, x)| vec![*t, x[1].powi(2) / (2.0 * m * l.powi(2))])
        .collect();
    let potential_over_time: Vec<Vec<f64>> = points
        .iter()
        .map(|(t, x)| vec![*t, m * g * l * (1.0 - f64::cos(x[0]))])
        .collect();
    let energy_over_time: Vec<Vec<f64>> = kinetic_over_time
        .iter()
        .zip(&potential_over_time)
        .map(|(kinetic, potential)| vec![kinetic[0], kinetic[1] + potential[1]])
        .collect();

    chart = chart
        .grid(Grid::new().height("55%"))
        .grid(Grid::new().height("30%").bottom("5%"))
        .x_axis(Axis::new().name("t"))
        .y_axis(Axis::new())
        .series(Line::new().data(tuple_to_vec(&points, 0)).name("theta"))
        .series(Line::new().data(tuple_to_vec(&points, 1)).name("p"))
        .series(Line::new().data(kinetic_over_time).name("kinetic energy"))
        .series(
            Line::new()
                .data(potential_over_time)
                .name("potential energy"),
        )
        .series(Line::new().data(energy_over_time).name("total energy"));

    // Bottom plot
    let p_over_theta: Vec<Vec<f64>> = points.iter().map(|(_, x)| vec![x[0], x[1]]).collect();

    chart = chart
        .x_axis(Axis::new().name("theta").grid_index(1))
        .y_axis(Axis::new().name("p").grid_index(1))
        .series(
            Line::new()
                .data(p_over_theta)
                .name("phase trajectory")
                .x_axis_index(1)
                .y_axis_index(1),
        )
        .legend(
            Legend::new()
                .data(vec![
                    "theta",
                    "p",
                    "kinetic energy",
                    "potential energy",
                    "total energy",
                    "phase trajectory",
                ])
                .top("bottom"),
        );

    save_chart(&chart, "simple_pendulum_adaptive", 1000, 1100);
}

/// A simple pendulum with the given mass `m`, gravitational acceleration `g` and length `l`.
/// Will use an adaptive stepsize. The plot compares the simulated solution with the analytic
/// solution in the small wings approximation of sin(x) ~= x.
pub fn simple_pendulum_against_small_swings(
    m: f64,
    g: f64,
    l: f64,
    method: impl AdaptiveIntegrationMethod,
    scheduler: impl StepsizeScheduler,
) {
    // x = (theta, p_theta)
    let theta_dot = Rc::new(move |x: &[f64]| x[1] / (m * l.powi(2)));
    let p_dot = Rc::new(move |x: &[f64]| -m * g * l * f64::sin(x[0]));

    let theta_start = std::f64::consts::FRAC_PI_3;
    let p_start = 0.0;

    let t_start = 0.0;
    let t_end = 5.0;
    let guess_stepsize = 0.1;
    let tolerances = Tolerances::new(1e-7, 1e-7);

    let system = OdeSystem::new(t_start, &[theta_start, p_start], &[theta_dot, p_dot]);
    let points = OdeAdaptiveSolver::new(system, method, scheduler, tolerances)
        .solve(t_end, guess_stepsize)
        .points;

    // Exact solution in small swings is a harmonic oscillator with freq = sqrt(g / l)
    let omega = f64::sqrt(g / l);
    let x = |t: f64, q0: f64, p0: f64| {
        q0 * f64::cos(omega * t) + p0 / (omega * m) * f64::sin(omega * t)
    };
    let points_exact: Vec<Vec<f64>> = (0..=50)
        .map(|i| {
            let t = i as f64 * 0.1;
            vec![t, x(t, theta_start, p_start)]
        })
        .collect();

    let chart = Chart::new()
        .title(Title::new().text(format!(
            "Simple pendulum (θ0 = {}°, p0 = {p_start}, mass = {m}, length = {l}, g = {g})",
            (theta_start * RAD_TO_DEG).round() as u32
        )))
        .background_color("white")
        .tooltip(Tooltip::new().trigger(Trigger::Item))
        .x_axis(Axis::new().name("t"))
        .y_axis(Axis::new())
        .series(Line::new().data(tuple_to_vec(&points, 0)).name("theta"))
        // .series(Line::new().data(tuple_to_vec(&points, 1)).name("p"))
        .series(
            Line::new()
                .data(points_exact)
                .name("theta exact (small swings)"),
        )
        .legend(
            Legend::new()
                .data(vec!["theta", "p", "theta exact (small swings)"])
                .top("bottom"),
        );

    save_chart(&chart, "simple_pendulum_against_small_swings", 1000, 800);
}

/// A simple pendulum with the given mass `m`, gravitational acceleration `g` and length `l`.
/// Will use an adaptive stepsize. Several starting conditions are compared to give an idea
/// of the behavior of the system, including full 360° swings.
pub fn simple_pendulum_comparison(
    m: f64,
    g: f64,
    l: f64,
    method: impl AdaptiveIntegrationMethod + Copy,
    scheduler: impl StepsizeScheduler + Copy,
) {
    // x = (theta, p_theta)
    let theta_dot = Rc::new(move |x: &[f64]| x[1] / (m * l.powi(2)));
    let p_dot = Rc::new(move |x: &[f64]| -m * g * l * f64::sin(x[0]));

    let pi = std::f64::consts::PI;
    let theta_starts = [0.0, pi / 12.0, pi / 6.0, pi / 3.0, pi / 2.0, pi / 1.5, pi];
    let p_start = 0.0;

    let t_start = 0.0;
    let t_end = 5.0;
    let guess_stepsize = 0.1;
    let tolerances = Tolerances::new(1e-7, 1e-7);

    // Simulate normal swings
    let mut simulated_t_theta = vec![];
    for theta_start in theta_starts {
        let system = OdeSystem::new(
            t_start,
            &[theta_start, p_start],
            &[theta_dot.clone(), p_dot.clone()],
        );
        let points = OdeAdaptiveSolver::new(system, method, scheduler, tolerances)
            .solve(t_end, guess_stepsize)
            .points;
        let degrees = (theta_start * RAD_TO_DEG).round() as u32;
        simulated_t_theta.push((degrees, points));
    }

    // Simulate a full swing
    let theta_start_fullswing = pi / 1.2;
    let p_start_fullswing = -2.0;
    let system = OdeSystem::new(
        t_start,
        &[theta_start_fullswing, p_start_fullswing],
        &[theta_dot.clone(), p_dot.clone()],
    );
    let mut points = OdeAdaptiveSolver::new(system, method, scheduler, tolerances)
        .solve(t_end, guess_stepsize)
        .points;
    // Make the angular coordinate loop
    loop_angular_coord(&mut points, 0);

    let degrees = (theta_start_fullswing * RAD_TO_DEG).round() as u32;
    simulated_t_theta.push((degrees, points));

    let mut chart = Chart::new()
        .title(Title::new().text(format!(
            "Simple pendulum (p0 = {p_start} except θ0 = {degrees}° for which p0 = {p_start_fullswing}, mass = {m}, length = {l}, g = {g})"
        )))
        .background_color("white")
        .tooltip(Tooltip::new().trigger(Trigger::Item))
        .grid(Grid::new().height("55%"))
        .grid(Grid::new().height("30%").bottom("5%"))
        .x_axis(Axis::new().name("t"))
        .y_axis(Axis::new().name("θ [degrees]"))
        .x_axis(Axis::new().name("θ [degrees]").grid_index(1))
        .y_axis(Axis::new().name("p").grid_index(1));

    for (deg, points) in &mut simulated_t_theta {
        // Convert to degrees
        points.iter_mut().for_each(|(_, x)| x[0] *= RAD_TO_DEG);

        let trajectory = tuple_to_vec(&points, 0);
        let phase: Vec<Vec<f64>> = points
            .into_iter()
            .map(|(_m, phase)| phase.clone())
            .collect();
        chart = chart
            .series(Line::new().data(trajectory).name(format!("θ0 = {deg}°")))
            .series(
                Line::new()
                    .data(phase)
                    .name(format!("θ0 = {deg}°"))
                    .x_axis_index(1)
                    .y_axis_index(1),
            )
    }

    chart = chart.legend(
        Legend::new()
            .data(
                simulated_t_theta
                    .iter()
                    .map(|(deg, _)| format!("θ0 = {deg}°"))
                    .collect(),
            )
            .top("bottom"),
    );

    save_chart(&chart, "simple_pendulum_comparison", 1000, 1100);
}

/// An elastic pendulum with mass `m` and gravitational acceleration `g`.
/// The pendulum's rod is considered to be a spring with rest length l0 and elastic
/// constant k. A few starting conditions are shown.
pub fn elastic_pendulum_comparison(
    m: f64,
    g: f64,
    l0: f64,
    k: f64,
    method: impl AdaptiveIntegrationMethod + Copy,
    scheduler: impl StepsizeScheduler + Copy,
) {
    // x = (r, θ, p_r, p_θ)
    //      0  1  2    3
    let r_dot = Rc::new(move |x: &[f64]| x[2] / m);
    let theta_dot = Rc::new(move |x: &[f64]| x[3] / (m * x[0].powi(2)));
    let p_r_dot = Rc::new(move |x: &[f64]| {
        x[3].powi(2) / (m * x[0].powi(3)) + m * g * f64::cos(x[1]) + k * (l0 - x[0])
    });
    let p_theta_dot = Rc::new(move |x: &[f64]| -m * g * x[0] * f64::sin(x[1]));

    let r_start = l0;
    let theta_starts = [
        0.0,
        std::f64::consts::FRAC_PI_6,
        std::f64::consts::FRAC_PI_2,
    ];
    let p_r_start = 0.0;
    let p_theta_start = 0.0;

    let t_start = 0.0;
    let t_end = 10.0;
    let starting_stepsize = 0.1;
    let tolerances = Tolerances::new(1e-7, 1e-7);

    // Simulate normal swings
    let mut simulated_points = vec![];
    for theta_start in theta_starts {
        let system = OdeSystem::new(
            t_start,
            &[r_start, theta_start, p_r_start, p_theta_start],
            &[
                r_dot.clone(),
                theta_dot.clone(),
                p_r_dot.clone(),
                p_theta_dot.clone(),
            ],
        );
        let points = OdeAdaptiveSolver::new(system, method, scheduler, tolerances)
            .solve(t_end, starting_stepsize)
            .points;
        let degrees = (theta_start * RAD_TO_DEG).round() as u32;
        let name = format!("{degrees}°");
        simulated_points.push((name, points));
    }

    // Simulate forced swing
    let theta_start = std::f64::consts::FRAC_PI_2;
    let system = OdeSystem::new(
        t_start,
        &[r_start, theta_start, -1.0, -2.0],
        &[
            r_dot.clone(),
            theta_dot.clone(),
            p_r_dot.clone(),
            p_theta_dot.clone(),
        ],
    );
    let mut points = OdeAdaptiveSolver::new(system, method, scheduler, tolerances)
        .solve(t_end, starting_stepsize)
        .points;
    loop_angular_coord(&mut points, 1);
    let degrees = (theta_start * RAD_TO_DEG).round() as u32;
    let name = format!("{degrees}° (p_r0 = -1, p_θ0 = -2)");
    simulated_points.push((name, points));

    // Define 2x2 plot layout
    let mut chart = Chart::new()
        .title(Title::new().text(format!(
            "Elastic pendulum (r0 = {r_start}, p_r0 = {p_r_start}, p_θ0 = {p_theta_start}, mass = {m}, rest length = {l0}, g = {g}, k = {k})"
        )).left("center"))
        .background_color("white")
        .tooltip(Tooltip::new().trigger(Trigger::Item))
        .grid(Grid::new().width("42%").height("40%").left("5%").top("7%"))
        .grid(Grid::new().width("42%").height("40%").right("5%").top("7%"))
        .grid(Grid::new().width("42%").height("40%").left("5%").bottom("5%"))
        .grid(Grid::new().width("42%").height("40%").right("5%").bottom("5%"))
        // theta over time
        .title(Title::new().text("θ over time").top("3%").left("20%"))
        .x_axis(Axis::new().name("t"))
        .y_axis(Axis::new().name("θ [degrees]"))
        // r over time
        .title(Title::new().text("r over time").top("3%").right("20%"))
        .x_axis(Axis::new().name("t").grid_index(1))
        .y_axis(Axis::new().name("r").grid_index(1))
        // cartesian trajectory
        .title(Title::new().text("Trajectory (Cartesian)").top("50%").left("20%"))
        .x_axis(Axis::new().name("x").grid_index(2))
        .y_axis(Axis::new().name("y").grid_index(2))
        // polar trajectory
        .title(Title::new().text("Phase portrait (θ, p_θ)").top("50%").right("16%"))
        .x_axis(Axis::new().name("θ [degrees]").grid_index(3))
        .y_axis(Axis::new().name("p_θ").grid_index(3));

    // Add the data
    for (name, points) in &mut simulated_points {
        // Convert to Cartesian coordinates (x = rsin(θ), y = rcos(θ))
        let cartesian = points
            .iter()
            .map(|(_, x)| vec![x[0] * f64::sin(x[1]), -x[0] * f64::cos(x[1])])
            .collect();

        // Convert radians to degrees
        points.iter_mut().for_each(|(_, x)| x[1] *= RAD_TO_DEG);

        // Find angular phase portrait (θ, p_θ)
        let ang_phase = points.iter().map(|(_, x)| vec![x[1], x[3]]).collect();

        // Make legend name
        let name = format!("θ0 = {name}");
        chart = chart
            // theta over time
            .series(Line::new().data(tuple_to_vec(&points, 1)).name(&name))
            // r over time
            .series(
                Line::new()
                    .data(tuple_to_vec(&points, 0))
                    .name(&name)
                    .x_axis_index(1)
                    .y_axis_index(1),
            )
            // Cartesian trajectory
            .series(
                Line::new()
                    .data(cartesian)
                    .name(&name)
                    .x_axis_index(2)
                    .y_axis_index(2),
            )
            // Angular phase portrait
            .series(
                Line::new()
                    .data(ang_phase)
                    .name(&name)
                    .x_axis_index(3)
                    .y_axis_index(3),
            );
    }

    chart = chart.legend(
        Legend::new()
            .data(
                simulated_points
                    .iter()
                    .map(|(name, _)| format!("θ0 = {name}"))
                    .collect(),
            )
            .top("bottom"),
    );

    save_chart(&chart, "elastic_pendulum_comparison", 1600, 1200);
}

pub fn heat_equation() {
    let x_start = 0.0;
    let x_end = 1.0;
    let grid_points = 20;

    // Initial time condition
    let ic = Rc::new(move |x: f64| f64::sin(PI / 2.0 * x));
    // Spatial discretization with second-order central finite differences
    let disc = Rc::new(move |state: &[f64], dx: f64, i: usize| {
        let delta_sq = dx.powi(2);
        if i == 0 {
            // Hard wall on left side
            0.0
        } else if i == grid_points - 1 {
            // Ghost point on right side
            2.0 * (state[i - 1] - state[i]) / delta_sq
        } else {
            // Second-order central FD in the middle
            (state[i + 1] - 2.0 * state[i] + state[i - 1]) / delta_sq
        }
    });

    let points = PdeSolver::new(RungeKutta4::default(), 0.001, ic, disc).solve(
        0.0,
        1.0,
        x_start,
        x_end,
        grid_points,
    );

    // data = [
    //   [[t0, x0, u00], [t1, x0, u01], [t2, x0, u02], ...], // ODE 0
    //   [[t0, x1, u10], [t1, x1, u11], [t2, x1, u12], ...], // ODE 1
    //   [[t0, x2, u20], [t1, x2, u21], [t2, x2, u22], ...], // ODE 2
    //   ...
    // ]
    let x_step = (x_end - x_start) / (grid_points - 1) as f64;
    let mut data: Vec<Vec<Vec<f64>>> = vec![Vec::new(); grid_points];
    for (t, x) in points {
        for i in 0..data.len() {
            data[i].push(vec![t, x_step * i as f64, x[i]]);
        }
    }

    // Write the numerical data out to JSON for plotting somewhere else
    // since charming's support of 3D plot is not great
    let out = serde_json::to_string(&data).unwrap();
    fs::write("gallery/data/heat_equation.json", out).unwrap();
    let mut chart = Chart::new()
        .title(Title::new().text("Heat equation, (X = time, Y = space, Z = Temperature)"))
        .x_axis3d(Axis3D::new())
        .y_axis3d(Axis3D::new())
        .z_axis3d(Axis3D::new())
        .grid3d(Grid3D::new());

    // Add all ODE solutions to plot
    for d in data {
        chart = chart.series(Line::new().data(d));
    }

    let renderer = HtmlRenderer::new("ODE Chart", 1200, 900);
    let html = renderer.render(&chart).unwrap().replace("line", "line3D"); // charming does not currently have bindings for line3D
    fs::write("gallery/interactive/heat_equation.html", html).unwrap();
}

pub fn schrodinger_equation() {
    let x_start = -5.0;
    let x_end = 5.0;
    let grid_points = 100;
    let dx = (x_end - x_start) / (grid_points - 1) as f64;

    // Initial time condition
    let sigma: f64 = 1.0;
    let ic = Rc::new(move |x: f64, j: usize| {
        if j == 0 || j == grid_points - 1 {
            // Open boundary conditions (edges are zero)
            Complex64::ZERO
        } else {
            // Gaussian wave packet
            Complex64::new(
                // (2 / 𝛔π)^(1/4) * e^(-x^2 / 𝛔)
                (2.0 / (sigma * PI)).powf(0.25) * f64::exp(-x.powi(2) / sigma),
                0.0,
            )
        }
    });

    // Spatial discretization (system of ODEs)
    let disc = Rc::new(move |psi: &[Complex64], x_j: f64, dx: f64, j: usize| {
        let i = Complex64::I;
        if j == 0 || j == grid_points - 1 {
            // Open boundary conditions (edges remain zero)
            Complex64::ZERO
        } else {
            // Inner grid uses second-order central finite difference
            i * (psi[j + 1] - 2.0 * psi[j] + psi[j - 1]) / (2.0 * dx * dx)
                - i * x_j.powi(4) / 2.0 * psi[j]
        }
    });

    let solution = PdeSolverComplex::new(RungeKutta4Complex::default(), 0.001, ic, disc).solve(
        0.0,
        6.0,
        x_start,
        x_end,
        grid_points,
    );

    let probs = solution.into_square_norms();

    // Write the numerical data out to JSON for plotting somewhere else
    // since charming's support of 3D plot is not great
    let out = serde_json::to_string(&probs).unwrap();
    fs::write("gallery/data/schrodinger_equation.json", out).unwrap();

    // Only plot the normalization and energy, do the actual wavefunction in Julia
    let norm: Vec<Vec<f64>> = solution
        .points
        .iter()
        .map(|(t, psi)| vec![*t, psi.iter().map(|psi_j| psi_j.norm_sqr() * dx).sum()])
        .collect();

    let energy: Vec<Vec<f64>> = solution
        .points
        .iter()
        .map(|(t, psi)| {
            let mut energy = 0.0;
            // Boundaries are always zero
            for j in 1..(psi.len() - 1) {
                let x_j = x_start + dx * j as f64;
                let e = psi[j].norm_sqr() * x_j.powi(4) / 2.0 * dx
                    - psi[j].conj() * (psi[j + 1] - 2.0 * psi[j] + psi[j - 1]) / (2.0 * dx);
                energy += e.re
            }
            vec![*t, energy]
        })
        .collect();

    let chart = Chart::new()
        .title(Title::new().text("Schrödinger equation conservation"))
        .grid(Grid::new().height("40%").left("5%").top("7%"))
        .x_axis(Axis::new().name("Time"))
        .y_axis(Axis::new().name("Normalization |ψ|^2"))
        .grid(Grid::new().height("40%").left("5%").bottom("7%"))
        .x_axis(Axis::new().name("Time").grid_index(1))
        .y_axis(Axis::new().name("Energy").grid_index(1))
        .background_color("white")
        .series(Line::new().data(norm).show_symbol(false))
        .series(
            Line::new()
                .data(energy)
                .x_axis_index(1)
                .y_axis_index(1)
                .show_symbol(false),
        );

    let mut renderer = ImageRenderer::new(1200, 900);
    renderer
        .save_format(
            ImageFormat::Png,
            &chart,
            "gallery/images/schrodinger_conservation.png",
        )
        .unwrap();
}

/// Convenience function to loop an angular coordinate between -pi and +pi in place.
fn loop_angular_coord(points: &mut Vec<(f64, Vec<f64>)>, arg: usize) {
    points.iter_mut().for_each(|(_, x)| {
        while x[arg] > PI {
            x[arg] -= 2.0 * PI;
        }
        while x[arg] < -PI {
            x[arg] += 2.0 * PI;
        }
    });
}

/// Convenience function to save a [`charming::Chart`] to disk.
fn save_chart(chart: &Chart, filename: &str, width: u32, height: u32) {
    let start = Instant::now();
    let mut renderer = HtmlRenderer::new("ODE Chart", width as u64, height as u64);
    renderer
        .save(&chart, format!("gallery/interactive/{filename}.html"))
        .unwrap();
    println!(
        "Saving {filename} as HTML took {} ms",
        start.elapsed().as_millis(),
    );

    let start = Instant::now();
    let mut image_renderer = ImageRenderer::new(width, height);
    image_renderer
        .save_format(
            ImageFormat::Png,
            &chart,
            format!("gallery/images/{filename}.png"),
        )
        .unwrap();
    println!(
        "Saving {filename} as PNG took {} ms",
        start.elapsed().as_millis(),
    );
}

/// Convenience function to unpack the tuple in the output of a `Solver` into a format that `charming` likes.
fn tuple_to_vec(vec: &Vec<(f64, Vec<f64>)>, arg: usize) -> Vec<Vec<f64>> {
    vec.iter().map(|(t, x)| vec![*t, x[arg]]).collect()
}
