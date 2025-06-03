use std::{rc::Rc, time::Instant};

use charming::{
    Chart, HtmlRenderer, ImageFormat, ImageRenderer,
    component::{Axis, Grid, Legend, Title},
    series::Line,
};

use crate::solvers::{Solver, System};
use crate::{
    methods::{AdaptiveIntegrationMethod, Euler, IntegrationMethod, Midpoint, RungeKutta4},
    schedulers::StepsizeScheduler,
    solvers::{AdaptiveSolver, Tolerances},
};

/// Simple example of the exponential decay ODE ẋ = -x over a variety of initial values.
/// Integration method can be chosen.
pub fn exponential_decay(method: impl IntegrationMethod + Copy) {
    let ode = Rc::new(|x: &[f64]| -x[0]);
    let t_start = 0.0;
    let t_end = 5.0;
    let stepsize = 0.1;

    let system2 = System::new(t_start, &[2.0], &[ode.clone()]);
    let system1 = System::new(t_start, &[1.0], &[ode.clone()]);
    let system0 = System::new(t_start, &[0.0], &[ode.clone()]);
    let systemm1 = System::new(t_start, &[-1.0], &[ode.clone()]);
    let systemm2 = System::new(t_start, &[-2.0], &[ode]);

    let points2 = Solver::new(system2, method, stepsize).solve(t_end);
    let points1 = Solver::new(system1, method, stepsize).solve(t_end);
    let points0 = Solver::new(system0, method, stepsize).solve(t_end);
    let pointsm1 = Solver::new(systemm1, method, stepsize).solve(t_end);
    let pointsm2 = Solver::new(systemm2, method, stepsize).solve(t_end);

    let chart = Chart::new()
        .title(Title::new().text("ẋ = -x for different initial values"))
        .background_color("white")
        .x_axis(Axis::new().name("t").max(5.0))
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

    let system2 = System::new(t_start, &[2.0], &[ode.clone()]);
    let system1 = System::new(t_start, &[1.0], &[ode.clone()]);
    let system0 = System::new(t_start, &[0.0], &[ode.clone()]);
    let systemm1 = System::new(t_start, &[-1.0], &[ode.clone()]);
    let systemm2 = System::new(t_start, &[-2.0], &[ode]);

    let points2 =
        AdaptiveSolver::new(system2, method, scheduler, tolerances).solve(t_end, guess_stepsize);
    let points1 =
        AdaptiveSolver::new(system1, method, scheduler, tolerances).solve(t_end, guess_stepsize);
    let points0 =
        AdaptiveSolver::new(system0, method, scheduler, tolerances).solve(t_end, guess_stepsize);
    let pointsm1 =
        AdaptiveSolver::new(systemm1, method, scheduler, tolerances).solve(t_end, guess_stepsize);
    let pointsm2 =
        AdaptiveSolver::new(systemm2, method, scheduler, tolerances).solve(t_end, guess_stepsize);

    let chart = Chart::new()
        .title(Title::new().text("ẋ = -x for different initial values (adaptive stepsize)"))
        .background_color("white")
        .x_axis(Axis::new().name("t").max(5.0))
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
    let system = System::new(t_start, &[2.0], &[ode]);

    // A low stepsize is recommended to make the the errors more visually obvious.
    let stepsize = 0.5;
    let points_euler = Solver::new(system.clone(), Euler::default(), stepsize).solve(t_end);
    let points_midpoint = Solver::new(system.clone(), Midpoint::default(), stepsize).solve(t_end);
    let points_rk4 = Solver::new(system.clone(), RungeKutta4::default(), stepsize).solve(t_end);

    // Smaller step size to see how methods scale
    let stepsize = 0.1;
    let points_euler_dense = Solver::new(system.clone(), Euler::default(), stepsize).solve(t_end);
    let points_midpoint_dense =
        Solver::new(system.clone(), Midpoint::default(), stepsize).solve(t_end);
    let points_rk4_dense = Solver::new(system, RungeKutta4::default(), stepsize).solve(t_end);

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

pub fn harmonic_oscillator(mass: f64, freq: f64, method: impl IntegrationMethod) {
    // x = (q, p), so x[0] = q and x[1] = p
    let q_dot = Rc::new(move |x: &[f64]| x[1] / mass);
    let p_dot = Rc::new(move |x: &[f64]| -mass * freq.powi(2) * x[0]);

    let q_start = 0.0;
    let p_start = 1.0;

    let t_start = 0.0;
    let t_end = 5.0;
    let stepsize = 0.1;

    let system = System::new(t_start, &[q_start, p_start], &[q_dot, p_dot]);
    let points = Solver::new(system, method, stepsize).solve(t_end);

    let chart = process_harmonic_oscillator(points, q_start, p_start, mass, freq);
    save_chart(&chart, "harmonic_oscillator", 1000, 1400);
}

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

    let system = System::new(t_start, &[q_start, p_start], &[q_dot, p_dot]);
    let points =
        AdaptiveSolver::new(system, method, scheduler, tolerances).solve(t_end, guess_stepsize);

    let chart = process_harmonic_oscillator(points, q_start, p_start, mass, freq);
    save_chart(&chart, "harmonic_oscillator_adaptive", 1000, 1400);
}

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
        .background_color("white");

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

    let system = System::new(t_start, &[theta_start, p_start], &[theta_dot, p_dot]);
    let points =
        AdaptiveSolver::new(system, method, scheduler, tolerances).solve(t_end, guess_stepsize);

    let mut chart = Chart::new()
        .title(Title::new().text(format!("Simple pendulum (theta0 = {theta_start}, p0 = {p_start}, mass = {m}, length = {l}, g = {g})")))
        .background_color("white");

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
        .grid(Grid::new().top("5%").height("42%"))
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
        .grid(Grid::new().bottom("5%").height("42%"))
        .x_axis(Axis::new().name("theta").grid_index(1).min(-1.2).max(1.2))
        .y_axis(Axis::new().name("p").grid_index(1).min(-1.2).max(1.2))
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
                .top("center"),
        );

    save_chart(&chart, "simple_pendulum_adaptive", 1000, 1400);
}

/// Convenience function to save a `charming::Chart` to disk.
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
