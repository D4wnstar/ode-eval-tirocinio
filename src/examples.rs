use std::time::Instant;

use charming::{
    Chart, HtmlRenderer, ImageFormat, ImageRenderer,
    component::{Axis, Grid, Legend, Title},
    series::Line,
};

use crate::solvers::{Euler, IntegrationMethod, Midpoint, RungeKutta4, Solver, System};

/// Simple example of the exponential decay ODE ẋ = -x over a variety of initial values.
/// Integration method can be chosen.
pub fn exponential_decay(method: impl IntegrationMethod + Copy) {
    let ode = |x: f64| -x;
    let t_start = 0.0;
    let t_end = 5.0;
    let stepsize = 0.1;

    let system2 = System::new(ode, t_start, 2.0);
    let system1 = System::new(ode, t_start, 1.0);
    let system0 = System::new(ode, t_start, 0.0);
    let systemm1 = System::new(ode, t_start, -1.0);
    let systemm2 = System::new(ode, t_start, -2.0);

    let points2 = Solver::new(system2, t_end, stepsize, method).solve();
    let points1 = Solver::new(system1, t_end, stepsize, method).solve();
    let points0 = Solver::new(system0, t_end, stepsize, method).solve();
    let pointsm1 = Solver::new(systemm1, t_end, stepsize, method).solve();
    let pointsm2 = Solver::new(systemm2, t_end, stepsize, method).solve();

    let chart = Chart::new()
        .title(Title::new().text("ẋ = -x for different initial values"))
        .background_color("white")
        .x_axis(Axis::new().name("t").max(5.0))
        .y_axis(Axis::new().name("x"))
        .series(Line::new().data(points2))
        .series(Line::new().data(points1))
        .series(Line::new().data(points0))
        .series(Line::new().data(pointsm1))
        .series(Line::new().data(pointsm2));

    save_chart(&chart, "exponential_decay", 1000, 800);
}

/// A comparison of several integration methods on the exponential decay ODE ẋ = -x.
pub fn method_comparison() {
    let ode = |x: f64| -x;
    let t_start = 0.0;
    let t_end = 5.0;
    let system = System::new(ode, t_start, 2.0);

    // A low stepsize is recommended to make the the errors more visually obvious.
    let stepsize = 0.5;
    let points_euler = Solver::new(system.clone(), t_end, stepsize, Euler::default()).solve();
    let points_midpoint = Solver::new(system.clone(), t_end, stepsize, Midpoint::default()).solve();
    let points_rk4 = Solver::new(system.clone(), t_end, stepsize, RungeKutta4::default()).solve();

    // Smaller step size to see how methods scale
    let stepsize = 0.1;
    let points_euler_dense = Solver::new(system.clone(), t_end, stepsize, Euler::default()).solve();
    let points_midpoint_dense =
        Solver::new(system.clone(), t_end, stepsize, Midpoint::default()).solve();
    let points_rk4_dense = Solver::new(system, t_end, stepsize, RungeKutta4::default()).solve();

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
        .series(Line::new().data(points_euler).name("Euler"))
        .series(Line::new().data(points_midpoint).name("Midpoint"))
        .series(
            Line::new()
                .data(points_rk4)
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
                .data(points_euler_dense)
                .name("Euler")
                .x_axis_index(1)
                .y_axis_index(1),
        )
        .series(
            Line::new()
                .data(points_midpoint_dense)
                .name("Midpoint")
                .x_axis_index(1)
                .y_axis_index(1),
        )
        .series(
            Line::new()
                .data(points_rk4_dense)
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
