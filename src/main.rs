use charming::{
    Chart, HtmlRenderer, ImageRenderer,
    component::{Axis, Title},
    series::Line,
};
use solvers::{Euler, Solver, System};

pub mod solvers;

fn main() {
    let ode = Box::new(|x: f64| -x);
    let t_start = 0.0;
    let t_end = 5.0;
    let stepsize = 0.1;

    let system2 = System::new(ode.clone(), t_start, 2.0);
    let system1 = System::new(ode.clone(), t_start, 1.0);
    let system0 = System::new(ode.clone(), t_start, 0.0);
    let systemm1 = System::new(ode.clone(), t_start, -1.0);
    let systemm2 = System::new(ode, t_start, -2.0);

    let points2 = Solver::new(system2, t_end, stepsize, Euler::default()).solve();
    let points1 = Solver::new(system1, t_end, stepsize, Euler::default()).solve();
    let points0 = Solver::new(system0, t_end, stepsize, Euler::default()).solve();
    let pointsm1 = Solver::new(systemm1, t_end, stepsize, Euler::default()).solve();
    let pointsm2 = Solver::new(systemm2, t_end, stepsize, Euler::default()).solve();

    let chart = Chart::new()
        .title(Title::new().text("dx/dt = -x"))
        .x_axis(Axis::new().name("t"))
        .y_axis(Axis::new().name("x"))
        .series(Line::new().data(points2))
        .series(Line::new().data(points1))
        .series(Line::new().data(points0))
        .series(Line::new().data(pointsm1))
        .series(Line::new().data(pointsm2));

    let mut renderer = HtmlRenderer::new("ODE Integration", 1000, 800);
    renderer
        .save(&chart, "gallery/interactive/exponential_decay.html")
        .unwrap();

    let mut image_renderer = ImageRenderer::new(1000, 800);
    image_renderer
        .save(&chart, "gallery/images/exponential_decay.svg")
        .unwrap();
}
