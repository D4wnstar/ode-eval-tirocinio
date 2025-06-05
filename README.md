An implementation of a numerical differential equation solver, written entirely in Rust, with support for a handful of solving methods. Currently implemented:
- Euler's method
- Midpoint method
- Fourth-order Runge-Kutta (RK4)
- Dormand-Prince 5(4) with adaptive stepsize and built-in interpolation

The `src/examples.rs` file contains several examples on well-known physical systems. Currently implemented:
- Exponential decay
- Harmonic oscillator
- Simple pendulum
- Elastic pendulum

Each example creates a plot in that is saved in the `gallery` folder. The `gallery/images` subfolder includes the plots as simple PNGs. The `gallery/interactive` subfolder provides HTML files that can be downloaded and opened in a browser (with an internet connection) to provide some basic interactivity with the plot, such as hovering over data points and hiding data by clicking on the legend icons. This is especially recommended for comparison plots to reduce overcrowding.
