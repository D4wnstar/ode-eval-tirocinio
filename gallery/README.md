These folders contain the plots created by the examples defined in `src/examples.rs`. `images` contains PNGs of the plots, whereas `interactive` contains HTML files that can be opened in a browser to interact with the data. `animations` contains videos animating some of the plots. `data` includes raw data from some examples.

The `charming` Rust library, which is used to generate most of these plots, is not particularly feature complete when it comes to 3D plots, so some plots are generated in Julia using the `GLMakie` package using data exported from the Rust examples. To run the Julia file, add the `GLMakie` and `Serde` packages.

Although ideally plots should be understable even without much context, the following are short explanations for each one, given in the order the examples were made in (and thus in increasing complexity).

### Exponential decay
This is a solution of a few starting conditions of the differential equation $\dot{x} = -x$ for, which is the exponential decay dynamics as found in, for instance, radioactive decay. This was mostly just a test to see if the basic methods were working, since the equation is very easy to solve manually.

### Exponential decay (adaptive)
Largely the same as above, but to test if the adaptive stepsize adjustment on the `DormandPrince54` method was working, especially in the edge case of a constant $x(t) = 0$ solution given by the $x(0)=0$ starting condition.

### Method comparison
A comparison of a few non-adaptive solvers at two different stepsizes to confirm that they progressively get more precise when all other variables are kept constant. Uses $\dot{x}=-x$ again for simplicity.

### Harmonic oscillator
A plot of the behavior of the classical, one-dimensional harmonic oscillator. The upper plot shows the behavior of the two simulated variables, $q$ and $p$, alongside calculations for kinetic, potential and total energy to prove that conserved quantities (total energy) are actually conserved. The bottom plot shows the phase portrait of the oscillator, which as we expect is an ellipse.

The equations used are the Hamilton equations for the Hamiltonian

$$H=\frac{p^{2}}{2m}+ \frac{1}{2}m\omega ^{2}q^{2}$$

which yield

$$\dot{q}=\frac{p}{m},\qquad\dot{p} = -m\omega^{2}q$$

### Harmonic oscillator (adaptive)
Same as above, but using `DormandPrince54` instead of `RungeKutta4`. In fact, all the following plots use `DormandPrince54`.

### Harmonic oscillator (interpolated)
A test to see if the built-in interpolation in `DormandPrince54` actually works. The simulated points are shown as a scatter plot and the interpolated points are shown with a line plot. Note that in a real scenario you'd probably combine the interpolated points with the simulated ones instead of keeping them separate like here.

### Simple pendulum (adaptive)
Like Harmonic oscillator (adaptive), but for the simple pendulum, including the energy conservation test and the phase portrait. The Hamiltonian used is in polar coordinates

$$H=\frac{p^{2}_{\theta}}{2mL^{2}}+ mgL(1-\cos \theta)$$

which yields the equations of motion

$$\dot{\theta}=\frac{p_{\theta}}{mL^{2}},\qquad \dot{p}_{\theta}=-mgL\sin \theta$$

### Simple pendulum (against small swings)
A comparison between the (real) simulated pendulum and the small-swing approximation $\sin\theta\simeq\theta$. Essentially a comparison between the pendulum and the harmonic oscillator.

### Simple pendulum (comparison)
Solutions of the simple pendulum for a variety of starting conditions, mostly different angles with no initial push, except for one to showcase full swings around the axis. Both the $\theta_{0}=0$ and $\theta_{0}=\pi$ starting conditions can be seen to be points of equilibrium, and the method (`DormandPrince54`) is sufficiently stable to show that even unstable equilibrium points are correctly simulated. This is actually only true with sufficiently precise constants: an initial test of this plot with $\pi=3.1415$ lead to the unstable point $\theta_{0}=\pi$ falling off and almost doing a full swing. This is likely because the method is sufficiently precise to "understand" the difference between *exactly* on the equilibrium point and *almost* on the equilibrium point. Switching to the built-in Rust constant `std::f64::const::PI`, which has over 15 significant digits, fixed the issue.

Also, the purple line plot has long straight lines in it because I made the angle loop between $-\pi$ and $\pi$, so that caused a "discontinuity" in the plot.

### Elastic pendulum (comparison)
Easily the most complicated and crowded plot here. I recommend opening the interactive version in a browser to hide some data by clicking on the legend at the bottom.

This is a pendulum with a spring of elastic constant $k$ and rest length $L_{0}$ instead of a rigid rod. This adds an extra degree of freedom to the system since the radial component $r$ is free to oscillate due to spring retention, but most importantly the two coordinates are coupled. The Hamiltonian used is

$$H=\frac{p_{\theta}^{2}}{2mr^{2}}+ \frac{p_{r}^{2}}{2m}+mg(L_{0}-r\cos \theta)+ \frac{k}{2}(L_{0}-r)^{2}$$

with equations of motion

```math
\left\{\begin{align*}
\dot{r}&=\frac{p_{r}}{m} \\
\dot{\theta}&=\frac{p_{\theta}}{mr^{2}} \\
\dot{p}_{r}&=\frac{p_{\theta}^{2}}{mr^{3}}+mg\cos \theta+k(L_{0}-r) \\
\dot{p}_{\theta}&=-mgr\sin \theta
\end{align*}\right.
```

Despite the relative simplicity, this system can exhibit chaotic behavior for sufficiently large initial conditions (either position, momentum or both). The top plots display the behavior of $\theta$ and $r$ over time. For $\theta$, outside of the correct equilibrium point (for $\theta$ only) in $\theta_{0}=0$, the remaining function shapes are not easily described with any analytic solution. The behavior of $r$ is that of a harmonic oscillator when $\theta_{0}=0$ (since it reduces to a simple oscillating spring) but quickly diverges for higher energies.

The bottom right plot is the phase portrait of the angular part of the system. It creates this interesting "square spiral" pattern for pretty much all starting conditions. The bottom right plot is arguably the easiest to interpret since it's the trajectory in Cartesian space, so basically the "real world". It's quite easy to see that the trajectories get very complicated very fast, with manifestly aperiodic motion making the behavior really difficult to predict even though it's "just" a pendulum.

### Heat equation
A simple test case for the PDE solver. It's the classic heat equation

$$\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}$$

No image graph is provided for this example since `charming` lacks proper 3D graph support. An interactive version is available.

### Schrödinger equation
The time dependent Schrödinger equation is the cornerstone of quantum mechanics. In one spatial dimension it reads

$$i\hbar\frac{\partial\psi}{\partial t}=-\frac{\hbar}{2m}\frac{\partial^2\psi}{\partial x^2}+V\psi$$

It can be nondimensionalized ($\hbar=m=1$) to read

$$i\frac{\partial\psi}{\partial t}=-\frac{1}{2}\frac{\partial^2\psi}{\partial x^2}+V\psi$$

The point is actually simulating this. It's a second order complex partial differential equation that results in the wave function $\psi$. This library uses the method of lines (MOL) to numerically evaluate PDEs by discretizing space into a grid of $m$ points $x_j$ and using finite central differences to reduce the time-and-space PDE into a system of time-only ODEs, one for each grid point. The second order finite central difference used to approximate the second derivative is

$$\frac{\partial^2\psi_j}{\partial x^2}\simeq  \frac{\psi_{j+1}-2\psi_{j}+\psi_{j-1}}{\Delta x^2}$$

where $j$ denotes the $j$-th grid point $x_j$, $\psi_j(t)\equiv\psi(x_j,t)$ is the wavefunction at that point and $\Delta x$ is the (constant) grid spacing. The Schrödinger equation then becomes

$$i\frac{\partial\psi_j}{\partial t}=- \frac{\psi_{j+1}-2\psi_{j}+\psi_{j-1}}{2\Delta x^2}+ V_{j}\psi_j$$

where $V_j\equiv V(x_j)$. This is a system of $m$ first order coupled *ordinary* differential equations for $\psi_j$ and as such, it can be solved with the usual methods for ODE system solving. In essence, we have $m$ ODEs that represent the dynamics of each individual grid point. By numerically evaluating many grid points, we can find many trajectories over time (lines, hence the name) from which we can interpolate the rest of the PDE.

For this example, a quartic potential was chosen:

$$V(x)=\frac{x^4}{2}$$

The simulated equation is

$$\frac{\partial\psi_j}{\partial t}=i\frac{\psi_{j+1}-2\psi_{j}+\psi_{j-1}}{2\Delta x^2}- i\frac{x_j^4}{2}\psi_j$$

The ODE system was solved with Runge-Kutta 4, which isn't ideal for this kind of system, but with a small enough time step, it gets the job done. Since `charming` doesn't provide good 3D plot support, the data from this example is exported to the `data` folder and is then plotted in `Julia` using `GLMakie` with the `make_plots.jl` script. Three animations are provided in the `animations` folder: `schrodinger_lines.webm` shows a 3D view of the simulated lines over time, space and probability; `schrodinger_surface.webm` is the same view, but interpolating to make a solid surface instead; `schrodinger_time_evo.webm` is a 2D view of the data showing time evolution as an animation. Furthermore, an additional `schrodinger_conservation.png` plot is available in the `images` folder which shows that normalization and energy are both constant in time (and that normalization is one).
