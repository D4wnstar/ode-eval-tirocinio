# Requires GLMakie and Serde
println("Loading packages")
using GLMakie
using Serde

json = read("data/schrodinger_equation.json", String)
data = deser_json(Vector{Vector{Tuple{Float64, Float64, Float64}}}, json)

# Make the line plot
f = Figure(size=(600, 600))
ax = Axis3(
	f[1, 1],
	title=L"Schrödinger equation, $V(x) = \frac{x^4}{2}$, starting state $\psi(x,0)=\left(\frac{2}{\sigma\pi}\right)^{1/4}e^{-x^2/\sigma}$",
	xlabel=L"$t$ (Time)",
	ylabel=L"$x$ (Space)",
	zlabel=L"$|\psi^2|$ (Probability)",
	protrusions=(60, 60, 30, 30),
	xreversed=true,
)

for ode in data
	lines!(f[1, 1], ode, color=:black)
end

println("Making Schrödinger equation animation (lines)")
record(
	f,
	"animations/schrodinger_lines.webm",
	range(0, 2π, length=120);
	framerate=20,
) do azim
	ax.azimuth = azim
end

# Make the surface plot
time = map(point -> point[1], data[1])
space = map(ode -> ode[1][2], data)
psi = zeros(length(time), length(space))
for t_i in eachindex(time)
	for x_i in eachindex(space)
		psi[t_i, x_i] = data[x_i][t_i][3]
	end
end

f = Figure(size=(600, 600))
ax = Axis3(
	f[1, 1],
	title=L"Schrödinger equation, $V(x) = \frac{x^4}{2}$, starting state $\psi(x,0)=\left(\frac{2}{\sigma\pi}\right)^{1/4}e^{-x^2/\sigma}$",
	xlabel=L"$t$ (Time)",
	ylabel=L"$x$ (Space)",
	zlabel=L"$|\psi^2|$ (Probability)",
	protrusions=(60, 60, 30, 30),
	xreversed=true,
)
surface!(f[1, 1], time, space, psi)

println("Making Schrödinger equation animation (surface)")
record(
	f,
	"animations/schrodinger_surface.webm",
	range(0, 2π, length=120);
	framerate=20,
) do azim
	ax.azimuth = azim
end

# Make the 2D animation
println("Making Schrödinger equation animation (time evolution)")
points = Observable([(0.0, 0.0)])
f = Figure(size=(600, 600))
ax = Axis(
	f[1, 1],
	title=L"Schrödinger equation, $V(x) = \frac{x^4}{2}$, starting state $\psi(x,0)=\left(\frac{2}{\sigma\pi}\right)^{1/4}e^{-x^2/\sigma}$",
	xlabel=L"$x$ (Space)",
	ylabel=L"$|\psi^2|$ (Probability)",
	limits=(-5, 5, 0, 1),
)
lines!(f[1, 1], points)

frames = round(Int, length(time) / 100)
record(f, "animations/schrodinger_time_evo.webm", 0:frames; framerate=20) do frame
	points[] = map(ode -> ode[frame*100+1][2:3], data)
end

println("Done! 🎉")
