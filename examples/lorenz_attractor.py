import numpy as np
import matplotlib.pyplot as plt

from ODEplotter import ODE, SolutionAnimator


plt.rc("axes3d", mouserotationstyle="azel")


class LorenzAttractor(ODE):
    def __init__(self, rho=28.0, sigma=10.0, beta=8/3):
        self.fixed_points = np.array([
            [np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1],
            [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho - 1],
        ])
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

        def derivative(t, state):
            x, y, z = state
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return np.array((dx, dy, dz))
        
        super().__init__(derivative)

    def get_lines_and_initial_conditions(self, ax, num_points_per_fixed_point, variance=20.0, **plot_kwargs):
        lines = ax.plot(*([] for _ in range(3 * 2 * num_points_per_fixed_point)), marker=".", markevery=[-1], **plot_kwargs)
        
        oranges = plt.get_cmap("autumn")
        blues = plt.get_cmap("winter")
        for line in lines[0::2]:
            line.set_color(oranges(np.random.random() * 0.6))
        for line in lines[1::2]:
            line.set_color(blues(np.random.random() * 0.4 + 0.2))
        
        # Generate points around [0, 0, 0]
        points = np.random.multivariate_normal(np.zeros(3), np.identity(3) * variance, 2 * num_points_per_fixed_point)
        # Move them to be centered on the fixed points
        points[0::2] += self.fixed_points[0]
        points[1::2] += self.fixed_points[1]
        return lines, points

    def get_plot_limits(self, radius):
        return (-radius, radius), (-radius, radius), (self.rho - 1 - radius, self.rho - 1 + radius)


# Problem definition
lorenz = LorenzAttractor()

# Figure and Axes
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection="3d")  # type: ignore
xlim, ylim, zlim = lorenz.get_plot_limits(radius=20.0)
ax.set(xlim=xlim, ylim=ylim, zlim=zlim, aspect="equal")

# Initial conditions
t0 = 0.0
number_of_solutions = 10
lines, initial_conditions = lorenz.get_lines_and_initial_conditions(ax, number_of_solutions, variance=5, linewidth=0.5)

# Animation
animator = SolutionAnimator()
for line, y0 in zip(lines, initial_conditions):
    sol = lorenz.solve(t0, y0, "RK4", 0.01, use_jit=True)
    animator.animate(
        sol, SolutionAnimator.phase_diagram_updater_3d(line, trail_length=200), SolutionAnimator.line_resetter_3d(line)
    )
animator.animate_time_text(ax)

# Rendering
animator.render(fig, t0, np.inf, repeat=False)
plt.show(block=True)
