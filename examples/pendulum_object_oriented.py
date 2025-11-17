import numpy as np
import matplotlib.pyplot as plt

from ODEplotter import ODE, SolutionAnimator


class Pendulum(ODE):
    def __init__(self, ax, length=1.0, gravity=10.0):
        self.length = length
        self.gravity = gravity
        self.pendulum_line, = ax.plot([0.0, 0.0], [0.0, 0.0], 'o-k', linewidth=2, markevery=[-1], zorder=2)
        self.trail_line, = ax.plot(np.array([]), np.array([]), '-', linewidth=1, color='gray', zorder=1)
        super().__init__(self.derivative)

    def derivative(self, t, y):
        angle, angular_velocity = y
        angular_acceleration = -self.gravity / self.length * np.sin(angle)
        return np.array([angular_velocity, angular_acceleration])

    def lines_updater(self, trail_length=0):
        def update_lines(ts, ys):
            if not len(ts):
                return ()
            angles = ys[:, 0]
            # Update pendulum line
            self.pendulum_line.set_xdata([0.0, np.sin(angles[-1]) * self.length])
            self.pendulum_line.set_ydata([0.0, -np.cos(angles[-1]) * self.length])
            # Update trail line
            old_trail_xdata, old_trail_ydata = self.trail_line.get_data()
            self.trail_line.set_xdata(np.concatenate((old_trail_xdata, np.sin(angles) * self.length))[-trail_length:])
            self.trail_line.set_ydata(np.concatenate((old_trail_ydata, -np.cos(angles) * self.length))[-trail_length:])
            return self.trail_line, self.pendulum_line
        return update_lines

    def lines_resetter(self):
        def reset_lines():
            self.pendulum_line.set_xdata([])
            self.pendulum_line.set_ydata([])
            self.trail_line.set_xdata(np.array([]))
            self.trail_line.set_ydata(np.array([]))
            return self.trail_line, self.pendulum_line
        return reset_lines


# Figure and Axes
fig, ax = plt.subplots(figsize=(5, 5))
lim = 2.5
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')

# Problem definition
pendulum = Pendulum(ax, length=2.0)

# Numerical solution
t0 = 0.0
y0 = np.array([np.radians(170), 0.0])
sol = pendulum.solve(t0, y0, 'RK4', 0.01)

# Animation
animator = SolutionAnimator()
animator.animate(sol, pendulum.lines_updater(trail_length=50), pendulum.lines_resetter())
animator.animate_time_text(ax)

# Rendering
animator.render(fig, t0, 20.0)
plt.show(block=True)