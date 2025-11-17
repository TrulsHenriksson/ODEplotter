import numpy as np
import matplotlib.pyplot as plt

from ODEplotter import ODE, SolutionAnimator


PENDULUM_LENGTH = 2.0
GRAVITY = 10.0

def pendulum_derivative(t, y):
    angle, angular_velocity = y
    angular_acceleration = -GRAVITY / PENDULUM_LENGTH * np.sin(angle)
    return np.array([angular_velocity, angular_acceleration])

def lines_updater(pendulum_line, trail_line, trail_length=0):
    def update_lines(ts, ys):
        if not len(ts):
            return ()
        angles = ys[:, 0]
        # Update pendulum line
        pendulum_line.set_xdata([0.0, np.sin(angles[-1]) * PENDULUM_LENGTH])
        pendulum_line.set_ydata([0.0, -np.cos(angles[-1]) * PENDULUM_LENGTH])
        # Update trail line
        old_trail_xdata, old_trail_ydata = trail_line.get_data()
        trail_line.set_xdata(np.concatenate((old_trail_xdata, np.sin(angles) * PENDULUM_LENGTH))[-trail_length:])
        trail_line.set_ydata(np.concatenate((old_trail_ydata, -np.cos(angles) * PENDULUM_LENGTH))[-trail_length:])
        return pendulum_line, trail_line        
    return update_lines

def lines_resetter(pendulum_line, trail_line):
    def reset_lines():
        pendulum_line.set_xdata([])
        pendulum_line.set_ydata([])
        trail_line.set_xdata(np.array([]))
        trail_line.set_ydata(np.array([]))
        return pendulum_line, trail_line
    return reset_lines


# Problem definition
ode = ODE(pendulum_derivative)

# Figure and Axes
fig, ax = plt.subplots(figsize=(5, 5))
lim = 2.5
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')

# Lines to animate
pendulum_line, = ax.plot([0.0, 0.0], [0.0, 0.0], 'o-k', linewidth=2, markevery=[-1], zorder=2)
trail_line, = ax.plot(np.array([]), np.array([]), '-', linewidth=1, color='gray', zorder=1)

# Numerical solution
t0 = 0.0
y0 = np.array([np.radians(170), 0.0])
sol = ode.solve(t0, y0, 'RK4', 0.01)

# Animation
animator = SolutionAnimator()
animator.animate(sol, lines_updater(pendulum_line, trail_line, trail_length=50), lines_resetter(pendulum_line, trail_line))
animator.animate_time_text(ax)

# Rendering
animator.render(fig, t0, 10.0)
plt.show(block=True)