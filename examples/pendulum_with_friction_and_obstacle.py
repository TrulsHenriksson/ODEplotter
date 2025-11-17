import numpy as np
import matplotlib.pyplot as plt

from ODEplotter import ODE, SolutionAnimator, Obstacle


class Pendulum(ODE):
    def __init__(self, ax, length=1.0, gravity=10.0, friction=0.0, obstacles: list[Obstacle] = []):
        self.length = length
        self.gravity = gravity
        self.friction = friction
        self.pendulum_line, = ax.plot([0.0, 0.0], [0.0, 0.0], 'o-k', linewidth=2, markevery=[-1], zorder=2)
        self.trail_line, = ax.plot(np.array([]), np.array([]), '-', linewidth=1, color='gray', zorder=1)
        super().__init__(self.derivative, obstacles)

    def derivative(self, t, y):
        angle, angular_velocity = y
        angular_acceleration = -self.gravity / self.length * np.sin(angle) - self.friction * angular_velocity
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

class BouncyObstacle(Obstacle):
    def __init__(self, angle, elasticity=1.0):
        self.angle = angle % (2 * np.pi)
        self.elasticity = elasticity
        super().__init__(self.distance_function, self.reflect_velocity, one_sided=False)
    
    def distance_function(self, ts, ys):
        angles = ys[:, 0]
        # Smooth function that equals zero iff angle - self.angle == 0 (mod 2*pi)
        result = np.sin(0.5 * (angles - self.angle))
        return result
    
    def reflect_velocity(self, t_hit, y_hit):
        angle, angular_velocity = y_hit
        return np.array((self.angle, -self.elasticity * angular_velocity))
    
    def plot(self, ax, length, color='black', linewidth=3, **plot_kwargs):
        x, y = np.sin(self.angle) * length, -np.cos(self.angle) * length
        return ax.plot([x * 0.75, x * 1.25], [y * 0.75, y * 1.25], color=color, linewidth=linewidth, **plot_kwargs)[0]


# Figure and Axes
fig, ax = plt.subplots(figsize=(5, 5))
lim = 2.5
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')

# Problem definition
bouncy_obstacle = BouncyObstacle(angle=np.radians(180), elasticity=0.95)
pendulum = Pendulum(ax, length=2.0, friction=0.05, obstacles=[bouncy_obstacle])

# Numerical solution
t0 = 0.0
y0 = np.array([np.radians(0), 10.0])
sol = pendulum.solve(t0, y0, 'RK4', 0.01)

# Animation and plotting
bouncy_obstacle.plot(ax, pendulum.length)
animator = SolutionAnimator()
animator.animate(sol, pendulum.lines_updater(trail_length=20), pendulum.lines_resetter())
animator.animate_time_text(ax)

# Rendering
animator.render(fig, t0, np.inf)
plt.show(block=True)