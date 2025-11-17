import numpy as np
import matplotlib.pyplot as plt

from ODEplotter import to_vector, ODE, SolutionAnimator
from ODEplotter.utils.types import Axes3D, Line3D


plt.rc('axes3d', mouserotationstyle='azel')


def cross_prod(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return np.array((a2*b3 - b2*a3, a3*b1 - b3*a1, a1*b2 - b1*a2))


class Curve:
    def __init__(self, start_pos, tangent, normal, curvature, torsion):
        self.start_pos = to_vector(start_pos)
        self.tangent = (vec := to_vector(tangent)) / np.linalg.norm(vec)
        self.normal = (vec := to_vector(normal)) / np.linalg.norm(vec)
        self.curvature = curvature
        self.torsion = torsion
        self.y0 = to_vector(np.concatenate((self.start_pos, self.tangent, self.normal)))

    def derivative(self, t, y):
        pos, tangent, normal = y[:3], y[3:6], y[6:]
        curvature = self.curvature(t)
        torsion = self.torsion(t)

        pos_diff = tangent
        tangent_diff = curvature * normal
        normal_diff = -curvature * tangent + torsion * cross_prod(tangent, normal)
        return np.concatenate((pos_diff, tangent_diff, normal_diff))

    def get_line(self, ax, style='o-', markevery=[-1], **kwargs) -> Line3D:
        line, = ax.plot([self.start_pos[0]], [self.start_pos[1]], [self.start_pos[2]], style, markevery=markevery, **kwargs)
        return line  # type: ignore


# Problem and solution
curve = Curve((0, 0, 0), (1, 0, 0), (0, 1, 0), lambda t: np.sqrt(1.0 + np.sin(t)), lambda t: np.sin(np.cos(t)))
ode = ODE(curve.derivative)
sol = ode.solve(0.0, curve.y0, 'RK4', 0.1)

# Figure and Axes
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
xlim = ylim = zlim = (-3, 3)
ax.set(xlim=xlim, ylim=ylim, zlim=zlim, aspect='equal')

# Line
line = curve.get_line(ax)

# Plotting/animating
animator = SolutionAnimator()
animator.animate(sol, SolutionAnimator.phase_diagram_updater_3d(line), SolutionAnimator.line_resetter_3d(line))
animator.animate_time_text(ax)

# Rendering
animator.render(fig, 0.0, np.inf, speed=5.0)
plt.show(block=True)
