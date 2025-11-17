import numpy as np
import matplotlib.pyplot as plt

from ODEplotter import to_vector, ODE, SolutionAnimator


type SimpleTwoVector = tuple[float, float]


def perpendicular(vec):
    return (-vec[1], vec[0])

def lengths_squared(positions):
    return positions[..., 0] ** 2 + positions[..., 1] ** 2


class FixedPoint:
    def __init__(self, pos, jacobian, attracting):
        self.pos = to_vector(pos)
        self.jacobian = jacobian
        self.attracting = attracting


class Knot(FixedPoint):
    def __init__(self, pos, axis1, axis2, speed1, speed2):
        eigenvalues = (speed1, speed2)
        eigenvectors = np.column_stack((axis1, axis2))
        jacobian = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
        super().__init__(pos, jacobian, speed1 < 0.0 and speed2 < 0.0)

class Saddle(Knot):
    def __init__(self, pos, in_axis, out_axis, in_speed, out_speed):
        super().__init__(pos, in_axis, out_axis, -abs(in_speed), abs(out_speed))


class Spiral(FixedPoint):
    def __init__(self, pos, stability, speed):
        jacobian = np.array(((stability, -speed), (speed, stability)), dtype=np.float64)
        super().__init__(pos, jacobian, stability < 0.0)

class Center(Spiral):
    def __init__(self, pos: tuple[float, float], speed):
        super().__init__(pos, 0.0, speed)


class ImproperKnot(FixedPoint):
    def __init__(self, pos, out_axis, skew, speed):
        basis = np.column_stack((out_axis, perpendicular(out_axis)))
        basis_representation = np.array(((speed, skew), (0.0, speed)))
        jacobian = basis @ basis_representation @ np.linalg.inv(basis)
        super().__init__(pos, jacobian, speed < 0.0)

class Star(ImproperKnot):
    def __init__(self, pos, speed):
        super().__init__(pos, (1.0, 0.0), 0.0, speed)


class PlanarFlow:
    def __init__(self, fixed_points: list[FixedPoint]):
        self.fixed_points = fixed_points
        self.fixed_point_positions = np.array([point.pos for point in self.fixed_points])
        self.count = len(self.fixed_points)

        self.midpoint = self.fixed_point_positions.mean(axis=0)
        self.radius = np.linalg.norm(self.fixed_point_positions - self.midpoint, axis=1).max()

        # https://www.desmos.com/calculator/zswnekcch3
        # Each fixed points adds two to the degree of the polynomial of the derivative (I think), so compensate
        self.dampening_power = 2 * self.count
        self.dampening_factor = lambda radii_from_midpoint: 1 + radii_from_midpoint**self.dampening_power * (radii_from_midpoint - 1)

        self.off_diagonal = np.identity(self.count) == 0.0
        self.fixed_point_distances = lengths_squared(self.fixed_point_positions[:, None] - self.fixed_point_positions[None, :])
        self.normalization_constants = np.prod(self.fixed_point_distances, axis=1, where=self.off_diagonal)

    def derivative(self, t, pos):
        diff = np.zeros(2)
        local_dampening_factor = self.dampening_factor(np.linalg.norm(pos - self.midpoint) / (self.radius * 10)) if self.count > 1 else 1.0

        # Add up the jacobians from each fixed point, weighted so that at each fixed point only that one's jacobian is taken
        for i, fixed_point in enumerate(self.fixed_points):
            all_but_i = self.off_diagonal[i]
            distances_to_other_fixed_points = lengths_squared(self.fixed_point_positions[all_but_i] - pos)
            local_field_strength = np.prod(distances_to_other_fixed_points) / self.normalization_constants[i]
            diff += local_field_strength / local_dampening_factor * fixed_point.jacobian @ (pos - fixed_point.pos)

        return to_vector(diff)

    def get_lines_and_initial_conditions(self, ax, num_points_per_fixed_point, variance=1.0, **plot_kwargs):
        lines = ax.plot(*([] for _ in range(2 * num_points_per_fixed_point * self.count)), marker='.', markevery=[-1], **plot_kwargs)
        # Generate points around (0, 0)
        points = np.random.multivariate_normal(np.zeros(2), np.identity(2)*variance, num_points_per_fixed_point * self.count)
        # Move them to be centered on the fixed points
        for i in range(self.count):
            points[i::self.count] += self.fixed_point_positions[i]
        return lines, points


# Problem definition
planar_flow = PlanarFlow([
    Star(pos=(-1, 0), speed=-1.0),
    Star(pos=(1, 0), speed=-1.0),
    Star(pos=(0, 2), speed=1.0),
    Star(pos=(0, -1), speed=1.0),
])
ode = ODE(planar_flow.derivative)

# Figure and axes
radius = 5
fig, ax = plt.subplots()
ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect='equal')


def animate():
    global animator
    # Initial conditions
    t0 = 0.0
    solutions_per_fixed_point = 10
    lines, initial_conditions = planar_flow.get_lines_and_initial_conditions(ax, solutions_per_fixed_point, variance=0.1, linewidth=0.5)

    # Animate
    animator = SolutionAnimator()
    for line, y0 in zip(lines, initial_conditions):
        sol = ode.solve(t0, y0, 'RKF', 0.01, tol=1e-8)
        animator.animate(sol, SolutionAnimator.phase_diagram_updater(line, xcoord=0, ycoord=1), SolutionAnimator.line_resetter(line))
    animator.animate_time_text(ax)

    # Render
    animator.render(fig, t0, 20.0)

def plot_vector_field():
    ode.draw_vector_field(0.0, np.zeros(2), xcoord=0, ycoord=1, ax=ax, density=50, scale=0.0)


plot_vector_field()
animate()
plt.show(block=True)
