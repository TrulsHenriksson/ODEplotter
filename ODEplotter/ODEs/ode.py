import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Generator
from ..utils.types import *
from ..utils.exceptions import *
from ..methods import METHODS

from ..discrete_solution import DiscreteSolution
from ..obstacles import Obstacle



class ODE:
    def __init__(self, derivative: DerivativeFunction, obstacles: list[Obstacle] | None = None):
        """Define the Ordinary Differential Equation `y'(t) = f(t, y(t))` from the function `f`.

        No initial values are given in this step, so the ODE can be solved for any number of
        initial values later, using `ODE.solve`.

        Arguments
        ---------
        derivative : (Time, Vector) -> Vector
            The function that gives the change in `y` when called as `derivative(t, y)`.
        obstacles : list[Obstacle] or None (default: None)
            List of Obstacles that the solver should respect.
        """
        obstacles = [] if obstacles is None else obstacles
        if not callable(derivative):
            raise TypeError('derivative must be callable')
        if not isinstance(obstacles, list) or not all(isinstance(obstacle, Obstacle) for obstacle in obstacles):
            raise TypeError('obstacles must be a list of Obstacles')
        self.derivative = derivative
        self.obstacles = obstacles

    def solve(self, t0: Time, y0: Vector, method_name: str, *args, use_jit: bool = False, **kwargs) -> DiscreteSolution:
        """Lazily solves the given IVP using a given method and step size.

        The Initial Value Problem is defined by the ODE itself, the initial time `t0`, and the
        initial vector `y0`. This can then be solved using any available method with any positive
        step size `h`, and, in the case of adaptive-step methods, any positive tolerance `tol`.

        Arguments
        ---------
        t0 : Time
            Initial time.
        y0 : Vector
            Initial value vector.
        method_name : str
            Any method name in ``ODEplotter.METHODS``.
        use_jit : bool (default: False)
            Whether to Just-In-Time compile the derivative and solution methods. This takes
            extra time the first time `ODE.solve` is called, but speeds up subsequent calls
            significantly. Currently unavailable for implicit methods.
        
        **Arguments specific to fixed-step methods:**

        h : Time (usually < 0.1)
            Step size. This is constant during integration.

        **Arguments specific to adaptive-step methods:**

        h0 : Time (usually < 0.1)
            Initial step size. This is changed during integration, and is less impactful
            than `tol` (see below).
        tol : float (usually < 1e-3)
            Error tolerance. The method keeps the local error estimate below this bound.
        min_h : float (default: 1e-15)
            Minimum step size. An error is raised if the method attempts to lower the step
            size below this bound.

        Returns
        -------
        sol : DiscreteSolution
            DiscreteSolution instance with methods to load points (solve), interpolate between them,
            and plot the solution.
        """
        t0, y0 = to_time(t0), to_vector(y0)
        # Get h / h0 primarily from kwargs, then args
        if "h" in kwargs:
            kwargs["h"] = to_time(kwargs["h"])
        elif "h0" in kwargs:
            kwargs["h0"] = to_time(kwargs["h0"])
        else:
            h, *args = args
            args = (to_time(h), *args)

        restarter = self.__get_restarter(method_name, *args, use_jit=use_jit, **kwargs)
        if self.obstacles:
            point_gen = self.__obstacle_solver(restarter, t0, y0)
        else:
            point_gen = restarter(t0, y0)
        return DiscreteSolution(point_gen, method_name)

    def __get_restarter(self, method_name: str, *args, use_jit: bool = False, **kwargs):
        method = METHODS[method_name.lower()]
        restarter = lambda new_t0, new_y0: method.solve(self.derivative, new_t0, new_y0, *args, use_jit=use_jit, **kwargs)
        return restarter

    def solve_single(self, t0: Time, y0: Vector, method_name: str, h: Time, **kwargs) -> SolutionPoint:
        method = METHODS[method_name.lower()]
        point_gen = method.solve(self.derivative, t0, y0, h, use_jit=False, **kwargs)
        return next(point_gen)

    def __obstacle_solver(
        self, restarter: Callable[[Time, Vector], Generator[SolutionPoint]], t0: Time, y0: Vector
    ) -> Generator[SolutionPoint]:
        # TODO: Simplify the flow
        new_t0, new_y0 = t0, y0
        # Keep restarting point_gen when the simulation hits an obstacle
        restart = True
        while restart:
            # Restart the solution after hitting an obstacle
            point_gen = restarter(new_t0, new_y0)
            # Load the very first point
            prev_t, prev_y = next(point_gen)
            yield prev_t, prev_y
            for t, y in point_gen:
                # Handle collisions
                candidates = [obstacle.was_hit(prev_t, prev_y, t, y) for obstacle in self.obstacles]
                restart = any(candidates)
                if restart:
                    # [(obstacle, t_hit, y_hit), ...]
                    collisions = [
                        (obstacle, *obstacle.get_collision(prev_t, prev_y, t, y))
                        for obstacle, was_hit in zip(self.obstacles, candidates)
                        if was_hit
                    ]
                    # Take the first point that was hit (there might be multiple)
                    hit_obstacle, t_hit, y_hit = min(collisions, key=lambda c: c[1])
                    new_t0 = t_hit
                    try:
                        new_y0 = hit_obstacle.hit_function(t_hit, y_hit)
                    except ObstacleStopSolving:
                        restart = False
                    # Break to restart the point_gen
                    break
                # No obstacles were hit, proceed without restarting point_gen
                yield t, y
                prev_t, prev_y = t, y

    def draw_vector_field(
        self, t: Time, origin: Vector, xcoord=-1, ycoord=0, *, ax: Axes | None = None, density: int = 15, scale: float = 1.0
    ) -> tuple[Quiver, Callable[[float, float], tuple[float, float]]]:
        """Draw arrows illustrating the ODE as a vector field.

        Plot the change in `y[xcoord]` on the x-axis (or in `t` if `xcoord=-1`) and
        similarly for the y-axis. This can be seen as showing the slice centered on `(*origin, t)`,
        and extending in the coordinates specified by `xcoord`, `ycoord`.

        If scale is set to zero, all arrows have the same length.
        """
        if ax is None:
            ax = plt.gca()
        arrow_direction = self.__get_arrow_direction_function(xcoord, ycoord, t, origin)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xs = np.linspace(xmin, xmax, density)
        ys = np.linspace(ymin, ymax, density)
        xgrid, ygrid = np.meshgrid(xs, ys)
        dxs, dys = zip(*[arrow_direction(x, y) for y in ys for x in xs])
        dxs, dys = np.array(dxs), np.array(dys)
        if scale == 0.0:
            lengths = np.hypot(dxs, dys)
            lengths[lengths < 1e-15] = 1.0
            dxs /= lengths
            dys /= lengths
            scale = density / min(xmax - xmin, ymax - ymin)
        else:
            scale = 1.0 / scale
        quiver = ax.quiver(
            xgrid,
            ygrid,
            dxs,
            dys,
            pivot="mid",
            color="lightgray",
            angles="xy",
            scale=scale,
            scale_units="xy",
        )
        return quiver, arrow_direction

    def __get_arrow_direction_function(
        self, xcoord: int, ycoord: int, t: Time, origin: Vector
    ) -> Callable[[float, float], tuple[float, float]]:
        # Use origin as a template for the arguments to self.function
        values = np.concatenate((to_vector(origin), (to_time(t),)))  # type: ignore
        # Time is last in values
        TIME_COORD = -1
        def arrow_direction(plot_x, plot_y):
            # Take the copy of values and put the plot_x, plot_y values in
            values[xcoord] = plot_x
            values[ycoord] = plot_y
            t = values[TIME_COORD]
            y = values[:TIME_COORD]
            change = list(self.derivative(t, y)) + [1.0]  # Time derivative is always 1
            return float(change[xcoord]), float(change[ycoord])
        return arrow_direction



"""
TODO:
+ Move JitODE to its own file
+ Make all jitting lazy, or even precompiled? Investigate.
- Move to using faster root-finding methods for implicit methods, maybe scipy.optimize
"""
