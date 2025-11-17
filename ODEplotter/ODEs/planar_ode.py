import numpy as np

from typing import Callable
from ..utils.types import *

from ..obstacles import Obstacle
from .ode import ODE


type CoefficientFunction = Callable[[Time], float]


class PlanarODE(ODE):
    def __init__(
        self,
        dx: Callable[[Time, float, float], float] | float,
        dy: Callable[[Time, float, float], float] | float,
        obstacles: list[Obstacle] | None = None,
    ):
        r"""Define the system `x'(t) = dx(t, x, y), y'(t) = dy(t, x, y)`.
        
        A `state` vector given to `PlanarODE.derivative` is assumed to be on the form
        `[x(t), y(t)]`.

        Arguments
        ---------
        dx, dy : ((Time, x, y) -> float) or float
            Derivative in `x` and `y` respectively. Either fixed (floats) or dependent
            on the current time and `x, y` position (functions `(Time, x, y) -> float`).
        obstacles : list[Obstacle] or None (default: None)
            List of Obstacles that the solver should respect.
        """
        if not isinstance(dx, (float, Callable)) or not isinstance(dy, (float, Callable)):
            raise TypeError("dx and dy must be floats or functions `(Time, x, y) -> float`")
        self.dx = dx if callable(dx) else float(dx)
        self.dy = dy if callable(dy) else float(dy)

        # Define the derivative function
        def derivative(t: Time, state: Vector) -> Vector:
            x, y = float(state[0]), float(state[1])
            dx = self.dx(t, x, y) if callable(self.dx) else self.dx
            dy = self.dy(t, x, y) if callable(self.dy) else self.dy
            return np.array([dx, dy])

        super().__init__(derivative, obstacles)

    def draw_vector_field(self, t: Time, *, ax: Axes | None = None, density: int = 15, scale: float = 1.0):
        return super().draw_vector_field(
            t, origin=np.zeros(2), xcoord=0, ycoord=1, ax=ax, density=density, scale=scale
        )
