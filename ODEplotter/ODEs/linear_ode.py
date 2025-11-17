import numpy as np

from typing import Callable
from ..utils.types import *

from ..obstacles import Obstacle
from .ode import ODE


type CoefficientFunction = Callable[[Time], float]


class LinearODE(ODE):
    def __init__(
        self,
        *coefficients: CoefficientFunction | float,
        right_hand_side: CoefficientFunction | float = 0.0,
        obstacles: list[Obstacle] | None = None,
    ):
        r"""Define the ODE $ y^{(n+1)}(t) + c_0(t)y(t) + c_1(t)y'(t) + \cdots + c_n(t)y^{(n)}(t) = f(t) $.
        
        A `y` vector given to `LinearODE.derivative` is assumed to be on the form
        `[y(t), y'(t), y''(t), ..., y^(n)(t)]`.

        Arguments
        ---------
        coefficients : tuple[((Time) -> float) or float, ...]
            Tuple of coefficients. Each coefficient can either be fixed (a float)
            or dependent on `t` (a function `(Time) -> float`). The i'th coefficient
            is multiplied by the i'th derivative of `y`.
        right_hand_side : ((Time) -> float) or float
            The value of the linear operator defined by `coefficients`, either 
            fixed or dependent on `t`.
        obstacles : list[Obstacle] or None (default: None)
            List of Obstacles that the solver should respect.
        """
        if not all(isinstance(coeff, (float, Callable)) for coeff in coefficients):
            raise TypeError("coefficients must all be either floats or functions")
        if not isinstance(right_hand_side, (float, Callable)):
            raise TypeError("right_hand_side must be a function or float")

        self.coefficients = coefficients
        self.right_hand_side = right_hand_side

        # Define the derivative function
        def derivative(t: Time, y: Vector) -> Vector:
            evaluated_coefficients = np.array([
                coeff(t) if callable(coeff) else coeff for coeff in self.coefficients
            ])
            highest_derivative = (
                self.right_hand_side(t) if callable(self.right_hand_side) else self.right_hand_side
            ) - evaluated_coefficients.dot(y)
            return np.concat((y[1:], [highest_derivative]))

        super().__init__(derivative, obstacles)
