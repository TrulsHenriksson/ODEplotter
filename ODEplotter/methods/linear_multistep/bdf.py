from numba import jit
import numpy as np

from typing import Callable, Generator
from ...utils.types import *

from ..solution_method import SolutionMethod
from ...root_finder import RootFinder


def backward_differential_formula(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
    y_weights: WeightArray,
    derivative_weight: float,
    corrector: Callable[[Callable[[Vector], Vector], Vector], Vector],
    deficit_getter,
) -> Generator[SolutionPoint]:
    # Start by assuming that the previous y values came from a constant derivative
    first_derivative = derivative(t, y)
    previous_ys = to_vector_array(y - h * first_derivative * np.arange(len(y_weights))[:, None])
    while True:
        yield t, y.copy()
        # Calculate the next y value by solving for deficit == 0
        deficit = deficit_getter(derivative, t, previous_ys, h, y_weights, derivative_weight)
        t += h
        y = corrector(deficit, y)
        # Add the new y and move the rest backward one step
        previous_ys[1:] = previous_ys[:-1]
        previous_ys[0] = y


def deficit_getter(
    derivative: DerivativeFunction,
    t: Time,
    previous_ys: VectorArray,
    h: Time,
    y_weights: WeightArray,
    derivative_weight: float,
) -> Callable[[Vector], Vector]:
    next_t = t + h
    previous_ys_average = y_weights.dot(previous_ys)
    def deficit(next_y: Vector) -> Vector:
        return next_y - (previous_ys_average + derivative_weight * h * derivative(next_t, next_y))
    return deficit


class BackwardDifferentialFormula(SolutionMethod):
    method = staticmethod(backward_differential_formula)
    compiled_method = None

    y_weights: WeightArray
    derivative_weight: float
    corrector: Callable[[Callable[[Vector], Vector], Vector], Vector]  # (y -> error, y0) -> minimizer

    def __init__(self, y_weights, derivative_weight, corrector: str = "newton"):
        """Method which takes a weighted sum of previous `y` and the current `f` value to approximate `y(t+h)`."""
        super().__init__()
        self._y_weights = y_weights
        self.derivative_weight = float(derivative_weight)
        self._corrector = corrector

    def _validate(self):
        if self.validated:
            return

        self.y_weights = to_weight_array(self._y_weights)
        self.corrector = RootFinder.methods[self._corrector]

        del self._y_weights, self._corrector
        self.validated = True

    def _prepare_arguments(self, derivative: DerivativeFunction, t0: Time, y0: Vector, h: Time, use_jit: bool):
        return (derivative, t0, y0, h, self.y_weights, self.derivative_weight, self.corrector, deficit_getter)
