from typing import Generator
from ...utils.types import *
from ...utils.root_finder import RootFinder

from ..solution_method import SolutionMethod, weighted_sum


def backward_differential_formula(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
    y_weights: WeightArray,
    derivative_weight: float,
) -> Generator[SolutionPoint]:
    # Start by assuming that the previous y values came from a constant derivative
    first_derivative = derivative(t, y)
    previous_ys = to_vector_array([y - i * h * first_derivative for i in range(len(y_weights))], y.shape)
    while True:
        yield t, y.copy()

        t += h

        previous_ys_average = weighted_sum(previous_ys, y_weights)
        def deficit(next_y: Vector) -> Vector:
            return next_y - (previous_ys_average + derivative_weight * h * derivative(t, next_y))

        # Find the next y value that makes the deficit function zero
        y = RootFinder.vector(deficit, y)

        # Add the new y and move the rest backward one step
        previous_ys[1:] = previous_ys[:-1]
        previous_ys[0] = y


class BackwardDifferentialFormula(SolutionMethod):
    method = staticmethod(backward_differential_formula)
    compiled_method = None

    y_weights: WeightArray
    derivative_weight: float

    def __init__(self, y_weights, derivative_weight):
        """Method which takes a weighted sum of previous `y` and the current `f` value to approximate `y(t+h)`."""
        super().__init__()
        self._y_weights = y_weights
        self.derivative_weight = float(derivative_weight)

    def _validate(self):
        if self.validated:
            return

        self.y_weights = to_weight_array(self._y_weights)

        del self._y_weights
        self.validated = True

    def _prepare_arguments(self, derivative: DerivativeFunction, t0: Time, y0: Vector, h: Time, use_jit: bool):
        return (derivative, t0, y0, h, self.y_weights, self.derivative_weight)
