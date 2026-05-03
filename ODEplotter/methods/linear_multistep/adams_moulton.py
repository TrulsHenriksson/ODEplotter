import numpy as np

from typing import Callable, Generator
from ...utils.types import *
from ...utils.predictors import Predictors
from ...utils.root_finder import RootFinder

from ..solution_method import SolutionMethod, weighted_sum


# Non-JIT compilable method (TODO: fix by using a JITed predictor-corrector pair)
def adams_moulton(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
    weights: WeightArray,
    predictor: Callable[[Time, Vector, VectorArray], Vector],
    corrector: Callable[[Callable[[Vector], Vector], Vector], Vector],
) -> Generator[SolutionPoint]:
    first_weight = weights[0]
    # Start with the previous derivatives all equal to the initial diff
    derivatives = np.array([derivative(t, y)] * len(weights))
    while True:
        yield t, y.copy()

        t += h

        previous_derivatives_average = weighted_sum(derivatives[:-1], weights[1:])
        def deficit(next_y: Vector) -> Vector:
            first_derivative = derivative(t, next_y)
            return next_y - y - h * (first_weight * first_derivative + previous_derivatives_average)

        # Find the next y value that makes the deficit function zero
        next_y_guess = predictor(h, y, derivatives)
        y = corrector(deficit, next_y_guess)

        # Move the old derivatives back one step and calculate the new one
        derivatives[1:] = derivatives[:-1]
        derivatives[0] = derivative(t, y)


class AdamsMoulton(SolutionMethod):
    method = staticmethod(adams_moulton)
    compiled_method = None

    weights: WeightArray
    predictor: Callable[[Time, Vector, VectorArray], Vector]  # (h, y, derivatives) -> next_y_guess

    def __init__(self, weights, predictor: str):
        """Method which uses a weighted sum of several previous and the next `f` values to approximate `y(t+h)`."""
        super().__init__()
        self._weights = weights
        self._predictor = predictor

    def _validate(self):
        if self.validated:
            return

        self.weights = to_weight_array(self._weights)
        self.predictor = Predictors.methods[self._predictor]

        del self._weights, self._predictor
        self.validated = True

    def _prepare_arguments(self, derivative: DerivativeFunction, t0: Time, y0: Vector, h: Time, use_jit: bool):
        return (derivative, t0, y0, h, self.weights, self.predictor, RootFinder.vector)
