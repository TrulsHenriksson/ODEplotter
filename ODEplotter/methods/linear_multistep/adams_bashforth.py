from numba import jit
import numpy as np

from typing import Generator
from ...utils.types import *

from ..solution_method import SolutionMethod


def adams_bashforth(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
    weights: WeightArray,
    derivatives: VectorArray,
) -> Generator[SolutionPoint]:
    while True:
        yield t, y.copy()
        t += h
        y += h * weights.dot(derivatives)

        # Move the old diffs back one step and calculate the new one
        derivatives[1:] = derivatives[:-1]
        derivatives[0] = derivative(t, y)


class AdamsBashforth(SolutionMethod):
    method = staticmethod(adams_bashforth)
    compiled_method = staticmethod(jit(adams_bashforth))

    weights: WeightArray

    def __init__(self, weights):
        """Method which uses a weighted sum of previous `f` (derivative) values to approximate the next `y` value."""
        super().__init__()
        self._weights = weights

    def _validate(self):
        if self.validated:
            return

        self.weights = to_weight_array(self._weights)
        assert abs(self.weights.sum() - 1.0) < 1e-14, "weights must sum to 1"

        del self._weights
        self.validated = True

    def _prepare_arguments(self, derivative: DerivativeFunction, t0: Time, y0: Vector, h: Time, use_jit: bool):
        # Start with the previous derivatives all equal to the initial derivative
        # This is seemingly impossible to do inside a JITed function, that's why we do it here
        derivatives = np.array([derivative(t0, y0)] * len(self.weights))
        return (derivative, t0, y0, h, self.weights, derivatives)
