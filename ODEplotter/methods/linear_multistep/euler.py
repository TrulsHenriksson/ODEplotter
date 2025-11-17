from numba import jit

from typing import Generator
from ...utils.types import *

from ..solution_method import SolutionMethod


def eulers_method(derivative: DerivativeFunction, t: Time, y: Vector, h: Time) -> Generator[SolutionPoint]:
    while True:
        yield t, y.copy()
        t += h
        y += h * derivative(t, y)


class EulersMethod(SolutionMethod):
    method = staticmethod(eulers_method)
    compiled_method = staticmethod(jit(eulers_method))

    def __init__(self):
        """Method that takes `y(t) + h*f(t, y(t))` as the estimate for `y(t + h)`."""
        super().__init__()

    def _validate(self):
        # Nothing to validate
        self.validated = True

    def _prepare_arguments(self, derivative: DerivativeFunction, t0: Time, y0: Vector, h: Time, use_jit: bool):
        return (derivative, t0, y0, to_time(h))
