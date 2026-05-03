from typing import Generator
from ...utils.types import *
from ...utils.root_finder import RootFinder

from ..solution_method import SolutionMethod


def implicit_eulers_method(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
) -> Generator[SolutionPoint]:
    while True:
        yield t, y.copy()
        last_derivative = derivative(t, y)

        t += h
        next_y_guess = y + h * last_derivative  # AB1 (Euler's method)

        # Find the next y value that makes the deficit function zero
        def deficit(next_y: Vector) -> Vector:
            return next_y - y - h * derivative(t, next_y)

        y = RootFinder.vector(deficit, next_y_guess)


class ImplicitEulersMethod(SolutionMethod):
    method = staticmethod(implicit_eulers_method)
    compiled_method = None

    def __init__(self):
        """Method that solves `y(t+h) - y(t) = h * f(t+h, y(t+h))` for `y(t+h)`."""
        super().__init__()

    def _validate(self):
        # Nothing to validate
        self.validated = True

    def _prepare_arguments(self, derivative: DerivativeFunction, t0: Time, y0: Vector, h: Time, use_jit: bool):
        return (derivative, t0, y0, h)
