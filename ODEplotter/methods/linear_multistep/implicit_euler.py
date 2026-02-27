from typing import Generator, Callable
from ...utils.types import *
from ...utils.root_finder import RootFinder

from ..solution_method import SolutionMethod


def implicit_eulers_method(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
    corrector: Callable[[Callable[[Vector], Vector], Vector], Vector],
) -> Generator[SolutionPoint]:
    while True:
        yield t, y.copy()
        last_derivative = derivative(t, y)
        # Next t
        t += h
        
        # Predict next y value using Euler's method (AB1)
        next_y_guess = y + h * last_derivative
        
        # Find the next y value that makes the deficit function zero
        deficit = lambda next_y: next_y - y - h * derivative(t, next_y)
        y = corrector(deficit, next_y_guess)


class ImplicitEulersMethod(SolutionMethod):
    method = staticmethod(implicit_eulers_method)
    compiled_method = None

    corrector: Callable[[Callable[[Vector], Vector], Vector], Vector]  # (y -> error, y0) -> minimizer

    def __init__(self, corrector: str = "newton"):
        """Method that solves `y(t+h) - y(t) = h * f(t+h, y(t+h))` for `y(t+h)`."""
        super().__init__()
        self._corrector = corrector

    def _validate(self):
        if self.validated:
            return

        self.corrector = RootFinder.methods[self._corrector]

        del self._corrector
        self.validated = True

    def _prepare_arguments(self, derivative: DerivativeFunction, t0: Time, y0: Vector, h: Time, use_jit: bool):
        return (derivative, t0, y0, h, self.corrector)
