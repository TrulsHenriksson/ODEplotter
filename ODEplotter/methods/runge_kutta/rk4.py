from numba import jit
import numpy as np

from typing import Generator

from ODEplotter.utils.types import DerivativeFunction, Vector
from ...utils.types import *

from ..solution_method import SolutionMethod


def runge_kutta_4(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
) -> Generator[SolutionPoint]:
    derivatives = np.zeros((4, len(y)), dtype=y.dtype)
    weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6
    half_h = h * 0.5
    while True:
        yield t, y.copy()
        derivatives[0] = derivative(t, y)
        derivatives[1] = derivative(t + half_h, y + half_h * derivatives[0])
        derivatives[2] = derivative(t + half_h, y + half_h * derivatives[1])
        derivatives[3] = derivative(t + h, y + h * derivatives[2])
        t += h
        y += h * weights.dot(derivatives)


class RungeKutta4(SolutionMethod):
    method = staticmethod(runge_kutta_4)
    compiled_method = staticmethod(jit(runge_kutta_4))

    def __init__(self):
        super().__init__()
    
    def _validate(self):
        pass

    def _prepare_arguments(self, derivative: DerivativeFunction, t0: Time, y0: Vector, h: Time, use_jit: bool):
        return (derivative, t0, y0, h)
