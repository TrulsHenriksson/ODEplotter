from numba import jit
import numpy as np
from math import sqrt, ulp

from typing import Generator, Callable, Literal
from ...utils.types import *
from ...utils.exceptions import StepSizeTooSmallError

from .adaptive_runge_kutta_PI import AdaptiveRungeKuttaPI
from ..method_data import RK43_NODES, RK43_MATRIX, RK43_WEIGHTS, RK43_ERROR


MACHINE_EPS = ulp(1.0)


def runge_kutta_43(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
    tol: float,
    min_h: float,
    norm: Callable[[Vector], float],
) -> Generator[SolutionPoint]:
    # Array of past derivatives
    derivatives = np.zeros((5, len(y)), dtype=y.dtype)

    error = tol
    error_exponent = 2 / 12
    last_error_exponent = -1 / 12

    while True:
        yield t, y.copy()
        derivatives[0] = derivative(t, y)

        last_error = error
        error = tol
        # Adjust h until the error estimate is less than tol
        while error >= tol:
            half_h = 0.5 * h
            derivatives[1] = derivative(t + half_h, y + half_h * derivatives[0])
            derivatives[2] = derivative(t + half_h, y + half_h * derivatives[1])
            derivatives[3] = derivative(t + h, y + h * (-derivatives[0] + 2 * derivatives[1]))
            derivatives[4] = derivative(t + h, y + h * derivatives[2])

            error_vector = (derivatives[1] - derivatives[2]) / 3 + (derivatives[3] - derivatives[4]) / 6
            error = max(h * norm(error_vector), MACHINE_EPS)
            # Update the step size (0.9 as a safety factor)
            h *= min(max(0.9 * (tol / error)**error_exponent * (tol / last_error)**last_error_exponent, 0.3), 2)

            if h < min_h:
                # Cannot use string formatting like {min_h:.1e} or {t = } inside JITed functions
                raise StepSizeTooSmallError(f"Adaptive time step got too small (<{min_h}) at t = {t}")

        t += h
        y += h * ((derivatives[0] + derivatives[4]) / 6 + (derivatives[1] + derivatives[2]) / 3)


class RungeKutta43(AdaptiveRungeKuttaPI):
    method = staticmethod(runge_kutta_43)
    compiled_method = staticmethod(jit(runge_kutta_43))

    def __init__(self):
        super().__init__(RK43_NODES, RK43_WEIGHTS, RK43_MATRIX, RK43_ERROR, order=4)

    def _prepare_arguments(
        self,
        derivative: DerivativeFunction,
        t0: Time,
        y0: Vector,
        h0: Time,
        tol: float,
        use_jit: bool,
        min_h: float = 1e-15,
        norm: Literal["one", "two", "max"] = "max",
    ):
        return (
            derivative,
            t0,
            y0,
            to_time(h0),
            tol,
            min_h,
            self.jit_norms[norm] if use_jit else self.norms[norm],
        )
