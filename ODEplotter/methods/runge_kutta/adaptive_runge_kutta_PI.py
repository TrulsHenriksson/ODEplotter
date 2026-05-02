from numba import jit
import numpy as np
from math import ulp

from typing import Generator, Callable
from ...utils.types import *
from ...utils.exceptions import StepSizeTooSmallError

from ..solution_method import weighted_sum
from .adaptive_runge_kutta import AdaptiveRungeKutta


MACHINE_EPS = ulp(1.0)


def adaptive_runge_kutta_PI(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
    tol: float,
    min_h: float,
    stages: int,
    nodes: TimeArray,
    weights: WeightArray,
    matrix: WeightMatrix,
    error_weights: WeightArray,
    order: int,
    first_same_as_last: bool,
    norm: Callable[[Vector], float],
) -> Generator[SolutionPoint]:
    # Array of past derivatives
    derivatives = np.zeros((stages, *y.shape), dtype=y.dtype)

    error = tol
    error_exponent = 2 / (3 * order)
    last_error_exponent = -1 / (3 * order)

    # Set up the last derivative for FSAL
    if first_same_as_last:
        derivatives[-1] = derivative(t, y)

    while True:
        yield t, y.copy()
        derivatives[0] = derivatives[-1] if first_same_as_last else derivative(t, y)

        last_error = error
        error = tol
        # Adjust h until the error estimate is less than tol
        while error >= tol:
            # Take successive steps forward, just like regular RK methods
            for i in range(1, stages):
                derivatives[i] = derivative(
                    t + h * nodes[i],
                    y + h * weighted_sum(derivatives[:i], matrix[i, :i]),
                )

            error_vector = weighted_sum(derivatives, error_weights)
            error = max(h * norm(error_vector), MACHINE_EPS)  # max() to not divide by zero
            # Update the step size using a PI controller (0.9 as a safety factor)
            h *= min(max(0.9 * (tol / error)**error_exponent * (tol / last_error)**last_error_exponent, 0.3), 2)

            if h < min_h:
                # Cannot use string formatting like {min_h:.1e} or {t = } inside JITed functions
                raise StepSizeTooSmallError(f"Adaptive time step got too small (<{min_h}) at t = {t}")

        t += h
        y += h * weighted_sum(derivatives, weights)


class AdaptiveRungeKuttaPI(AdaptiveRungeKutta):
    method = staticmethod(adaptive_runge_kutta_PI)
    compiled_method = staticmethod(jit(adaptive_runge_kutta_PI))
