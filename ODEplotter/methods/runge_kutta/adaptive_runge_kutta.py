from numba import jit
import numpy as np
from math import sqrt, ulp

from typing import Generator, Callable, Literal
from ...utils.types import *
from ...utils.exceptions import StepSizeTooSmallError

from ..solution_method import weighted_sum
from .runge_kutta import RungeKutta


MACHINE_EPS = ulp(1.0)


def one_norm(vec: Vector) -> float:
    return float(np.abs(vec).sum())

def two_norm(vec: Vector) -> float:
    vec = vec.ravel()
    return sqrt(vec.dot(vec.conj()).real)

def max_norm(vec: Vector) -> float:
    return float(max(np.abs(vec.real).max(), np.abs(vec.imag).max()))


def adaptive_runge_kutta(
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
    # step_size_factor: Callable[..., float],
) -> Generator[SolutionPoint]:
    # Array of past derivatives
    derivatives = np.zeros((stages, *y.shape), dtype=y.dtype)

    error_exponent = 1 / (order + 1)

    # Set up the last derivative for FSAL
    if first_same_as_last:
        derivatives[-1] = derivative(t, y)

    while True:
        yield t, y.copy()
        derivatives[0] = derivatives[-1] if first_same_as_last else derivative(t, y)

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
            error = max(h * norm(error_vector), MACHINE_EPS)
            # Update the step size (0.9 as a safety factor)
            h *= min(max(0.9 * (tol / error) ** error_exponent, 0.3), 2)  # max() to not divide by zero

            if h < min_h:
                # Cannot use string formatting like {min_h:.1e} or {t = } inside JITed functions
                raise StepSizeTooSmallError(f"Adaptive time step got too small (<{min_h}) at t = {t}")

        t += h
        y += h * weighted_sum(derivatives, weights)


class AdaptiveRungeKutta(RungeKutta):
    method = staticmethod(adaptive_runge_kutta)
    compiled_method = staticmethod(jit(adaptive_runge_kutta))
    norms = {"one": one_norm, "two": two_norm, "max": max_norm}
    jit_norms = {"one": jit(one_norm), "two": jit(two_norm), "max": jit(max_norm)}

    stages: int
    nodes: TimeArray
    weights: WeightArray
    matrix: WeightMatrix
    error_weights: WeightArray
    order: int

    def __init__(self, nodes, weights, matrix, error_weights, order: int):
        """Runge-Kutta method that chooses step size automatically to keep error below a given tolerance."""
        super().__init__(nodes, weights, matrix)
        self._error_weights = error_weights
        self.order = order

    def _validate(self):
        if self.validated:
            return

        super()._validate(complete_validation=False)
        self.error_weights = self._validate_error_weights()
        self.first_same_as_last = self.nodes[-1] == 1.0 and np.allclose(self.matrix[-1], self.weights[0])

        del self._error_weights
        self.validated = True

    def _validate_error_weights(self) -> WeightArray:
        error_weights = to_weight_array(self._error_weights)
        assert error_weights.shape == (self.stages,), f"error_weights must have shape ({self.stages},)"
        assert abs(error_weights.sum()) < 1e-14, f"error_weights must sum to 0, not {error_weights.sum()}"
        return error_weights

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
            h0,
            tol,
            min_h,
            self.stages,
            self.nodes,
            self.weights,
            self.matrix,
            self.error_weights,
            self.order,
            self.first_same_as_last,
            self.jit_norms[norm] if use_jit else self.norms[norm],
        )
