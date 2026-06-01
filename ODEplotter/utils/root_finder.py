import numpy as np

from typing import Callable, Any
from .types import Time, Vector
from .norms import jit_two_norm

from math import ulp, sqrt


SQRT_MACHINE_EPS: float = sqrt(ulp(1.0))


"""Methods for finding roots to functions, both scalar- and vector-valued ones.

The methods herein are used in two main places, with different requirements:
1. In `Obstacles`.

    There, it is always used with scalar-to-scalar functions, and with
    brackets. The function can be fairly expensive to calculate
    (`distance_function` after `interpolate_y`), but it is also assumed
    to be close to linear, so Newton's method is used with a final bounds check.

2. In implicit `SolutionMethod`s, for example Implicit Euler.

    There, it is used to solve for `deficit(y_next) == 0`, with a single initial
    value. The vectors can be any shape, but the input and output shapes are
    always the same. The deficit function has a root close to the initial value
    if the step size is small, so if Newton's method diverges, the user should
    consider decreasing it.

In both cases, Newton's method is suitable. They are implemented here to be
simple and fast, but also uncompromizing. If it diverges, or throws an error,
the caller is deemed responsible for the failure.
"""


@np.errstate(over="raise", divide="raise", invalid="raise")
def scalar_newton(
    function: Callable[[Time], float | np.floating],
    a: Time,
    b: Time,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-12,
    eps: float = SQRT_MACHINE_EPS,
    max_iterations: int = 20,
    raise_if_exceeded: bool = True,
):
    """Find a root of `function` in the interval `[a, b]` using Newton's method."""
    x = (a + b) * 0.5
    if max_iterations <= 0:
        return x

    fx = function(x)
    residual = abs(fx)
    convergence_tolerance = rtol * residual + atol

    for i in range(max_iterations):
        # Newton step
        deriv = (function(x + eps) - fx) / eps
        try:
            x -= fx / deriv
        except (ZeroDivisionError, FloatingPointError):
            # This should be very rare
            raise RuntimeError(f"Newton's method got stuck at {x}") from None

        fx = function(x)
        residual = abs(fx)
        if residual <= convergence_tolerance:
            # Converged
            break
    else:
        if raise_if_exceeded:
            raise RuntimeError(f"Newton's method did not converge in {max_iterations} iterations")

    if not a <= x <= b:
        raise RuntimeError(f"Newton's method converged to {x} which is outside the interval [{a}, {b}]")
    return x


@np.errstate(over="raise", divide="raise", invalid="raise")
def vector_newton(
    function: Callable[[Vector], Vector],
    x0: Vector,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-12,
    max_iterations: int = 10,
    raise_if_exceeded: bool = True,
):
    """Find a root of `function` close to `x0` using Newton's method.

    `function` must return a `Vector` of the same shape as `x0`.
    """
    if max_iterations <= 0:
        return x0
    # The Jacobian can only be approximated for flat vectors
    x = x0.flatten()
    flattened_func = lambda x: function(x.reshape(x0.shape)).ravel()  # .flatten() copies, .ravel() does not

    fx = flattened_func(x)
    residual = jit_two_norm(fx)
    convergence_tolerance = rtol * residual + atol

    for i in range(max_iterations):
        # Newton step
        jac = jacobian(flattened_func, x, fx)
        try:
            x -= np.linalg.solve(jac, fx)
        except np.linalg.LinAlgError:
            # The Jacobian was singular. This should be exceedingly rare in practice.
            raise RuntimeError(f"Newton's method got stuck at {x}") from None

        fx = flattened_func(x)
        residual = jit_two_norm(fx)
        if residual <= convergence_tolerance:
            # Converged
            break
    else:
        if raise_if_exceeded:
            raise RuntimeError(f"Newton's method did not converge in {max_iterations} iterations.")

    return x.reshape(x0.shape)


def jacobian(
    function: Callable[[Vector], Vector],
    x: Vector,
    fx: Vector | None = None,
    *,
    eps: float = SQRT_MACHINE_EPS,
):
    if x.ndim != 1:
        raise ValueError("Can only calculate the Jacobian for one-dimensional Vectors")
    n, = x.shape
    if fx is None:
        fx = function(x)

    # The rows become df/dx_0, df/dx_1, ..., df/dx_{n-1}
    jac = np.empty((n, n), dtype=fx.dtype)
    perturbation = np.zeros(n, dtype=x.dtype)
    for i in range(n):
        # Perturb in one coordinate at a time
        perturbation[i] = eps
        jac[:, i] = (function(x + perturbation) - fx) / eps
        perturbation[i] = 0.0

    return jac

