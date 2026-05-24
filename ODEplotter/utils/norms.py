import numpy as np
from math import sqrt
from numba import jit

from typing import Callable
from .types import Vector


"""
Performance-optimized vector norms.
"""


def one_norm(vec: Vector) -> float:
    """Compute the 1-norm of `vec`."""
    return float(np.abs(vec).sum())

def two_norm(vec: Vector) -> float:
    """Compute the 2-norm of `vec`, formally equivalent to the Frobenius norm."""
    vec = vec.ravel()
    return sqrt(vec.dot(vec.conj()).real)

def max_norm(vec: Vector) -> float:
    """Compute the max-norm of `vec`, aka the infinity norm."""
    return float(max(np.abs(vec.real).max(), np.abs(vec.imag).max()))


jit_one_norm: Callable[[Vector], float] = jit(one_norm)
jit_two_norm: Callable[[Vector], float] = jit(two_norm)
jit_max_norm: Callable[[Vector], float] = jit(max_norm)
