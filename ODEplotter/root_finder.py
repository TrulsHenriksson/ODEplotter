import numpy as np

from typing import overload
from .utils.types import Vector, Time, is_time, is_vector


MACHINE_EPS = 7/3 - 4/3 - 1 # Machine epsilon (float hack) 


# Class for finding roots of functions. Works for both one-dimensional and
# multidimensional functions.
class RootFinder:
    newton_eps = float(np.sqrt(MACHINE_EPS)) # Determined to be at least close to optimal
    method_iterations = {'newton': 5, 'fixedpoint': 10}

    # Fixed-point method of finding roots. Works for both 1-dim and multidim functions.
    @staticmethod
    def fixedpoint(function, x0: Vector, *, iterations=method_iterations['fixedpoint']) -> Vector:
        x = x0.copy()
        for i in range(iterations):
            x -= function(x)
        return x

    @staticmethod
    @overload
    def newton(function, x0: Time, *, eps: float, iterations: int) -> Time:
        ...
    @staticmethod
    @overload
    def newton(function, x0: Vector, *, eps: float, iterations: int) -> Vector:
        ...
    @staticmethod
    def newton(function, x0, *, eps=newton_eps, iterations=method_iterations['newton']):
        if is_time(x0):
            return RootFinder._newton1D(function, float(x0), eps=eps, iterations=iterations)
        elif is_vector(x0):
            return RootFinder._newtonMD(function, x0, eps=eps, iterations=iterations)
        raise TypeError('x0 must be a float, Time, or Vector.')

    # One-dimensional Newton's method. Not meant to be called outside of the class.
    @staticmethod
    def _newton1D(function, x0: float, *, eps: float, iterations: int):
        x = x0
        for i in range(iterations):
            fxk = function(x)
            deriv = (function(x + eps) - fxk) / eps
            x = (x - fxk/deriv) if deriv != 0 else x
        return x

    # Multidimensional Newton's method. Not meant to be called outside of the class.
    @staticmethod
    def _newtonMD(function, x0: Vector, *, eps: float, iterations: int):
        x = x0.copy()
        for i in range(iterations):
            fxk = function(x)
            jac = RootFinder.jacobian(function, x, fxk, eps=eps)
            try:
                x -= np.linalg.solve(jac, fxk)
            except np.linalg.LinAlgError:
                break # It might be close to a solution, stop and return
        return x

    @staticmethod
    def jacobian(function, x: Vector, fx=None, *, eps=newton_eps):
        dim, = x.shape
        if fx is None: fx = function(x)
        # The elements become df/dx_0, df/dx_1, ..., df/dx_n
        derivs = [(function(x + eps*e_i) - fx)/eps for e_i in np.identity(dim)]
        # Returns each element (array) in deriv as a column in the Jacobian
        return np.column_stack(derivs)

    methods = {
        'newton': newton,
        'fixedpoint': fixedpoint,
    }


"""
TODO:
- Refactor to use subclasses, like methods/solution_methods.py
- Optimize for speed, or even use scipy.optimize
"""