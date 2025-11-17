from numba import jit
from abc import abstractmethod, ABCMeta

from typing import Callable, Generator, Any
from ..utils.types import *


class SolutionMethod(metaclass=ABCMeta):
    method: Callable[..., Generator[SolutionPoint]]
    compiled_method: Callable[..., Generator[SolutionPoint]] | None

    compiled_derivatives: dict[DerivativeFunction, DerivativeFunction] = {}  # Shared between all SolutionMethods

    def __init__(self):
        """Give method data to define a specific method."""
        self.validated = False
    
    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    @abstractmethod
    def _validate(self):
        """Make sure the given method data defines a consistent method.

        This is not done in `__init__` because a lot of methods are defined,
        and it's unnecessary to validate all of them before they're used.
        """
        ...

    @abstractmethod
    def _prepare_arguments(
        self, derivative: DerivativeFunction, t: Time, y: Vector, *args, use_jit: bool, **kwargs
    ) -> tuple[Any, ...]:
        """Prepare user-given arguments to pass to `self.method`, whose arguments cannot include `self`.

        In order for `self.method` to be JIT-compilable, the arguments can't
        include `self`. Therefore, a `SolutionMethod` that has a `weights` variable,
        for example, must pass this as an argument to `self.method` instead of
        the method just accessing `self.weights` directly.
        """
        ...

    def _compile_derivative(self, derivative: DerivativeFunction) -> DerivativeFunction:
        """Compile the derivative function, or reuse an already compiled one.

        Because `self.compiled_derivatives` is a mutable class variable of the
        top-level class `SolutionMethod`, all subclasses and instances receive
        a reference to the same dictionary. In this case, this is really useful,
        since it means that two different methods can be called with `use_jit=True`
        and use the same compiled derivative, saving time.
        """
        if derivative in self.compiled_derivatives:
            return self.compiled_derivatives[derivative]

        try:
            compiled_derivative = jit(derivative)
        except Exception as e:
            raise ValueError("Could not JIT compile the derivative function") from e
        self.compiled_derivatives.update({derivative: compiled_derivative})
        return compiled_derivative

    def solve(
        self, derivative: DerivativeFunction, t0, y0, *args, use_jit: bool = False, **kwargs
    ) -> Generator[SolutionPoint]:
        """Approximate the solution to `y'(t) = derivative(t, y(t))` where `y(t0) = y0`."""
        # Short-circuits if already validated
        self._validate()

        t0 = to_time(t0)
        y0 = to_vector(y0, copy=True)

        if use_jit:
            if self.compiled_method is None:
                raise NotImplementedError(f"{type(self).__name__} cannot be solved with use_jit=True")

            compiled_derivative = self._compile_derivative(derivative)
            arguments = self._prepare_arguments(compiled_derivative, t0, y0, *args, use_jit=use_jit, **kwargs)
            return self.compiled_method(*arguments)
        else:
            arguments = self._prepare_arguments(derivative, t0, y0, *args, use_jit=use_jit, **kwargs)
            return self.method(*arguments)
