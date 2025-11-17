import pytest
import numpy as np
from time import time

from ODEplotter.methods import METHODS, AdaptiveRungeKutta


fixed_methods = {name: method for name, method in METHODS.items() if not isinstance(method, AdaptiveRungeKutta)}
adaptive_methods = {name: method for name, method in METHODS.items() if isinstance(method, AdaptiveRungeKutta)}

# Get a dict of only the methods that are first of their types
unique_method_types = set()
unique_methods = {}
for name, method in METHODS.items():
    if type(method) not in unique_method_types:
        unique_method_types.add(type(method))
        unique_methods.update({name: method})

EXP_DERIVATIVE = lambda t, y: -y


def test_method_consistency():
    for method in METHODS.values():
        method._validate()


@pytest.fixture
def example_problem():
    return dict(derivative=EXP_DERIVATIVE, t0=0.0, y0=np.array([1.0, 2.0]), h=0.01)

@pytest.fixture
def example_adaptive_problem():
    return dict(derivative=EXP_DERIVATIVE, t0=0.0, y0=np.array([1.0, 2.0]), h0=0.01, tol=1e-1)


@pytest.mark.parametrize(
    "method_name", [name for name, method in unique_methods.items() if method.compiled_method is not None]
)
def test_method_jitting(method_name: str, example_problem, example_adaptive_problem):
    """Test both that JITing works and that it reuses it once it is compiled."""
    method = METHODS[method_name]

    if isinstance(method, AdaptiveRungeKutta):
        t1 = time()
        method.solve(**example_adaptive_problem, use_jit=True)
        t2 = time()
        method.solve(**example_adaptive_problem, use_jit=True)
        t3 = time()
    else:
        t1 = time()
        method.solve(**example_problem, use_jit=True)
        t2 = time()
        method.solve(**example_problem, use_jit=True)
        t3 = time()
    # The second time should reuse the already compiled method and therefore be much faster
    compile_time = t2 - t1
    already_compiled_time = t3 - t2
    assert compile_time / already_compiled_time > 1000


@pytest.fixture
def simple_derivative():
    """Defines an ODE with constant derivative. Every method should work perfectly on this example."""
    derivative = lambda t, y: np.ones_like(y)
    return derivative


# @pytest.mark.parametrize("use_jit", [True, False])  # Takes much longer
@pytest.mark.parametrize("method_name", METHODS.keys())
def test_simple_ode_solution(method_name: str, simple_derivative, use_jit: bool = False):
    method = METHODS[method_name]
    if use_jit and method.compiled_method is None:
        # Can't do this, just treat it as a success
        return

    t0 = 0.0
    y0 = np.array([1.0, 2.0])
    h = 1.0

    if isinstance(method, AdaptiveRungeKutta):
        point_gen = method.solve(simple_derivative, t0, y0, h, tol=1e-2, use_jit=use_jit)
    else:
        point_gen = method.solve(simple_derivative, t0, y0, h, use_jit=use_jit)

    for i in range(10):
        t, y = next(point_gen)
        assert y == pytest.approx(y0 + t)
