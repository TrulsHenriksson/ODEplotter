import pytest
import numpy as np

from ODEplotter import Time, Vector
from ODEplotter.utils.root_finder import RootFinder


@pytest.fixture
def scalar_problem():
    def func(x: Time) -> Time:
        return x**2 - x - 1
    a, b = 1.0, 2.8
    root = (1 + 5**0.5) / 2
    return func, a, b, root

@pytest.fixture
def scalar_linear_problem():
    def func(x: Time) -> Time:
        return 15 * x - 23
    a, b = 1.0, 10.0
    root = 23 / 15
    return func, a, b, root

@pytest.fixture
def vector_problem():
    def derivative(x: Vector) -> Vector:
        return np.rot90(x, 2) ** 2
    x0 = np.arange(4, dtype=np.float64).reshape((2, 2))
    h = 1 / 16
    # Emulates the implicit Euler method
    def func(x: Vector) -> Vector:
        return x - x0 - h * derivative(x)
    # Solved for by WolframAlpha
    root = np.array([
        [0.5701445028041533193155493, 1.276091281518077028148926],
        [2.101775559922903007392680, 3.020316547129862199512018],
    ])
    return func, x0, root

@pytest.fixture
def vector_linear_problem():
    A = np.array([[2.0, 3.0], [-6.0, -3.0]])
    b = np.array([-1.0, -15.0])
    def func(x: Vector) -> Vector:
        return A @ x - b
    x0 = np.array([7.4, 2.3])
    root = np.array([4.0, -3.0])
    return func, x0, root


def test_scalar(scalar_problem):
    func, a, b, root = scalar_problem
    approx_root = RootFinder.scalar(func, a, b)
    assert approx_root == pytest.approx(root)

def test_vector(vector_problem):
    func, x0, root = vector_problem
    approx_root = RootFinder.vector(func, x0)
    assert approx_root == pytest.approx(root)

def test_scalar_linear(scalar_linear_problem):
    func, a, b, root = scalar_linear_problem
    approx_root = RootFinder.scalar(func, a, b, max_iterations=1)
    assert approx_root == pytest.approx(root)

def test_vector_linear(vector_linear_problem):
    func, x0, root = vector_linear_problem
    approx_root = RootFinder.vector(func, x0, max_iterations=1)
    assert approx_root == pytest.approx(root)
