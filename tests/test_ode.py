import pytest
import numpy as np
from matplotlib import pyplot as plt

from ODEplotter import ODE, Obstacle


@pytest.fixture
def example_ode():
    derivative = lambda t, y: -y
    return ODE(derivative)

@pytest.fixture
def example_ode_with_obstacle():
    derivative = lambda t, y: -y
    # "Bounces" any y-value that is below 0.5, higher and higher as time goes on
    obstacle = Obstacle(
        lambda ts, ys: np.abs(ys).min(axis=1) - 0.5,
        lambda t, y: np.where(np.abs(y) < 0.5, y * (1 + t**0.5), y),
    )
    return ODE(derivative, [obstacle])


def test_ode(example_ode):
    sol = example_ode.solve(0, [1, 2, 3, 4], "euler", 0.05)
    sol.load(100)

def test_ode_with_obstacle(example_ode_with_obstacle):
    sol = example_ode_with_obstacle.solve(0, [1, 2, 3, 4], "euler", h=0.05)
    sol.load(100)
