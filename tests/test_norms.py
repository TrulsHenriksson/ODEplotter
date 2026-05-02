import pytest
from contextlib import contextmanager

import numpy as np
from numba import jit

from ODEplotter.methods.runge_kutta.adaptive_runge_kutta import one_norm, two_norm, max_norm


# Boilerplate from https://gist.github.com/oisinmulvihill/45c14271fad7794a4a52516ecb784e69 (modified)
@contextmanager
def not_raises(*expected_exceptions):
    try:
        yield
    except expected_exceptions as error:
        pytest.fail(f"Raised exception {error} when it should not!")
    except Exception as error:
        pytest.fail(f"An unexpected exception {error} raised.")


jit_one_norm = jit(one_norm)
jit_two_norm = jit(two_norm)
jit_max_norm = jit(max_norm)


@pytest.mark.parametrize("norm", [one_norm, jit_one_norm, two_norm, jit_two_norm, max_norm, jit_max_norm])
def test_zeros(norm):
    # **Exactly** zero
    assert norm(np.zeros(3, dtype=np.float32)) == 0.0
    assert norm(np.zeros(3, dtype=np.float64)) == 0.0
    assert norm(np.zeros(3, dtype=np.complex64)) == 0.0
    assert norm(np.zeros(3, dtype=np.complex128)) == 0.0

@pytest.mark.parametrize("norm", [one_norm, jit_one_norm])
def test_one_norm(norm):
    assert norm(-np.ones(4, dtype=np.float32)) == pytest.approx(4.0)
    assert norm(-np.ones(4, dtype=np.float64)) == pytest.approx(4.0)
    assert norm(-np.ones(4, dtype=np.complex64)) == pytest.approx(4.0)
    assert norm(-np.ones(4, dtype=np.complex128)) == pytest.approx(4.0)
    assert norm(-np.ones((2, 2))) == pytest.approx(4.0)
    assert norm(-np.ones((2, 3, 4))) == pytest.approx(24.0)

@pytest.mark.parametrize("norm", [two_norm, jit_two_norm])
def test_two_norm(norm):
    assert norm(-np.ones(4, dtype=np.float32)) == pytest.approx(2.0)
    assert norm(-np.ones(4, dtype=np.float64)) == pytest.approx(2.0)
    assert norm(-np.ones(4, dtype=np.complex64)) == pytest.approx(2.0)
    assert norm(-np.ones(4, dtype=np.complex128)) == pytest.approx(2.0)
    assert norm(-np.ones((2, 2))) == pytest.approx(2.0)
    assert norm(-np.ones((2, 3, 4))) == pytest.approx(24**0.5)

@pytest.mark.parametrize("norm", [max_norm, jit_max_norm])
def test_max_norm(norm):
    assert norm(-np.ones(4, dtype=np.float32)) == pytest.approx(1.0)
    assert norm(-np.ones(4, dtype=np.float64)) == pytest.approx(1.0)
    assert norm(-np.ones(4, dtype=np.complex64)) == pytest.approx(1.0)
    assert norm(-np.ones(4, dtype=np.complex128)) == pytest.approx(1.0)
    assert norm(-np.ones((2, 2))) == pytest.approx(1.0)
    assert norm(-np.ones((2, 3, 4))) == pytest.approx(1.0)
