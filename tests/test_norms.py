import pytest
import numpy as np

from ODEplotter.utils.norms import one_norm, two_norm, max_norm, jit_one_norm, jit_two_norm, jit_max_norm



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
