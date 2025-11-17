import pytest
from contextlib import contextmanager

import numpy as np

from ODEplotter.utils.types import *


# Boilerplate from https://gist.github.com/oisinmulvihill/45c14271fad7794a4a52516ecb784e69 (modified)
@contextmanager
def not_raises(*expected_exceptions):
    try:
        yield
    except tuple(expected_exceptions) as error:
        pytest.fail(f"Raised exception {error} when it should not!")
    except Exception as error:
        pytest.fail(f"An unexpected exception {error} raised.")


@pytest.mark.parametrize(
    "error, value",
    [
        (TypeError, None),
        (TypeError, [1, 2, 3]),
        (TypeError, (1, 2, 3)),
        (ValueError, "abc"),
        (ValueError, "1e-0.5"),
        (None, 0.0),
        (None, 0),
        (None, 99999999999999999999999999999999),
        (None, "800"),
        (None, "inf"),
        (None, "nan"),
        (None, to_time(1.0)),
    ],
)
def test_to_time_exceptions(error, value):
    if error is None:
        with not_raises(TypeError, ValueError):
            to_time(value)
    else:
        with pytest.raises(error):
            to_time(value)


def test_to_time_equality():
    assert 1.0 == to_time(1.0)
    assert to_time(2314.512) == to_time(to_time(2314.512))
    assert to_time(np.float64("1.16")) == 1.16
    assert isinstance(to_time(np.float64("1.16")), float)


def test_is_time():
    assert is_time(0.0)
    assert is_time(-0.0)
    assert is_time(float("nan"))  # Maybe disallow nan?
    assert is_time(float("inf"))
    assert is_time(np.float16("1"))
    assert is_time(np.float32("1"))
    assert is_time(np.float64("1"))
    assert not is_time(0)
    assert not is_time("1.0")
    assert not is_time([1, 2, 3])
    assert not is_time(np.zeros(1))
    assert not is_time(np.array(1.0))


@pytest.mark.parametrize(
    "error, value",
    [
        (TypeError, dict(abc=5)),
        (TypeError, "0.5"),
        (TypeError, "abcdef"),
        (TypeError, None),
        (TypeError, lambda x: x + 1),
        (ValueError, [[[]], []]),
        (ValueError, [-3.0, "inf"]),
        (ValueError, [[5.0]]),
        (ValueError, np.array(["abd", "egf"])),
        (None, 5.0),
        (None, np.array(5.0)),
        (None, (1, 2, 3)),
        (None, [-3.0, float("inf")]),
        (None, range(15)),
        (None, []),
        (None, np.array([2.0, 3.0, 5.0])),
    ],
)
def test_to_time_array_exceptions(value, error):
    if error is None:
        with not_raises(TypeError, ValueError):
            to_time_array(value)
    else:
        with pytest.raises(error):
            to_time_array(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (a := np.array([1, 2, 3], dtype=np.float64), a),
        ([1, 2, 3], np.array([1.0, 2.0, 3.0])),
        ((1, 2, 3), np.array([1.0, 2.0, 3.0])),
        (range(1, 4), np.array([1.0, 2.0, 3.0])),
        (np.arange(3, dtype=np.float16), np.arange(3, dtype=np.float16)),
        (np.arange(3, dtype=np.int8), np.arange(3, dtype=np.float64)),
        (np.arange(3, dtype=np.int16), np.arange(3, dtype=np.float64)),
        (np.arange(3, dtype=np.int32), np.arange(3, dtype=np.float64)),
        (np.arange(3, dtype=np.int64), np.arange(3, dtype=np.float64)),
    ],
)
def test_to_time_array_equality(value, expected):
    time_array = to_time_array(value)
    assert time_array.dtype == expected.dtype
    assert np.array_equal(time_array, expected)


@pytest.mark.parametrize(
    "value",
    [
        to_time_array([1, 2, 3]),
        np.arange(5, dtype=np.float16),
        np.arange(5, dtype=np.float32),
        np.arange(5, dtype=np.float64),
    ],
)
def test_to_time_array_identity(value):
    assert to_time_array(value, copy=False) is value
    assert to_time_array(value, copy=True) is not value
    assert to_time_array(value) is value  # Doesn't copy by default


def test_is_time_array():
    assert is_time_array(np.arange(3, dtype=np.float16))
    assert is_time_array(np.arange(3, dtype=np.float32))
    assert is_time_array(np.arange(3, dtype=np.float64))
    assert is_time_array(np.repeat(np.inf, 5))
    assert not is_time_array([1.0, 2.0, 3.0])
    assert not is_time_array(range(3))
    assert not is_time_array(np.array(1.0))
    assert not is_time_array(np.zeros((2, 2)))
    assert not is_time_array(np.zeros(3, dtype=np.complex128))
    assert not is_time_array(np.zeros(3, dtype=np.int64))


@pytest.mark.parametrize(
    "error, value",
    [
        (TypeError, ...),
        (TypeError, None),
        (TypeError, "abcdef"),
        (TypeError, "1.0"),
        (TypeError, lambda x: x + 1),
        (ValueError, [[[]], []]),
        (ValueError, []),
        (ValueError, [[9.0]]),
        (ValueError, ["9.0"]),
        (ValueError, ("9.0",)),
        (None, np.array([4.0, 1.0, -999.0])),
        (None, range(8)),
        (None, [8, 9]),
        (None, (8, 9)),
        (None, 1.0),
        (None, 1 + 2j),
        (None, np.arange(3) + np.ones(3) * 1j),
        (None, np.arange(3, dtype=np.int64)),
        (None, np.arange(3, dtype=np.complex64)),
    ],
)
def test_to_vector_exceptions(value, error):
    if error is None:
        with not_raises(TypeError, ValueError):
            to_vector(value)
    else:
        with pytest.raises(error):
            to_vector(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (v := to_vector([0, 1, 2]), v),
        (range(3), v),
        ([0, 1, 2], v),
        ((0, 1, 2), v),
        (np.arange(3), v),
        (np.arange(3, dtype=np.int16), v),
        (np.arange(3, dtype=np.int64), v),
    ],
)
def test_to_vector_equality(value, expected):
    vector = to_vector(value)
    assert vector.dtype == expected.dtype
    assert np.array_equal(vector, expected)


@pytest.mark.parametrize(
    "value",
    [
        np.arange(3, dtype=np.float16),
        np.arange(3, dtype=np.float32),
        np.arange(3, dtype=np.float64),
        np.arange(3, dtype=np.complex64),
        np.arange(3, dtype=np.complex128),
    ],
)
def test_to_vector_identity(value):
    assert to_vector(value, copy=False) is value
    assert to_vector(value, copy=True) is not value
    assert to_vector(value) is value  # Doesn't copy by default


def test_is_vector():
    assert is_vector(to_vector(0.0))
    assert is_vector(to_vector([0.0]))
    assert is_vector(np.array([0.0]))
    assert is_vector(np.array([0.0], dtype=np.float16))
    assert is_vector(np.array([0.0], dtype=np.float32))
    assert is_vector(np.array([0.0 + 0.0j]))
    assert is_vector(np.array([0.0 + 0.0j], dtype=np.complex64))
    assert is_vector(np.array([0.0 + 0.0j], dtype=np.complex128))
    assert not is_vector(0.0)
    assert not is_vector((0.0,))
    assert not is_vector([0.0])
    assert not is_vector(range(1))
    assert not is_vector(np.arange(3, dtype=np.int8))
    assert not is_vector(np.ones((2, 2)))


@pytest.mark.parametrize(
    "error, dimension, value",
    [
        (TypeError, None, None),
        (TypeError, None, ...),
        (TypeError, None, lambda x: x + 1),
        (TypeError, None, "abc"),
        (TypeError, None, "[[1.0]]"),
        (ValueError, None, [[], 0.0]),
        (ValueError, None, [["0.0", "0.1"]]),
        (ValueError, None, [[[]]]),
        (ValueError, None, 0.0),
        (ValueError, None, []),
        (ValueError, 3, [0.0, 0.1]),
        (ValueError, 3, [[0.0, 0.1]]),
        (ValueError, None, [[]]),
        (ValueError, None, [[], []]),
        (ValueError, 2, [[], []]),
        (None, None, [[0.1]]),
        (None, 1, [[0.1]]),
        (None, None, [0]),
        (None, None, np.array([[1]], dtype=np.int8)),
        (None, None, np.array([[1]], dtype=np.complex64)),
        (None, None, [[True, False]]),
        (None, 2, []),
        (None, 2, [[]]),
        (None, 2, np.empty((0,))),
        (None, 2, np.empty((0, 0))),
        (None, 2, np.empty((0, 2))),
        (None, 2, np.empty((1, 0))),
    ],
)
def test_to_vector_array_exceptions(error, dimension, value):
    if error is None:
        with not_raises(TypeError, ValueError):
            to_vector_array(value, dimension)
    else:
        with pytest.raises(error):
            to_vector_array(value, dimension)


@pytest.mark.parametrize(
    "dimension, value, expected",
    [
        (None, v := to_vector_array([[0.0, 1.0, 2.0]]), v),
        (None, range(3), v),
        (None, [0, 1, 2], v),
        (None, [[0, 1, 2]], v),
        (None, (0, 1, 2), v),
        (None, np.arange(3), v),
        (None, v := to_vector_array([[0 + 1j, 1 + 1j, 2 + 1j]]), v),
        (None, np.arange(3) + np.ones(3) * 1j, v),
        (2, [], np.empty((0, 2), dtype=np.float64)),
        (2, [[]], np.empty((0, 2), dtype=np.float64)),
        (2, np.array([]), np.empty((0, 2), dtype=np.float64)),
        (2, np.empty((0,)), np.empty((0, 2), dtype=np.float64)),
        (2, np.empty((0, 0)), np.empty((0, 2), dtype=np.float64)),
        (2, np.empty((1, 0)), np.empty((0, 2), dtype=np.float64)),
        (2, np.empty((0, 2)), np.empty((0, 2), dtype=np.float64)),
    ],
)
def test_to_vector_array_equality(value, dimension, expected):
    array = to_vector_array(value, dimension)
    assert array.dtype == expected.dtype
    assert np.array_equal(array, expected)


@pytest.mark.parametrize("dimension", [None, 2])
@pytest.mark.parametrize(
    "value",
    [
        np.array([[1.0, 2.0]]),
        np.empty((0, 2), dtype=np.float16),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float64),
        np.empty((0, 2), dtype=np.complex64),
        np.empty((0, 2), dtype=np.complex128),
        np.ones((3, 2), dtype=np.float64),
        np.ones((3, 2), dtype=np.complex128),
    ],
)
def test_to_vector_array_identity(value, dimension):
    assert to_vector_array(value, dimension, copy=False) is value
    assert to_vector_array(value, dimension, copy=True) is not value
    assert to_vector_array(value, dimension) is value  # Doesn't copy by default


def test_is_vector_array():
    assert not is_vector_array(None)
    assert not is_vector_array([])
    assert not is_vector_array(np.array(0.0))
    assert not is_vector_array(np.array([0.0]))
    assert not is_vector_array(np.array([[]]), None)
    assert not is_vector_array(np.empty((0,)))
    assert not is_vector_array(np.empty((0, 0)))
    assert not is_vector_array(np.empty((1, 0)))
    assert not is_vector_array(np.empty((2, 0)))
    assert not is_vector_array(np.ones((2, 3)), 2)
    assert not is_vector_array(np.array([[True, False]]))
    assert not is_vector_array(np.array([["abc", "def"]]))
    assert not is_vector_array(np.array([[1, 0]]))
    assert is_vector_array(np.ones((3, 2)), 2)
    assert is_vector_array(np.ones((3, 2)), None)
    assert is_vector_array(np.empty((0, 2)), 2)
    assert is_vector_array(np.empty((0, 2)), None)
