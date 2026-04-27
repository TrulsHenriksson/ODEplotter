import pytest
from contextlib import contextmanager
import numpy as np
from matplotlib import pyplot as plt

from typing import Generator
from ODEplotter import DiscreteSolution
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



def example_point_gen_1D() -> Generator[tuple[Time, Vector]]:
    t = 0.0
    while True:
        yield t, np.full(3, t)
        t += 1

def example_point_gen_2D() -> Generator[tuple[Time, Vector]]:
    t = 0.0
    while True:
        yield t, np.full((3, 4), t)
        t += 1


def example_solution_1D() -> DiscreteSolution:
    sol = DiscreteSolution(example_point_gen_1D())
    sol.load_until(10.0)
    return sol

def example_solution_2D() -> DiscreteSolution:
    sol = DiscreteSolution(example_point_gen_2D())
    sol.load_until(10.0)
    return sol


def extrapolate_expected_ys(expected_ts, y_shape):
    """Convenience function that extrapolates the expected ys from the expected ts for the example solutions."""
    t_ndim = np.ndim(expected_ts)
    y_ndim = len(y_shape)
    # Add singleton dimensions to the end of expected_ts, and then tile in parallel
    expected_ys = np.tile(np.reshape(expected_ts, [-1] + [1] * (y_ndim + t_ndim - 1)), y_shape)
    return expected_ys

@pytest.mark.parametrize(
    ["ts", "y_shape", "expected_ys"],
    [
        (0.0, (2,), [0.0, 0.0]),
        (0.0, (2, 3), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ([0.0], (2, 3), [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        ([0.0, 1.0], (2, 3), [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
    ],
)
def test_reshape_ts_to_ys(ts, y_shape, expected_ys):
    assert np.array_equal(expected_ys, extrapolate_expected_ys(ts, y_shape))


@pytest.mark.parametrize("point_gen", [example_point_gen_1D, example_point_gen_2D])
def test_loading(point_gen):
    sol = DiscreteSolution(point_gen())
    sol.load(9)
    assert len(sol.ts) == 10
    assert len(sol.ys) == 10
    sol.load_until(19.0)
    assert len(sol.ts) == 20
    assert len(sol.ys) == 20


@pytest.mark.parametrize("solution", [example_solution_1D, example_solution_2D])
def test_interpolation(solution):
    sol = solution()
    # Times and expected values at those times
    times = [-5.5, 0.5, 2.7, 17.5]
    expected_values = [0.0, 0.5, 2.7, 10.0]

    # Test individual times
    for t, expected in zip(times, expected_values):
        y_interp = sol(t, load=False)
        assert y_interp.shape == sol.y0.shape
        assert np.allclose(y_interp, expected)

    # Test TimeArrays
    y_interp = sol(times, load=False)
    assert y_interp.shape == (len(expected_values), *sol.y0.shape)
    for y, expected in zip(y_interp, expected_values):
        assert np.allclose(y, expected)

    # Test zero-length TimeArrays
    y_interp = sol([], load=False)
    assert y_interp.shape == (0, *sol.y0.shape)


@pytest.mark.parametrize(
    ["start_index", "end_index", "expected_ts"],
    [
        (None, None, np.arange(11)),
        (None, 20, np.arange(11)),
        (5, None, np.arange(5, 11)),
        (5, 8, np.arange(5, 8)),
        (8, 5, np.empty(0)),
    ],
)
@pytest.mark.parametrize("solution", [example_solution_1D, example_solution_2D])
def test_get_arrays(solution, start_index, end_index, expected_ts):
    sol = solution()

    ts, ys = sol.get_arrays(start_index, end_index, load=False)
    assert ts == pytest.approx(expected_ts)
    assert ys == pytest.approx(extrapolate_expected_ys(expected_ts, sol.y0.shape))


@pytest.mark.parametrize(
    ["idx", "expected_ts"],
    [
        (slice(None, None), np.arange(11)),
        (slice(None, 20), np.arange(11)),
        (slice(5, None), np.arange(5, 11)),
        (slice(5, 8), np.arange(5, 8)),
        (slice(8, 5), np.empty(0)),
        (slice(3, 10, 3), np.arange(3, 10, 3)),
        (0, 0.0),
        (19, IndexError),
        ("0", TypeError),
    ],
)
@pytest.mark.parametrize("solution", [example_solution_1D, example_solution_1D])
def test_getitem(solution, idx, expected_ts):
    sol = solution()

    if not isinstance(idx, (int, slice)):
        with pytest.raises(TypeError):
            ts, ys = sol[idx]
    elif isinstance(idx, int) and idx >= len(sol.ts):
        with pytest.raises(IndexError):
            ts, ys = sol[idx]
    else:
        ts, ys = sol[idx]
        assert ts == pytest.approx(expected_ts)
        assert ys == pytest.approx(extrapolate_expected_ys(expected_ts, sol.y0.shape))


@pytest.mark.parametrize("t_intervals, closed_side, expected_ts_list",
    [
        ([0.5, 2.5], "left", [[1.0, 2.0]]),
        ([0, 1, 2], "left", [[0.0], [1.0]]),
        ([-10, 1, 20], "left", [[0.0], np.arange(1, 11)]),
        ([0.5, 1, 2, 4, 8], "right", [[1.0], [2.0], [3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        ([1], "left", []),
    ],
)
@pytest.mark.parametrize("solution", [example_solution_1D, example_solution_2D])
def test_get_arrays_between(solution, t_intervals, closed_side, expected_ts_list):
    sol = solution()

    ts_list, ys_list = sol.get_arrays_between(t_intervals, closed_side, load=False)
    assert len(ts_list) == len(expected_ts_list)
    for ts, ys, expected_ts in zip(ts_list, ys_list, expected_ts_list):
        assert ts == pytest.approx(expected_ts)
        assert ys == pytest.approx(extrapolate_expected_ys(expected_ts, sol.y0.shape))


@pytest.mark.parametrize("error, xcoord, ycoord", [
    (None, -1, 0),
    (None, 0, 1),
    (None, (0,), (1,)),
    (None, -1, (-2,)),
    (IndexError, -1, 3),
    (IndexError, 5, 0),
    (ValueError, -1, slice(None, None, 2)),  # ys[:, [0, 2]] is not 1-dim
])
def test_plot_solution_1D(error, xcoord, ycoord):
    sol = example_solution_1D()

    if error is None:
        with not_raises():
            sol.plot(0.0, 10.0, xcoord=xcoord, ycoord=ycoord, load=False)
    else:
        with pytest.raises(error):
            sol.plot(0.0, 10.0, xcoord=xcoord, ycoord=ycoord, load=False)

@pytest.mark.parametrize("error, xcoord, ycoord", [
    (None, -1, (0, 0)),
    (None, -1, (2, 3)),
    (None, -1, -1),
    (ValueError, 0, -1),  # ys[:, 0] is not 1-dim
    (ValueError, -1, (slice(None, None, 2), 1)),  # ys[:, ::2, 1] is not 1-dim
])
def test_plot_solution_2D(error, xcoord, ycoord):
    sol = example_solution_2D()

    if error is None:
        with not_raises():
            sol.plot(0.0, 10.0, xcoord=xcoord, ycoord=ycoord, load=False)
    else:
        with pytest.raises(error):
            sol.plot(0.0, 10.0, xcoord=xcoord, ycoord=ycoord, load=False)


@pytest.mark.parametrize("error, xcoord, ycoord, zcoord", [
    (None, -1, 0, 1),
    (None, 0, 0, (0,)),
    (None, (0,), (1,), (2,)),
    (None, -1, -1, -1),
    (IndexError, -1, 0, 3),
    (IndexError, -1, 0, (3, 3)),
    (ValueError, -1, 0, ([1, 2], )),  # ys[:, [1, 2]] is not 1-dim
])
def test_plot_3d_solution_1D(error, xcoord, ycoord, zcoord):
    sol = example_solution_1D()
    ax: Axes3D = plt.subplot(111, projection="3d")  # type: ignore (matplotlib gets it wrong)

    if error is None:
        with not_raises():
            sol.plot_3d(0.0, 10.0, xcoord=xcoord, ycoord=ycoord, zcoord=zcoord, ax=ax, load=False)
    else:
        with pytest.raises(error):
            sol.plot_3d(0.0, 10.0, xcoord=xcoord, ycoord=ycoord, zcoord=zcoord, ax=ax, load=False)

@pytest.mark.parametrize("error, xcoord, ycoord, zcoord", [
    (None, -1, (0, 0), (1, 2)),
    (None, (2, 1), (1, 0), (0, 1)),
    (None, -1, -1, -1),
    (IndexError, -1, (0, 0), (3, 4)),
    (IndexError, -1, (0, 0), (3, 3, 3)),
    (ValueError, -1, 0, 0),  # ys[:, 0] is not 1-dim
])
def test_plot_3d_solution_2D(error, xcoord, ycoord, zcoord):
    sol = example_solution_2D()
    ax: Axes3D = plt.subplot(111, projection="3d")  # type: ignore (matplotlib gets it wrong)

    if error is None:
        with not_raises():
            sol.plot_3d(0.0, 10.0, xcoord=xcoord, ycoord=ycoord, zcoord=zcoord, ax=ax, load=False)
    else:
        with pytest.raises(error):
            sol.plot_3d(0.0, 10.0, xcoord=xcoord, ycoord=ycoord, zcoord=zcoord, ax=ax, load=False)
