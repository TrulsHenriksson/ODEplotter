import numpy as np
import matplotlib.pyplot as plt
from itertools import pairwise
from numba import jit

from typing import overload, Literal, Generator, Sequence, Callable
from .utils.types import *


@jit
def interp_nd(x: TimeArray, xp: TimeArray, fp: VectorArray) -> VectorArray:
    """Interpolate like np.interp, but with n-dimensional fp (along axis 0)."""
    # Using x, xp, fp, just like in np.interp
    if x.size == 0:
        return np.empty((0, *fp.shape[1:]), dtype=fp.dtype)
    x_indices = np.searchsorted(xp, x)
    after = np.minimum(x_indices, len(xp) - 1)
    before = np.maximum(x_indices - 1, 0)
    numerators = x - xp[before]
    denominators = xp[after] - xp[before]
    denominators[denominators == 0.0] = 1.0  # Set to 1 to not divide by zero
    fractions = numerators / denominators
    # Transpose to broadcast against the first axis
    # TODO: Make more efficient
    interpolated_fp = (1 - fractions) * np.transpose(fp[before]) + fractions * np.transpose(fp[after])
    return np.transpose(interpolated_fp)  # type: ignore


def pack(index: int | tuple[int, ...]) -> tuple[int, ...]:
    """Pack an array index to be able to unpack it."""
    return index if isinstance(index, tuple) else (index,)


class DiscreteSolution:
    point_gen: Generator[SolutionPoint]
    ts: list[Time]
    ys: list[Vector]
    t0: Time
    y0: Vector
    y_shape: tuple[int, ...]

    def __init__(self, point_gen: Generator[SolutionPoint]):
        """Create a lazy discrete solution from a `SolutionPoint` generator.

        A true solution to an initial value problem is a vector function `y(t)` where
        `y(t0) == y0`. This is approximated here by a linear interpolation between
        discrete points `(t, y)` given by `point_gen`.

        A DiscreteSolution stores solved values, but only solves for them when they are needed
        (e.g. in `.plot()`) or it is explicitly asked to (using `.load()`, `.load_until()`).

        Arguments
        ---------
        point_gen : generator -> (Time, Vector)
            Infinite iterator that returns (Time, Vector) pairs.
        """
        self.point_gen = point_gen
        # Lists! Only converted to numpy arrays when needed since those are immutable.
        self.ts: list[Time] = []
        self.ys: list[Vector] = []
        # Load the first point and save it
        self.load()
        self.t0 = self.ts[0]
        self.y0 = self.ys[0]
        self.y_shape = self.y0.shape

    def __repr__(self) -> str:
        return f'<DiscreteSolution with t0={self.t0}, y0={self.y0}>'

    def load(self, num: int = 1):
        """Load (numerically solve for) `num` more points."""
        if num <= 0:
            return
        # Load num t, y values from point_gen and return True if it succeeded
        new_points = [next(self.point_gen) for i in range(num)]
        new_ts, new_ys = zip(*new_points)
        self.ts.extend(new_ts)
        self.ys.extend(new_ys)

    def load_until(self, t: Time):
        """Load (numerically solve for) points until the time `t` is reached."""
        if t == np.inf:
            raise ValueError('Cannot load until infinity')
        new_t = self.ts[-1]
        # TODO: Figure out how to make this more efficient. (It feels like it should be possible)
        while new_t < t:
            new_t, new_y = next(self.point_gen)
            self.ts.append(new_t)
            self.ys.append(new_y)

    @overload
    def __call__(self, t: Time, *, load: bool = True) -> Vector:
        ...
    @overload
    def __call__(self, t: Sequence[Time] | TimeArray, *, load: bool = True) -> VectorArray:
        ...
    def __call__(self, t: Time | Sequence[Time] | TimeArray, *, load: bool = True) -> Vector | VectorArray:
        """Return the `y` value(s) from interpolating between the solution points.

        Arguments
        ---------
        t : Time or Sequence[Time] or TimeArray
            The time(s) to evaluate the linear interpolation at.

        Returns
        -------
        y(s) : Vector or VectorArray
            The interpolated `y` value(s). Returns a single `Vector` for single `Time` inputs, and
            a `VectorArray` for a sequence of `Time` inputs.
        """
        x = to_time_array(t)
        if load:
            self.load_until(x[-1])
        xp = np.array(self.ts)
        fp = np.array(self.ys)
        result = interp_nd(x, xp, fp)
        if is_time(t):
            return result[0]
        return result

    def get_arrays(self,
        start_index: int | None = None,
        end_index: int | None = None,
        *,
        load: bool = True
    ) -> tuple[TimeArray, VectorArray]:
        """Convenience function to get the `(t, y)` points between given indices.

        Similar to `DiscreteSolution.__getitem__`, but loads new points as needed.
        """
        if start_index is None and end_index is None:
            return to_time_array(self.ts), to_vector_array(self.ys, self.y_shape)
        if load and end_index is not None:
            self.load(end_index - len(self.ts))
        return to_time_array(self.ts[start_index:end_index]), to_vector_array(self.ys[start_index:end_index], self.y_shape)

    @overload
    def __getitem__(self, idx: int) -> tuple[Time, Vector]:
        ...
    @overload
    def __getitem__(self, idx: slice) -> tuple[TimeArray, VectorArray]:
        ...
    def __getitem__(self, idx: int | slice):
        """Convenience function to get the `(t, y)` points between given indices, or at a given index.

        Doesn't load new points.
        """
        if isinstance(idx, int):
            return self.ts[idx], self.ys[idx]
        elif isinstance(idx, slice):
            return to_time_array(self.ts[idx]), to_vector_array(self.ys[idx], self.y_shape)
        else:
            raise TypeError(f'DiscreteSolution indices must be integers or slices, not {type(idx).__name__}')

    def get_arrays_between(
        self,
        t_intervals: Sequence[Time] | TimeArray,
        closed_side: Literal["left", "right"] = "left",
        *,
        load: bool = True,
    ) -> tuple[list[TimeArray], list[VectorArray]]:
        """Get the `(ts, ys)` arrays between each pair of consecutive t_values.

        For example, if `self.ts == [0, 1, 2, 3, 4, 5]`, then calling this with `t_intervals = [1, 2, 5]`
        will return the list of `TimeArray`s `[[1], [2, 3, 4]]` and the list of corresponding `VectorArray`s.
        Essentially, `[1, 2, 5]` is interpreted as the half-open intervals `[1, 2), [2, 5)` (or as
        `(1, 2], (2, 5]` if `closed_side = "right"`).
        """
        if load:
            self.load_until(t_intervals[-1])
        ts_array = to_time_array(self.ts)
        ys_array = to_vector_array(self.ys, self.y_shape)
        indices = np.searchsorted(ts_array, to_time_array(t_intervals), side=closed_side)
        return (
            [ts_array[before:after] for before, after in pairwise(indices)],
            [ys_array[before:after] for before, after in pairwise(indices)],
        )

    def plot(
        self,
        t_start: Time,
        t_end: Time,
        xcoord: int = -1,
        ycoord: int | tuple[int, ...] = 0,
        *,
        ax: Axes | None = None,
        style: str = "-",
        load: bool = True,
        projection: Callable[[TimeArray, VectorArray], tuple[np.ndarray, np.ndarray]] | None = None,
        **plot_kwargs,
    ) -> Line2D:
        """Plot two coordinates of the discrete solution.

        Plot `(*y, t)[xcoord]` on the x-axis and `(*y, t)[ycoord]` on the y-axis for all y-values.
        """
        if ax is None:
            ax = plt.gca()

        t_start, t_end = to_time(t_start), to_time(t_end)
        (ts,), (ys,) = self.get_arrays_between([t_start, t_end], load=load)
        if projection is None:
            xaxis = ts if xcoord == -1 else ys[:, *pack(xcoord)]
            yaxis = ts if ycoord == -1 else ys[:, *pack(ycoord)]
        else:
            xaxis, yaxis = projection(ts, ys)

        if xaxis.ndim != 1 or yaxis.ndim != 1:
            if projection is None:
                raise ValueError(f"The given coordinates do not index a single element in an array with shape {self.y_shape}")
            raise ValueError("The projection did not return two one-dimensional arrays")
        return ax.plot(xaxis, yaxis, style, **plot_kwargs)[0]

    def plot_3d(
        self,
        t_start: Time,
        t_end: Time,
        xcoord: int | tuple[int, ...] = 0,
        ycoord: int | tuple[int, ...] = 1,
        zcoord: int | tuple[int, ...] = 2,
        *,
        ax: Axes3D,
        style: str = "-",
        load: bool = True,
        projection: Callable[[TimeArray, VectorArray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        **plot_kwargs,
    ) -> Line3D:
        """Plot three coordinates of the discrete solution."""
        t_start, t_end = to_time(t_start), to_time(t_end)
        (ts,), (ys,) = self.get_arrays_between([t_start, t_end], load=load)
        if projection is None:
            xaxis = ts if xcoord == -1 else ys[:, *pack(xcoord)]
            yaxis = ts if ycoord == -1 else ys[:, *pack(ycoord)]
            zaxis = ts if zcoord == -1 else ys[:, *pack(zcoord)]
        else:
            xaxis, yaxis, zaxis = projection(ts, ys)

        if xaxis.ndim != 1 or yaxis.ndim != 1 or zaxis.ndim != 1:
            if projection is None:
                raise ValueError(f"The given coordinates do not index a single element in an array with shape {self.y_shape}")
            raise ValueError("The projection did not return three one-dimensional arrays")
        return ax.plot(xaxis, yaxis, zaxis, style, **plot_kwargs)[0]  # type: ignore (matplotlib gets it wrong)

    def plot_time_steps(self, *, ax: Axes | None = None, style: str = '-', **plot_kwargs) -> Line2D:
        """Plot the lengths of the time steps taken over time.

        Only useful for adaptive-step solutions.
        """
        if ax is None:
            ax = plt.gca()
        xaxis = self.ts[:-1]
        yaxis = np.diff(to_time_array(self.ts))
        return ax.semilogy(xaxis, yaxis, style, **plot_kwargs)[0]
