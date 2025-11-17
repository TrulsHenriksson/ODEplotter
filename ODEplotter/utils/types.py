import numpy as np

from typing import Any, Literal, Callable, Generator, TypeIs
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D


__all__ = [
    "Time",
    "TimeArray",
    "Vector",
    "VectorArray",
    "SolutionPoint",
    "DerivativeFunction",
    "WeightArray",
    "WeightMatrix",
    "UpdateFunction",
    "NoSolUpdateFunction",
    "InitFunction",
    "XYGetter",
    "XYZGetter",
    "Distance",
    "DistanceArray",
    "to_time",
    "is_time",
    "to_time_array",
    "is_time_array",
    "to_vector",
    "is_vector",
    "to_vector_array",
    "is_vector_array",
    "to_weight_array",
    "is_weight_array",
    "Artist",
    "Figure",
    "Axes",
    "Axes3D",
    "Line2D",
    "Line3D",
    "Quiver",
    "Text",
]


type TimeType = np.dtype[np.floating]
type VectorType = np.dtype[np.inexact]

# Types that are used everywhere
type Time = float | np.floating
type TimeArray = np.ndarray[tuple[int], TimeType]

type Vector = np.ndarray[tuple[int], VectorType]
type VectorArray = np.ndarray[tuple[int, int], VectorType]

type SolutionPoint = tuple[Time, Vector]
type DerivativeFunction = Callable[[Time, Vector], Vector]

# Types used in method_data.py
type WeightArray = np.ndarray[tuple[int], np.dtype[np.floating]]
type WeightMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]]

# Types used in solution_animator.py
type UpdateFunction = Callable[[TimeArray, VectorArray], tuple[Artist, ...]]
type NoSolUpdateFunction = Callable[[TimeArray], tuple[Artist, ...]]
type InitFunction = Callable[[Time], tuple[Artist, ...]]

type XYGetter = Callable[[TimeArray, VectorArray], tuple[np.ndarray, np.ndarray]]
type XYZGetter = Callable[[TimeArray, VectorArray], tuple[np.ndarray, np.ndarray, np.ndarray]]

# Types used in obstacle.py
type Distance = float | np.floating
type DistanceArray = np.ndarray[tuple[int], np.dtype[np.floating]]


def to_time(val: Any) -> Time:
    """Cast a value to `Time` (which is really just a float)."""
    try:
        return float(val)
    except TypeError:
        raise TypeError(f'Cannot cast {type(val)} to Time') from None
    except ValueError:
        raise ValueError(f'Cannot convert value to Time: {val}') from None

def is_time(val: Any) -> TypeIs[Time]:
    """Return whether `val` is a valid `Time`.

    Valid `Time`s are either `float` or `np.floating`.
    """
    return isinstance(val, (float, np.floating))


def to_time_array(val: Any, copy: bool = False) -> TimeArray:
    """Cast a value to `TimeArray`, which is a one-dimensional array of `np.float64` values.

    The np.dtype is left unchanged if it is already a subdtype of `np.floating`.
    """
    try:
        val_arr = np.atleast_1d(val)
    except ValueError as e:
        raise ValueError(f'Could not cast value to array: {val}') from e
    if not isinstance(val, (np.ndarray, list, tuple)) and not np.issubdtype(val_arr.dtype, np.number):
        raise TypeError(f'Cannot cast {type(val)} to TimeArray')
    if not np.issubdtype(val_arr.dtype, np.floating):
        if np.can_cast(val_arr, np.float64, 'safe'):
            val_arr = val_arr.astype(np.float64)
        else:
            raise ValueError(f'Cannot cast np.dtype {val_arr.dtype} to TimeArray')
    if val_arr.ndim != 1:
        raise ValueError('Object must be one-dimensional to cast to TimeArray')
    return val_arr.copy() if copy and val is val_arr else val_arr

def is_time_array(val: Any) -> TypeIs[TimeArray]:
    """Return whether `val` is a valid `TimeArray`.

    Valid `TimeArray`s are one-dimensional numpy arrays with `np.floating` np.dtype.
    """
    return isinstance(val, np.ndarray) and val.ndim == 1 and np.issubdtype(val.dtype, np.floating)


def to_vector(val: Any, copy: bool = False) -> Vector:
    """Cast a value to a `Vector`, which is a one-dimensional array of np.floating-point or complex values.

    The np.dtype is left unchanged if it is already a subdtype of `np.inexact`.

    If `copy` is true, the value is copied if necessary.
    """
    try:
        val_arr = np.atleast_1d(val)
    except ValueError as e:
        raise ValueError(f'Could not cast value to array: {val}') from e
    if not isinstance(val, (np.ndarray, list, tuple)) and not np.issubdtype(val_arr.dtype, np.number):
        raise TypeError(f'Cannot cast {type(val)} to Vector')
    if not np.issubdtype(val_arr.dtype, np.inexact):
        if np.can_cast(val_arr, np.float64, 'safe'):
            val_arr = val_arr.astype(np.float64)
        else:
            raise ValueError(f'Cannot cast np.dtype {val_arr.dtype} to Vector')
    if val_arr.ndim != 1:
        raise ValueError(f'Cannot cast a multidimensional object to Vector: {val}')
    if val_arr.size == 0:
        raise ValueError('Cannot cast a 0-dimensional array to Vector')
    return val_arr.copy() if copy and val is val_arr else val_arr

def is_vector(val: Any) -> TypeIs[Vector]:
    """Return whether `val` is a valid `Vector`.

    Valid `Vector`s are one-dimensional numpy arrays with `np.inexact` np.dtype.
    """
    return isinstance(val, np.ndarray) and val.ndim == 1 and np.issubdtype(val.dtype, np.inexact)


def to_vector_array(val: Any, expected_dimension: int | None = None, copy: bool = False) -> VectorArray:
    """Cast a value to `VectorArray`, which is a two-dimensional array of np.floating-point or complex values.

    The np.dtype is left unchanged if it is already a subdtype of `np.inexact`.

    When `expected_dimension` is given, the dimension of the vectors is checked to match. Casting
    an empty array requires that `expected_dimension` is given.

    If `copy` is true, the value is copied if necessary.
    """
    # Cast to array
    try:
        val_arr = np.asarray(val)
    except ValueError as e:
        raise ValueError(f'Could not cast value to array: {val}') from e
    # Normalize np.dtype
    if not isinstance(val, (np.ndarray, list, tuple)) and not np.issubdtype(val_arr.dtype, np.number):
        raise TypeError(f'Cannot cast {type(val)} to VectorArray')
    if not np.issubdtype(val_arr.dtype, np.inexact):
        if np.can_cast(val_arr, np.float64, 'safe'):
            val_arr = val_arr.astype(np.float64)
        else:
            raise ValueError(f'Cannot cast np.dtype {val_arr.dtype} to VectorArray')
    # Normalize shape
    match val_arr.shape:
        case ():
            raise ValueError(f'Cannot cast a 0-dimensional array to VectorArray')
        case (0,) | (0, 0) | (1, 0):
            # If given an empty array, e.g. np.array([[]]) or np.array([]), reshape it to zero vectors of the expected dimension
            if expected_dimension is None:
                raise ValueError(f'expected_dimension must be specified for an empty VectorArray')
            val_arr = val_arr.reshape((0, expected_dimension))
        case (dim,):
            if expected_dimension is not None and dim != expected_dimension:
                raise ValueError(f'The dimension of the vectors is {dim}, expected {expected_dimension}')
            val_arr = val_arr.reshape((1, dim))
        case (_, 0):
            raise ValueError('Cannot cast 0-dimensional Vectors to VectorArray')
        case (_, dim):
            if expected_dimension is not None and dim != expected_dimension:
                raise ValueError(f'The dimension of the vectors is {dim}, expected {expected_dimension}')
        case _:
            raise ValueError(f'Cannot cast an array with >2 dimensions to VectorArray')
    return val_arr.copy() if copy and val is val_arr else val_arr

def is_vector_array(val: Any, expected_dimension: int | None = None) -> TypeIs[VectorArray]:
    """Return whether `val` is a valid `VectorArray`.

    Valid `VectorArray`s are two-dimensional numpy arrays with `np.inexact` np.dtype,
    with second dimension matching `expected_dimension`, if given.
    """
    if not isinstance(val, np.ndarray) or val.ndim != 2 or not np.issubdtype(val.dtype, np.inexact):
        return False
    if expected_dimension is None:
        return val.shape[1] != 0
    return val.shape[1] == expected_dimension


def to_weight_array(val: Any, copy: bool = False) -> WeightArray:
    """Cast a value to `WeightArray`, which is a 1D array of `np.float64` values.

    The dtype is left unchanged if it is already a subdtype of `np.floating`.
    """
    try:
        val_arr = np.asarray(val)
    except ValueError as e:
        raise ValueError(f'Could not cast value to array: {val}') from e
    if not isinstance(val, (np.ndarray, list, tuple)) and not np.issubdtype(val_arr.dtype, np.number):
        raise TypeError(f'Cannot cast {type(val)} to WeightArray')
    if not np.issubdtype(val_arr.dtype, np.floating):
        if np.can_cast(val_arr, np.float64, 'safe'):
            val_arr = val_arr.astype(np.float64)
        else:
            raise ValueError(f'Cannot cast np.dtype {val_arr.dtype} to WeightArray')
    if val_arr.ndim != 1:
        raise ValueError('Object must be one-dimensional to cast to WeightArray')
    return val_arr.copy() if copy and val is val_arr else val_arr

def is_weight_array(val: Any) -> TypeIs[WeightArray]:
    """Return whether `val` is a valid `WeightArray`.

    Valid `WeightArray`s are 1D arrays of `np.float64` values.
    """
    return isinstance(val, np.ndarray) and val.ndim == 1 and np.issubdtype(val.dtype, np.floating)

"""
TODO:
- Simplify to_vector_array shape logic
- Disallow nan as Time? Perchance.
"""
