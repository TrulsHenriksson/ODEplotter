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

type Vector = np.ndarray[tuple[int, ...], VectorType]
type VectorArray = np.ndarray[tuple[int, ...], VectorType]

type SolutionPoint = tuple[Time, Vector]
type DerivativeFunction = Callable[[Time, Vector], Vector]

# Types used in method_data.py
type WeightArray = np.ndarray[tuple[int], np.dtype[np.floating]]
type WeightMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]]

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
    """Cast a value to a `Vector`, which is a (>=1)-dimensional array of np.floating-point or complex values.

    The np.dtype is left unchanged if it is already a subdtype of `np.inexact`.

    If `copy` is true, the value is copied if necessary.
    """
    try:
        val_arr = np.asarray(val)
    except ValueError as e:
        raise ValueError(f"Could not cast value to array: {val}") from e
    except TypeError as e:
        raise TypeError(f"Cannot not cast object of type {type(val).__name__} to VectorArray") from e
    if not isinstance(val, (np.ndarray, list, tuple)) and not np.issubdtype(val_arr.dtype, np.number):
        raise TypeError(f"Cannot cast {type(val)} to Vector")
    if not np.issubdtype(val_arr.dtype, np.inexact):
        if np.can_cast(val_arr, np.float64, "safe"):
            val_arr = val_arr.astype(np.float64)
        else:
            raise ValueError(f"Cannot cast np.dtype {val_arr.dtype} to Vector")
    if val_arr.ndim < 1:
        raise ValueError(f"Cannot cast a zero-dimensional object to Vector: {val}")
    if val_arr.size == 0:
        raise ValueError("Cannot cast a 0-size array to Vector")
    return val_arr.copy() if copy and val is val_arr else val_arr

def is_vector(val: Any) -> TypeIs[Vector]:
    """Return whether `val` is a valid `Vector`.

    Valid `Vector`s are (>=1)-dimensional numpy arrays with `np.inexact` np.dtype.
    """
    return isinstance(val, np.ndarray) and val.ndim >= 1 and np.issubdtype(val.dtype, np.inexact)


def to_vector_array(val: Any, vector_shape: tuple[int, ...] | None = None, copy: bool = False) -> VectorArray:
    """Cast a value to `VectorArray`, which is a (>=2)-dimensional array of np.floating-point or complex values.

    The np.dtype is left unchanged if it is already a subdtype of `np.inexact`.

    When `vector_shape` is given, the shape of the vectors is checked to match. Casting
    an empty array requires that `expected_shape` is given.

    If `copy` is true, the value is copied if necessary.
    """
    # Cast to array
    try:
        val_arr = np.asarray(val)
    except ValueError as e:
        raise ValueError(f'Could not cast value to array: {val}') from e
    except TypeError as e:
        raise TypeError(f"Cannot not cast object of type {type(val).__name__} to VectorArray") from e
    # Normalize np.dtype
    if not isinstance(val, (np.ndarray, list, tuple)) and not np.issubdtype(val_arr.dtype, np.number):
        raise TypeError(f'Cannot cast {type(val)} to VectorArray')
    if not np.issubdtype(val_arr.dtype, np.inexact):
        if np.can_cast(val_arr, np.float64, 'safe'):
            val_arr = val_arr.astype(np.float64)
        else:
            raise ValueError(f'Cannot cast np.dtype {val_arr.dtype} to VectorArray')
    # Normalize shape
    if val_arr.ndim == 0:
        raise ValueError("Cannot cast a 0-dimensional array to VectorArray")
    elif val_arr.size == 0:
        # If given an empty array, e.g. np.array([[]]) or np.array([]), reshape it to zero vectors of the expected dimension
        if vector_shape is None:
            raise ValueError(f'expected_shape must be specified for an empty VectorArray')
        val_arr = val_arr.reshape((0, *vector_shape))
    elif vector_shape is None:
        val_arr = np.atleast_2d(val_arr)
    # Is it a single vector?
    elif val_arr.ndim == len(vector_shape):
        if val_arr.shape != vector_shape:
            raise ValueError(f"The shape of the vectors is {val_arr.shape}, expected {vector_shape}")
        val_arr = val_arr.reshape((1, *vector_shape))
    # Is it an array of vectors?
    elif val_arr.ndim == len(vector_shape) + 1:
        if val_arr.shape[1:] != vector_shape:
            raise ValueError(f"The shape of the vectors is {val_arr.shape[1:]}, expected {vector_shape}")
    else:
        raise ValueError(f"The shape is {val_arr.shape}, which is not a 1D array of vectors with shape {vector_shape}")
    return val_arr.copy() if copy and val is val_arr else val_arr

def is_vector_array(val: Any, vector_shape: tuple[int, ...] | None = None) -> TypeIs[VectorArray]:
    """Return whether `val` is a valid `VectorArray`.

    Valid `VectorArray`s are (>=2)-dimensional numpy arrays with `np.inexact` np.dtype,
    with its `shape[1:]` matching `vector_shape`, if given.
    """
    if not isinstance(val, np.ndarray) or val.ndim < 2 or not np.issubdtype(val.dtype, np.inexact):
        return False
    if vector_shape is None:
        return val.size != 0
    return val.shape[1:] == vector_shape


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
"""
