import numpy as np
from functools import wraps

from typing import overload, Callable
from .utils.types import *
from .utils.exceptions import ObstacleStopSolving

from .utils.root_finder import RootFinder


def sign(x) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)

def allow_single_inputs(func):
    @overload
    def wrapped_func(t: Time, y: Vector) -> Distance:
        ...
    @overload
    def wrapped_func(t: TimeArray, y: VectorArray) -> DistanceArray:
        ...
    @wraps(func)
    def wrapped_func(t, y):
        if is_time(t) and is_vector(y):
            return func(np.array((t,)), np.array((y,)))[0]
        return func(t, y)
    return wrapped_func


class Obstacle:
    """Class for detecting and handling collisions with obstacles."""

    dist_eps = 1e-14

    def __init__(
        self,
        distance_function: Callable[[TimeArray, VectorArray], DistanceArray],
        hit_function: Callable[[Time, Vector], Vector] | None = None,
        one_sided: bool = True,
    ):
        """Initialize an obstacle with boundary defined by distance_function.

        Arguments
        ---------
        distance_function : (TimeArray, VectorArray) -> DistanceArray
            Smooth (continuously differentiable) function that is 0
            on the boundary of the obstacle, positive outside it, and
            negative inside.
        hit_function : ((Time, Vector) -> Vector) or None (default: None)
            Function that returns a new y value when called with
            `hit_function(t_hit, y_hit)`, defining the behavior of hitting
            the obstacle. The default is to stop solving by raising
            `ObstacleStopSolving`.
        one_sided : bool (default: True)
            Whether a hit should register only when entering the obstacle.
        """
        self.distance_function = allow_single_inputs(distance_function)
        if hit_function is None:
            def stop(t: Time, y: Vector) -> Vector:
                raise ObstacleStopSolving(f"The obstacle was hit at t = {t}, y = {y}.")
            self.hit_function = stop
        else:
            self.hit_function = hit_function
        self.one_sided = one_sided

    def was_hit(self, t1: Time, y1: Vector, t2: Time, y2: Vector, *, dist_eps=dist_eps) -> bool:
        """Return whether the obstacle was hit between (t1, y1) and (t2, y2).

        For the purposes of hit detection, there are three regions:

        1. Middle, where `abs(distance) < dist_eps`,
        2. Outside, where `distance >= dist_eps`, and
        3. Inside, where `distance <= -dist_eps`.

        For two-sided obstacles, the inside and outside function the same. For one-sided,
        a hit only registers if t1 is outside and t2 is in the middle or inside. If
        t1 is in the middle, it never registers a hit.
        """
        dist1 = self.distance_function(t1, y1)
        dist2 = self.distance_function(t2, y2)
        if self.one_sided:
            # Was hit if t1 is outside and t2 is in the middle or inside
            return bool(dist2 <= dist_eps <= dist1)
        else:
            # Was hit if t1 is not in the middle, and t2 is in the middle or on the other side
            return bool(abs(dist1) >= dist_eps and (sign(dist2) != sign(dist1) or abs(dist2) <= dist_eps))

    def get_collision(
        self, t1: Time, y1: Vector, t2: Time, y2: Vector, *, max_iterations: int = 5, **newton_kwargs
    ) -> tuple[Time, Vector]:
        """Return the interpolated time and y value from when the obstacle was hit.

        Assumes the obstacle was hit between t1 and t2. Newton iteration will most likely diverge otherwise.
        """
        duration: Time = t2 - t1
        if duration == 0.0:
            return t1, y1
        elif duration < 0.0:
            raise ValueError('t1 must be before t2')
        # Interpolate linearly between (t1, y1) and (t2, y2)
        interpolate_y: Callable[[Time], Vector] = lambda t: (t2 - t) / duration * y1 + (t - t1) / duration * y2  # type: ignore
        # Return the distance at time t between (t1, y1) and (t2, y2)
        distance_from_time: Callable[[Time], Distance] = lambda t: self.distance_function(t, interpolate_y(t))
        # Find the time when the distance equals zero
        t_hit = RootFinder.scalar(
            distance_from_time, t1, t2, max_iterations=max_iterations, raise_if_exceeded=False, **newton_kwargs
        )
        y_hit = interpolate_y(t_hit)
        return t_hit, y_hit

    def get_collisions(
        self, ts: TimeArray, ys: VectorArray, *, dist_eps=dist_eps, newton_iterations: int | None = 5, **newton_kwargs
    ) -> tuple[TimeArray, VectorArray]:
        """Return the interpolated t and y values from all the collisions with the obstacle."""
        if len(ts) != len(ys):
            raise ValueError('ts and ys must have the same length')
        if len(ts) < 2:
            raise ValueError('ts and ys must have at least two elements')
        ts_ = to_time_array(ts)
        ys_ = to_vector_array(ys)
        # Get distances, first try for a vectorized approach
        try:
            distances = np.asarray(self.distance_function(ts_, ys_))
        except TypeError:
            distances = np.array([self.distance_function(t, y) for t, y in zip(ts_, ys_)])
        # Get an array of bools where collided[i] == True means there's a collision between ts[i] and ts[i+1].
        distances_before = distances[:-1]
        distances_after = distances[1:]
        if self.one_sided:
            collided = distances_after <= dist_eps <= distances_before
        else:
            collided = (abs(distances_before) >= dist_eps) & (
                (np.sign(distances_before) != np.sign(distances_after)) | (abs(distances_after) <= dist_eps)
            )
        if not np.any(collided):
            return np.empty(0, dtype=np.float64), np.empty((0, *ys_.shape[1:]), dtype=ys_.dtype)
        ts_before, ts_after = ts_[:-1][collided], ts_[1:][collided]
        ys_before, ys_after = ys_[:-1][collided], ys_[1:][collided]
        if newton_iterations is None:
            # If no interpolation was wanted, just return ts_after, ys_after
            return ts_after, ys_after
        # If interpolation is wanted, use self.get_collision on each interval
        collisions = [
            self.get_collision(t1, y1, t2, y2, max_iterations=newton_iterations, **newton_kwargs)
            for t1, y1, t2, y2 in zip(ts_before, ys_before, ts_after, ys_after)
        ]
        ts_hit, ys_hit = zip(*collisions)
        return np.array(ts_hit), np.array(ys_hit)


"""
TODO:
"""
