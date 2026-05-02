import numpy as np

from typing import Callable
from .types import Time, Vector, VectorArray, WeightArray


type PredictorMethod = Callable[[Time, Vector, VectorArray], Vector]


class Predictors:
    """Class for predicting the next point for implicit methods."""

    @staticmethod
    def adams_bashforth(weights: WeightArray) -> PredictorMethod:
        def method(h: Time, y: Vector, diffs: VectorArray) -> Vector:
            return y + h * (diffs.T.dot(weights)).T
        return method

    last: PredictorMethod = lambda h, y, diffs: y

    AB1 = adams_bashforth(np.array([1.0]))

    AB2 = adams_bashforth(np.array([1.5, -0.5]))

    AB3 = adams_bashforth(np.array([23, -16, 9]) / 12)

    AB4 = adams_bashforth(np.array([55, -59, 37, -9]) / 24)

    AB5 = adams_bashforth(np.array([1901, -2774, 2616, -1274, 251]) / 720)

    methods: dict[str, PredictorMethod] = {
        "AB0": last,
        "AB1": AB1,
        "AB2": AB2,
        "AB3": AB3,
        "AB4": AB4,
        "AB5": AB5,
        "last": last,
        "euler": AB1,
    }


"""
TODO:
- Is this even necessary? Should the predictors be given to the AM methods directly?
"""
