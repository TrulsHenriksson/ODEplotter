from numba import jit

from .types import VectorArray, WeightArray, Vector


@jit
def weighted_sum(vectors: VectorArray, weights: WeightArray) -> Vector:
    """Dot product along the first axis of `vectors`, equivalent to `vectors.T.dot(weights).T`."""
    return (vectors.T * weights).sum(axis=-1).T
