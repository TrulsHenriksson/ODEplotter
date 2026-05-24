from .types import VectorArray, WeightArray, Vector


def weighted_sum(vectors: VectorArray, weights: WeightArray) -> Vector:
    """Dot product along the first axis of `vectors`."""
    return vectors.T.dot(weights).T
