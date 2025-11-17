from numba import jit
import numpy as np

from typing import Generator
from ...utils.types import *

from ..solution_method import SolutionMethod



def runge_kutta(
    derivative: DerivativeFunction,
    t: Time,
    y: Vector,
    h: Time,
    stages: int,
    nodes: TimeArray,
    weights: WeightArray,
    matrix: WeightMatrix,
) -> Generator[SolutionPoint]:
    derivatives = np.zeros((stages, len(y)), dtype=y.dtype)
    while True:
        yield t, y.copy()
        derivatives[0] = derivative(t, y)
        for i in range(1, stages):
            derivatives[i] = derivative(
                t + h * nodes[i],
                y + h * matrix[i, :i].dot(derivatives[:i]),
            )
        t += h
        y += h * weights.dot(derivatives)


class RungeKutta(SolutionMethod):
    method = staticmethod(runge_kutta)
    compiled_method = staticmethod(jit(runge_kutta))

    stages: int
    nodes: TimeArray
    weights: WeightArray
    matrix: WeightMatrix

    def __init__(self, nodes, weights, matrix):
        """Method which takes sub-steps in every step, each decided by a weighted sum of the previous derivatives."""
        super().__init__()
        self._nodes = nodes
        self._weights = weights
        self._matrix = matrix

    def _validate(self, complete_validation=True):
        """Perform basic checks on the nodes, weights and matrix."""
        if self.validated:
            return

        self.stages = len(self._nodes)
        self.nodes = self._validate_nodes()
        self.weights = self._validate_weights()
        self.matrix = self._validate_matrix()

        del self._nodes, self._weights, self._matrix
        self.validated = complete_validation

    def _validate_nodes(self) -> TimeArray:
        nodes = to_time_array(self._nodes)
        assert nodes.shape == (self.stages,), "nodes must be a 1D array"
        assert nodes[0] == 0.0, "The first node must be 0"
        return nodes

    def _validate_weights(self) -> WeightArray:
        weights = to_weight_array(self._weights)
        assert weights.shape == (self.stages,), f"weights must have shape ({self.stages},)"
        assert abs(weights.sum() - 1.0) < 1e-14, f"weights must sum to 1, not {weights.sum()}"
        return weights

    def _validate_matrix(self) -> WeightMatrix:
        matrix = np.asarray(self._matrix, dtype=np.float64)
        assert matrix.shape == (self.stages, self.stages), f"matrix must have shape ({self.stages}, {self.stages})"
        assert np.allclose(
            matrix.sum(axis=1), self.nodes
        ), f"The row sums of the matrix do not equal the nodes. Difference: {matrix.sum(axis=1) - self.nodes}"
        return matrix

    def _prepare_arguments(self, derivative: DerivativeFunction, t0: Time, y0: Vector, h: Time, use_jit: bool):
        return (derivative, t0, y0, to_time(h), self.stages, self.nodes, self.weights, self.matrix)
