import numpy as np
from numba import jit
from timeit import timeit

from ODEplotter.utils.weighted_sum import weighted_sum


if __name__ == "__main__":
    # Benchmark performance of different implementations of weighted_sum.

    rng = np.random.default_rng(1)
    vecs_1d_real = rng.standard_normal(size=(30, 40))
    vecs_1d_complex = rng.standard_normal(size=(30, 40)) + rng.standard_normal(size=(30, 40)) * 1j
    vecs_2d_real = rng.standard_normal(size=(30, 8, 5))
    vecs_2d_complex = rng.standard_normal(size=(30, 8, 5)) + rng.standard_normal(size=(30, 8, 5)) * 1j
    weights = rng.uniform(0.0, 1.0, size=(30,))

    functions = {
        "jitable": lambda vs, ws: (vs.T * ws).sum(axis=-1).T,
        "jited": jit(lambda vs, ws: (vs.T * ws).sum(axis=-1).T),
        "Standard": weighted_sum,
    }
    vector_arrays = {
        "Real 1D": vecs_1d_real,
        "Real 2D": vecs_2d_real,
        "Complex 1D": vecs_1d_complex,
        "Complex 2D": vecs_2d_complex,
    }

    # Warm up the jitted function
    for vs in vector_arrays.values():
        functions["Standard"](vs, weights)

    times = np.empty((len(functions), len(vector_arrays)))
    for i, (function_name, weight_sum) in enumerate(functions.items()):
        for j, (vec_name, vs) in enumerate(vector_arrays.items()):
            times[i, j] = min(timeit("weight_sum(vs, weights)", globals=globals(), number=1000) for _ in range(10))

    print(" " * 15, *(f"{vec_name:^15}" for vec_name in vector_arrays.keys()), sep=" ┃ ")
    print(*("━" * 15 for _ in range(len(vector_arrays) + 1)), sep="━╋━")
    for function_name, row in zip(functions.keys(), times):
        print(f"{function_name:^15}", *(f"{time:^15.8f}" for time in row), sep=" ┃ ")
