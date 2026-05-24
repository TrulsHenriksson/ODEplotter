import numpy as np
from math import sqrt
from timeit import timeit

from ODEplotter.utils.norms import two_norm, jit_two_norm


if __name__ == "__main__":
    # Benchmark performance of different implementations of 2-norms.

    rng = np.random.default_rng()
    vec_1d_real = rng.standard_normal(size=(1200,))
    vec_2d_real = np.resize(vec_1d_real, (30, 40))
    vec_1d_complex = rng.standard_normal(size=(1200,)) + rng.standard_normal(size=(1200,)) * 1j
    vec_2d_complex = np.resize(vec_1d_complex, (30, 40))

    norms = {
        "Sum-multiply": lambda vec: sqrt(np.sum((vec * vec.conj()).real)),
        "NumPy": lambda vec: np.linalg.norm(vec, "fro") if vec.ndim >= 2 else np.linalg.norm(vec),
        "two_norm": two_norm,
        "jit_two_norm": jit_two_norm,
    }
    vectors = {
        "Real 1D": vec_1d_real,
        "Real 2D": vec_2d_real,
        "Complex 1D": vec_1d_complex,
        "Complex 2D": vec_2d_complex,
    }

    # Warm up the jitted norm
    for vec in vectors.values():
        jit_two_norm(vec)

    times = np.empty((len(norms), len(vectors)))
    for i, (norm_name, norm) in enumerate(norms.items()):
        for j, (vec_name, vec) in enumerate(vectors.items()):
            times[i, j] = min(timeit("norm(vec)", globals=globals(), number=1000) for _ in range(10))

    print(" " * 15, *(f"{vec_name:^15}" for vec_name in vectors.keys()), sep=" ┃ ")
    print(*("━" * 15 for _ in range(len(vectors) + 1)), sep="━╋━")
    for norm_name, row in zip(norms.keys(), times):
        print(f"{norm_name:^15}", *(f"{time:^15.8f}" for time in row), sep=" ┃ ")
