import ctypes
from pathlib import Path

import numpy as np


LIBRARY = Path(__file__).parents[1] / "maze_poisson" / "libmaze_poisson.so"


def _residual_norm(lib, n: int, discretization: int) -> float:
    n2 = n * n
    n3 = n * n2
    h = 1.0 / n
    raw_u = np.zeros((n + 4) * n2, dtype=np.float64)
    raw_f = np.zeros_like(raw_u)
    u = raw_u[2 * n2:2 * n2 + n3]
    f = raw_f[2 * n2:2 * n2 + n3]

    x = np.arange(n, dtype=np.float64) / n
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    values = np.sin(2 * np.pi * xx) * np.cos(4 * np.pi * yy) * np.sin(2 * np.pi * zz)
    lap_exact = -6.0 * (2 * np.pi) ** 2 * values
    u[:] = values.ravel()
    f[:] = lap_exact.ravel()
    Au = np.empty(n3, dtype=np.float64)
    Bf = np.empty(n3, dtype=np.float64)

    ptr = ctypes.POINTER(ctypes.c_double)
    lib.laplace_set_discretization(discretization)
    lib.laplace_filter(u.ctypes.data_as(ptr), Au.ctypes.data_as(ptr), n, n)
    lib.laplace_filter_rhs(f.ctypes.data_as(ptr), Bf.ctypes.data_as(ptr), n, n)
    return float(np.max(np.abs(Au - h * h * Bf)))


def _solution_error(lib, n: int, discretization: int) -> tuple[float, int]:
    n2 = n * n
    n3 = n * n2
    h = 1.0 / n
    raw_source = np.zeros((n + 4) * n2, dtype=np.float64)
    raw_rhs = np.zeros_like(raw_source)
    raw_solution = np.zeros_like(raw_source)
    source = raw_source[2 * n2:2 * n2 + n3]
    rhs = raw_rhs[2 * n2:2 * n2 + n3]
    solution = raw_solution[2 * n2:2 * n2 + n3]

    x = np.arange(n, dtype=np.float64) / n
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    exact = np.sin(2 * np.pi * xx) * np.cos(4 * np.pi * yy) * np.sin(2 * np.pi * zz)
    source[:] = (-6.0 * (2 * np.pi) ** 2 * exact).ravel()

    ptr = ctypes.POINTER(ctypes.c_double)
    lib.laplace_set_discretization(discretization)
    lib.laplace_filter_rhs(
        source.ctypes.data_as(ptr), rhs.ctypes.data_as(ptr), n, n
    )
    rhs *= h * h
    iterations = lib.multigrid_solve(
        1.0e-10, rhs.ctypes.data_as(ptr), solution.ctypes.data_as(ptr), n, n, 0
    )
    numerical = solution.reshape((n, n, n))
    numerical -= numerical.mean()
    return float(np.sqrt(np.mean((numerical - exact) ** 2))), int(iterations)


def _gradient_error(lib, n: int, gradient_order: int) -> float:
    n2 = n * n
    n3 = n * n2
    h = 1.0 / n
    raw_phi = np.zeros((n + 4) * n2, dtype=np.float64)
    phi = raw_phi[2 * n2:2 * n2 + n3]

    x = np.arange(n, dtype=np.float64) / n
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    phi[:] = (
        np.sin(2 * np.pi * xx)
        + np.cos(4 * np.pi * yy)
        + np.sin(6 * np.pi * zz)
    ).ravel()

    # The same physical grid node is available on both N=16 and N=32.
    scale = n // 16
    ix, iy, iz = 3 * scale, 5 * scale, 7 * scale
    pos = np.array([[ix * h, iy * h, iz * h]], dtype=np.float64)
    neighbors = np.array([ix, iy, iz], dtype=np.int64)
    charges = np.ones(1, dtype=np.float64)
    forces = np.zeros(3, dtype=np.float64)
    exact_field = np.array([
        -2 * np.pi * np.cos(2 * np.pi * pos[0, 0]),
        4 * np.pi * np.sin(4 * np.pi * pos[0, 1]),
        -6 * np.pi * np.cos(6 * np.pi * pos[0, 2]),
    ])

    ptr = ctypes.POINTER(ctypes.c_double)
    long_ptr = ctypes.POINTER(ctypes.c_long)
    lib.compute_force_fd(
        n, 1, n, 0, h, 1,
        phi.ctypes.data_as(ptr), neighbors.ctypes.data_as(long_ptr),
        charges.ctypes.data_as(ptr), pos.ctypes.data_as(ptr),
        forces.ctypes.data_as(ptr), ctypes.cast(lib.spread_cic, ctypes.c_void_p),
        gradient_order,
    )
    return float(np.linalg.norm(forces - exact_field))


def test_discretization_orders():
    lib = ctypes.CDLL(str(LIBRARY))
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.laplace_set_discretization.argtypes = [ctypes.c_int]
    lib.laplace_filter.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int]
    lib.laplace_filter_rhs.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int]
    lib.multigrid_solve.argtypes = [
        ctypes.c_double, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.multigrid_solve.restype = ctypes.c_int
    lib.compute_force_fd.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_int, ptr, ctypes.POINTER(ctypes.c_long),
        ptr, ptr, ptr, ctypes.c_void_p, ctypes.c_int,
    ]
    lib.compute_force_fd.restype = ctypes.c_double
    lib.set_print_convergence.argtypes = [ctypes.c_int]
    lib.multigrid_set_krylov.argtypes = [ctypes.c_int]
    lib.init_mpi()
    lib.set_print_convergence(0)
    try:
        standard_ratio = _residual_norm(lib, 16, 0) / _residual_norm(lib, 32, 0)
        mehrstellen4_ratio = _residual_norm(lib, 16, 1) / _residual_norm(lib, 32, 1)
        standard_error_16, standard_cycles = _solution_error(lib, 16, 0)
        standard_error_32 = _solution_error(lib, 32, 0)[0]
        standard_solution_ratio = standard_error_16 / standard_error_32
        lib.multigrid_set_krylov(1)
        standard_krylov_error, standard_krylov_iterations = _solution_error(lib, 16, 0)
        lib.multigrid_set_krylov(0)
        mehrstellen4_solution_ratio = _solution_error(lib, 16, 1)[0] / _solution_error(lib, 32, 1)[0]
        gradient2_ratio = _gradient_error(lib, 16, 2) / _gradient_error(lib, 32, 2)
        gradient4_ratio = _gradient_error(lib, 16, 4) / _gradient_error(lib, 32, 4)
    finally:
        lib.cleanup_mpi()

    # Dimensionless equation residuals scale as h^4 and h^6 respectively,
    # corresponding to second- and fourth-order solutions after division by h^2.
    assert 12.0 < standard_ratio < 20.0
    assert 40.0 < mehrstellen4_ratio < 90.0
    assert 3.5 < standard_solution_ratio < 4.7
    assert 12.0 < mehrstellen4_solution_ratio < 20.0
    assert np.isclose(standard_krylov_error, standard_error_16, rtol=1.0e-5)
    assert 0 < standard_krylov_iterations <= standard_cycles
    assert 3.5 < gradient2_ratio < 4.5
    assert 12.0 < gradient4_ratio < 20.0
