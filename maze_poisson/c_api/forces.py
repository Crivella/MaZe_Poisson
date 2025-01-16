# Interface and fallbacks to forces.c
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import library, logger

try:
    # double compute_force_fd(int n_grid, int n_p, double h, double *phi, double *q, long int *neighbors, double *forces)
    c_compute_force_fd = library.compute_force_fd
    c_compute_force_fd.restype = ctypes.c_double
    c_compute_force_fd.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.int64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ]

    # double compute_tf_forces(int n_p, double L, double *pos, double B, double *params, double r_cut, double *forces)
    c_compute_tf_forces = library.compute_tf_forces
    c_compute_tf_forces.restype = ctypes.c_double
    c_compute_tf_forces.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ]
except:
    from .forces_fallbacks import c_compute_force_fd, c_compute_tf_forces
    logger.warning("C_API: compute_force_fd not available. Using Python instead.")
else:
    logger.info("C_API: compute_force_fd loaded successfully")
