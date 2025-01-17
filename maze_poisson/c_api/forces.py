# Interface and fallbacks to forces.c
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import capi
from .forces_fallbacks import compute_force_fd, compute_tf_forces

capi.register_function(
    'compute_force_fd', ctypes.c_double, [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.int64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
    compute_force_fd
)

capi.register_function(
    'compute_tf_forces', ctypes.c_double, [
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ],
    compute_tf_forces
)
