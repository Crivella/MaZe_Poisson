import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import capi
from .charges_fallbacks import update_charges

capi.register_function(
    'update_charges', ctypes.c_double, [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.int64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),   
    ],
    update_charges
)
