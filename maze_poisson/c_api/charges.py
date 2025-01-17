import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import library
from .mympi import c_get_n_loc, c_get_n_start

try:
    # double update_charges(
    #     int n_grid, int n_p, double h,
    #     double *pos, long int *neighbors, double *charges, double *q
    # )
    c_update_charges = library.update_charges
    c_update_charges.restype = ctypes.c_double
    c_update_charges.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.int64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ]
except:
    def g(x, L, h):
        x = np.abs(x - L * np.rint(x / L))
        return 1 - x / h

    def c_update_charges(n_grid, n_p, h, pos, neighbors, charges, q):
        """Update the charges on the grid.
        
        Args:
            particles (Particles): Particles object.

        Returns:
            float: Total charge contribution.
        """
        n_loc = c_get_n_loc()
        n_start = c_get_n_start()

        L = n_grid * h
        q.fill(0)
        diff = pos[:, np.newaxis, :] - neighbors * h  # shape (N_p, 8, 3)

        indices = tuple(neighbors.reshape(-1, 3).T)

        updates = (charges[:, np.newaxis] * np.prod(g(diff, L, h), axis=2)).flatten()
        for i,j,k,upd in zip(*indices, updates):
            i -= n_start
            if 0 <= i < n_loc:
                q[i, j, k] += upd
  
        q_tot = np.sum(updates)

        return q_tot
