import ctypes

import numpy as np
import numpy.ctypeslib as npct

from ..myio import disable
from . import capi


class MPIBase:
    mpi = False
    initialized = False
    size = 1
    rank = 0
    master = False
    n_loc = 0
    n_loc_start = 0

def init_mpi_grid(n: int):
    MPIBase.n_loc = n
    return n

def _collect_grid_buffer(data: np.ndarray, recv: np.ndarray, n: int):
    recv[:] = data

def collect_grid_buffer(data: np.ndarray, n: int) -> np.ndarray:
    if MPIBase.size == 1:
        recv = data
    else:
        if MPIBase.master:
            recv = np.empty((n,n,n), dtype=np.float64)
        else:
            recv = np.empty((0,0,0), dtype=np.float64)
        capi.collect_grid_buffer(data, recv, n)
    # if MPIBase.master:
    #     np.set_printoptions(linewidth=3000)
    #     print(recv[0,0,:10])
    #     print(recv[60,0,:10])

    return recv

capi.register_function('init_mpi', ctypes.c_int, [], lambda: 1)
capi.register_function('init_mpi_grid', ctypes.c_int, [ctypes.c_int], init_mpi_grid)
capi.register_function('cleanup_mpi', None, [], lambda: None)
capi.register_function('get_n_start', ctypes.c_int, [], lambda: 0)
capi.register_function('get_n_loc', ctypes.c_int, [], lambda: MPIBase.n_loc)
capi.register_function('get_rank', ctypes.c_int, [], lambda: 0)
capi.register_function(
    'collect_grid_buffer', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_int
    ],
    _collect_grid_buffer
)

def init_mpi():
    if not MPIBase.initialized:
        size = capi.init_mpi()
        if size > 1:
            MPIBase.mpi = True

        MPIBase.size = size
        MPIBase.initialized = True

        MPIBase.rank = capi.get_rank()
        MPIBase.master = MPIBase.rank == 0
        if not MPIBase.master:
            disable()

capi.register_init(init_mpi)
# capi.register_finalize('cleanup_mpi')
