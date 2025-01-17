import atexit
import ctypes

import numpy as np
import numpy.ctypeslib as npct

from ..myio import disable
from . import library


class MPIBase:
    mpi = False
    initialized = False
    size = 1
    rank = 0
    master = False
    n_loc = 0
    n_loc_start = 0

try:
    # int init_mpi();
    _c_init_mpi = library.init_mpi
    _c_init_mpi.restype = ctypes.c_int
    _c_init_mpi.argtypes = []

    # int init_mpi_grid(int n);
    c_init_mpi_grid = library.init_mpi_grid
    c_init_mpi_grid.restype = ctypes.c_int
    c_init_mpi_grid.argtypes = [ctypes.c_int]

    # void cleanup_mpi();
    c_cleanup_mpi = library.cleanup_mpi
    c_cleanup_mpi.restype = None
    c_cleanup_mpi.argtypes = []

    # int get_n_start();
    c_get_n_start = library.get_n_start
    c_get_n_start.restype = ctypes.c_int
    c_get_n_start.argtypes = []

    # # int get_n_loc();
    c_get_n_loc = library.get_n_loc
    c_get_n_loc.restype = ctypes.c_int
    c_get_n_loc.argtypes = []

    # int get_rank();
    c_get_rank = library.get_rank
    c_get_rank.restype = ctypes.c_int

    # void collect_grid_buffer(double *data, double *recv, int n)
    _c_collect_grid_buffer = library.collect_grid_buffer
    _c_collect_grid_buffer.restype = None
    _c_collect_grid_buffer.argtypes = [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_int
    ]
    def c_collect_grid_buffer(data: np.ndarray, n: int) -> np.ndarray:
        if MPIBase.size == 1:
            recv = data
        else:
            if MPIBase.master:
                recv = np.empty((n,n,n), dtype=np.float64)
            else:
                recv = np.empty((0,0,0), dtype=np.float64)
            _c_collect_grid_buffer(data, recv, n)
        # if MPIBase.master:
        #     np.set_printoptions(linewidth=3000)
        #     print(recv[0,0,:10])
        #     print(recv[60,0,:10])

        return recv

except:
    def _c_init_mpi() -> int:
        return 1
    def c_init_mpi_grid(n: int) -> None:
        MPIBase.n_loc = n
    c_init_mpi_grid = lambda n: n
    
    _c_init_mpi = lambda *args: 1
    c_get_rank = lambda: 0
    c_cleanup_mpi = lambda: None
    c_get_n_start = lambda: 0
    def c_get_n_loc() -> int:
        return MPIBase.n_loc
    c_collect_grid_buffer = lambda data, n: data

def c_init_mpi():
    if not MPIBase.initialized:
        size = _c_init_mpi()
        if size > 1:
            MPIBase.mpi = True

        MPIBase.size = size
        MPIBase.initialized = True

        MPIBase.rank = c_get_rank()
        MPIBase.master = MPIBase.rank == 0
        if not MPIBase.master:
            disable()

        atexit.register(c_cleanup_mpi)

c_init_mpi()
