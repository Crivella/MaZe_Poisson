import atexit
import ctypes

from . import library

try:
    # int init_mpi(int n, void *comm_ptr)
    _c_init_mpi = library.init_mpi
    _c_init_mpi.restype = ctypes.c_int
    _c_init_mpi.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    # void cleanup_mpi();
    c_cleanup_mpi = library.cleanup_mpi
    c_cleanup_mpi.restype = None
    c_cleanup_mpi.argtypes = []
except:
    _c_init_mpi = lambda *args: None
    c_cleanup_mpi = lambda: None 

initialized = False
def c_init_mpi(*args):
    global initialized
    if not initialized:
        _c_init_mpi(*args)
        initialized = True
        atexit.register(c_cleanup_mpi)
