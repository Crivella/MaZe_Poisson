import ctypes

from ..myio import disable_io
from . import capi

IO_NODE = 0

capi.register_function('init_mpi', ctypes.c_int, [], lambda: 1)
capi.register_function('get_rank', ctypes.c_int, [], lambda: 0)

def check_mpi():
    size = capi.init_mpi()
    rank = capi.get_rank()
    if rank != IO_NODE:
        disable_io()

capi.register_init(check_mpi)
