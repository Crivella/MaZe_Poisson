import ctypes
import os
import signal

from ..mpi import MPIBase
from ..myio.loggers import logger

mpi = MPIBase()

try:
    library = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), '..', 'libmaze_poisson.so'))
except:
    logger.warning("C_API: Could not load shared library.")
    library = None
else:
    logger.info("C_API: Loaded successfully")

try:
    c_get_omp_info = library.get_omp_info
    c_get_omp_info.restype = ctypes.c_int
    c_get_omp_info.argtypes = []
except:
    c_get_omp_info = lambda: 0


from .fftw import cleanup_fftw, init_fftw_omp, init_rfft, rfft_solve
from .forces import c_compute_force_fd, c_compute_tf_forces
from .laplace import c_conj_grad, c_laplace

# from .mympy import c_get_mpi_data, c_init_mpi

# c_init_mpi(100, mpi.comm_address)
# data = c_get_mpi_data()
# print(data)


__all__ = [
    'c_compute_force_fd', 'c_compute_tf_forces', 'c_conj_grad', 'c_laplace',
    'cleanup_fftw', 'init_fftw_omp', 'init_rfft', 'rfft_solve'
]

# Enable Ctrl-C to interrupt the C code
signal.signal(signal.SIGINT, signal.SIG_DFL)

if library:
    # Check if OpenMP is enabled
    num_threads = c_get_omp_info()
    if num_threads > 0:
        logger.info("C_API: Number of OpenMP threads: %d", c_get_omp_info())
    else:
        logger.warning("C_API: OpenMP not enabled")
