import atexit
import ctypes
import os
import signal

from ..myio import logger


class CAPI:
    def __init__(self):
        super().__init__()
        self.library = None

        self.functions = {}
        self.toinit = []
        self.tofina = []
        self.toregister = []

        atexit.register(self.finalize)

    def __getattr__(self, name):
        if name in self.functions:
            return self.functions[name]
        return super().__getattr__(name)

    def initialize(self):
        if self.library is not None:
            return
        try:
            self.library = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), '..', 'libmaze_poisson.so'))
        except:
            self.library = None
            logger.warning("C_API: Could not load shared library.")
        else:
            logger.debug("C_API: Loaded successfully")

        for fname, restype, argtypes, fallback in self.toregister:
            try:
                func = getattr(self.library, fname)
                func.restype = restype
                func.argtypes = argtypes
                self.functions[fname] = func
                logger.debug(f"C_API: Registered function {fname}")
            except:
                logger.warning(f"C_API: Could not register function {fname}, using fallback")
                self.functions[fname] = fallback

        for fname, args, kwargs in self.toinit:
            if isinstance(fname, str):
                func = self.functions[fname]
            else:
                func = fname
            func(*args, **kwargs)


    def finalize(self):
        for fname, args, kwargs in self.tofina:
            if isinstance(fname, str):
                func = self.functions[fname]
            else:
                func = fname
            func(*args, **kwargs)

    def register_function(self, name, restype, argtypes, fallback = None):
        self.toregister.append((name, restype, argtypes, fallback))

    def register_init(self, fname, args = None, kwargs = None):
        self.toinit.append((fname, args or [], kwargs or {}))

    def register_finalize(self, fname, args = None, kwargs = None):
        self.tofina.append((fname, args or [], kwargs or {}))


capi = CAPI()
capi.register_function('get_omp_info', ctypes.c_int, [], lambda: 0)
capi.register_init('get_omp_info')
capi.register_init(signal.signal, (signal.SIGINT, signal.SIG_DFL))

# Needed to register functions from other modules
from . import charges, fftw, forces, laplace, mympi

__all__ = ['capi']

capi.initialize()
