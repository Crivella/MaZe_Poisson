import atexit
import ctypes
import os
import signal

from ..myio import logger


class SafeCallable:
    def __init__(self, func, name):
        self.func = func
        self.name = name

    def __call__(self, *args, **kwargs):
        if self.func is None:
            logger.error(f"Function {self.name} not available")
            exit()
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            exit()

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
        return super().__getattribute__(name)

    def initialize(self):
        if self.library is not None:
            return
        self.path = os.path.join(os.path.dirname(__file__), '..', 'libmaze_poisson.so')
        try:
            self.library = ctypes.cdll.LoadLibrary(self.path)
        except Exception as e:
            self.library = None
            logger.warning(f"C_API: Could not load shared library. {self.path}")
            logger.warning(f"C_API: {e}")
        else:
            logger.debug("C_API: Loaded successfully")

        for fname, restype, argtypes, fallback in self.toregister:
            try:
                func = getattr(self.library, fname)
                func.restype = restype
                func.argtypes = argtypes
                self.functions[fname] = SafeCallable(func, fname)
                logger.debug(f"C_API: Registered function {fname}")
            except Exception as e:
                logger.warning(f"C_API: Could not register function {fname}, using fallback")
                logger.warning(f"C_API: {e}")
                self.functions[fname] = SafeCallable(fallback, fname)

        for fname, args, kwargs in self.toinit:
            if isinstance(fname, str):
                func = self.functions[fname]
            else:
                func = fname
            func(*args, **kwargs)


    def finalize(self):
        for fname, args, kwargs in self.tofina:
            if isinstance(fname, str):
                try:
                    func = self.functions[fname]
                except:
                    continue
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
capi.register_init(signal.signal, (signal.SIGINT, signal.SIG_DFL))

# Needed to register functions from other modules
# from . import charges, fftw, forces, laplace, mympi, solver
from . import mympi, solver

__all__ = ['capi']

# capi.initialize()
