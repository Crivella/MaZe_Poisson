import numpy as np


def init_fftw_omp():
    pass

def cleanup_fftw():
    pass

def init_rfft(n: int):
    pass

def rfft_solve(n: int, in_: np.ndarray, ig2: np.ndarray, out: np.ndarray):
    return np.fft.irfftn(np.fft.rfftn(in_) * ig2, out=out, s=in_.shape)
