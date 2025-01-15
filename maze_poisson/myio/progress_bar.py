from . import mpi

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

class ProgressBar:
    def __init__(self, n: int):
        self.n = n
        
    def __iter__(self):
        if mpi.master:
            return iter(tqdm(range(self.n)))
        return iter(range(self.n))